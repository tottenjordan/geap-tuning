"""Design-of-experiments (DOE) orchestration for SFT tuning sweeps.

Declares a hyperparameter sweep once and runs it as a unit: expand a grid into
one :class:`RunSpec` per point (each with a deterministic display name), launch
each idempotently (reusing a prior job with the same display name — cost
control), evaluate offline, and log to Vertex AI Experiments so the runs can be
compared side by side. The pure helpers (:func:`expand_grid`,
:func:`run_spec_slug`, :func:`build_run_specs`, :func:`select_best_run`,
:func:`aggregate_results`) carry no SDK and are unit-tested directly; the
:func:`run_sweep` driver takes injectable seams so it too is testable without
live GCP.

Scope: **SFT only** (dispatches :func:`geap_tuning.sft.tune.launch_sft_job`).
DPO/RLFT sweeps are a documented follow-up (see
``docs/notes/doe-and-visualization.md``). The tune call stays on the
``google.genai`` path; the only SDK mix is the Experiments logging, which is
routed through :mod:`geap_tuning.experiments` (the one sanctioned mix point).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

from geap_tuning.experiments import log_summary_metrics, track_run
from geap_tuning.jobs import (
    checkpoint_endpoint,
    find_tuning_job_by_display_name,
    list_checkpoints,
    tuned_endpoint,
    wait_for_tuning_job,
)
from geap_tuning.sft.tune import launch_sft_job

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_DEFAULT_METRICS = ("accuracy", "macro_f1")


@dataclass(frozen=True)
class SweepConfig:
    """Declarative full-factorial SFT hyperparameter sweep.

    ``grid`` maps a :func:`launch_sft_job` keyword (``epochs``, ``adapter_size``,
    ``learning_rate_multiplier``) to the values to cross; ``fixed`` supplies
    constant kwargs applied to every run. ``name`` prefixes each run's display
    name (the idempotency key) and is a natural experiment name.
    """

    name: str
    base_model: str = "gemini-2.5-flash-lite"
    grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    fixed: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    """One resolved grid point: the inputs for a single tuning run."""

    name: str
    display_name: str
    base_model: str
    params: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    """A completed run: its spec, resolved job/endpoint, and offline metrics."""

    spec: RunSpec
    job_name: str
    endpoint: str
    metrics: dict[str, Any]
    reused: bool


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Return the full-factorial cross-product of ``grid`` as one dict per point.

    Iterates keys in sorted order for deterministic output; an empty grid yields
    a single empty point ``[{}]``.
    """
    if not grid:
        return [{}]
    keys = sorted(grid)
    value_lists = [list(grid[key]) for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


def run_spec_slug(point: Mapping[str, Any]) -> str:
    """Return a deterministic, display-name-safe slug for a grid ``point``.

    Keys are sorted; each ``key<value>`` pair is joined by ``-``; any character
    that is not alphanumeric or ``-`` (e.g. the ``.`` in a float) becomes ``_``.
    An empty point slugs to ``"default"``.
    """
    if not point:
        return "default"
    raw = "-".join(f"{key}{point[key]}" for key in sorted(point))
    return "".join(char if char.isalnum() or char == "-" else "_" for char in raw)


def build_run_specs(sweep: SweepConfig) -> list[RunSpec]:
    """Expand ``sweep`` into one :class:`RunSpec` per grid point.

    Each run's ``params`` is ``sweep.fixed`` overlaid with the grid point (grid
    wins on conflict), and its ``display_name`` is ``geap-doe-{sweep.name}-{slug}``.
    """
    specs: list[RunSpec] = []
    for point in expand_grid(sweep.grid):
        slug = run_spec_slug(point)
        specs.append(
            RunSpec(
                name=slug,
                display_name=f"geap-doe-{sweep.name}-{slug}",
                base_model=sweep.base_model,
                params={**sweep.fixed, **point},
            )
        )
    return specs


def select_best_run(results: Mapping[str, dict[str, Any]], *, metric: str = "accuracy") -> str:
    """Return the run name with the highest ``metric`` (tie-break by name).

    ``results`` maps a run name to its metrics dict. Raises ``ValueError`` when
    empty. Generalizes ``sft_vision.evaluate.select_best_experiment`` to any
    metric key.
    """
    if not results:
        msg = "No run results to select from"
        raise ValueError(msg)
    return max(sorted(results), key=lambda name: results[name][metric])


def aggregate_results(
    results: Sequence[RunResult],
    *,
    metrics: Sequence[str] = _DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    """Flatten runs into comparable rows for a table or plots.

    One row per run: ``{"run", "base_model", **spec.params, **selected metrics}``.
    A metric absent from a run's ``metrics`` is simply omitted for that row.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"run": result.spec.name, "base_model": result.spec.base_model}
        row.update(result.spec.params)
        for metric in metrics:
            if metric in result.metrics:
                row[metric] = result.metrics[metric]
        rows.append(row)
    return rows


def _default_launch(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    spec: RunSpec,
    train_uri: str,
    val_uri: str | None,
) -> Any:  # noqa: ANN401 - returns the SDK job object
    """Launch ``spec`` as an SFT job, exporting all checkpoints for curve eval."""
    return launch_sft_job(
        client,
        train_uri=train_uri,
        val_uri=val_uri,
        display_name=spec.display_name,
        base_model=spec.base_model,
        export_last_checkpoint_only=False,
        **spec.params,
    )


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep only scalar numeric metric values (Experiments params are flat/scalar)."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, Real) and not isinstance(value, bool)
    }


def run_sweep(  # noqa: PLR0913 - explicit injectable seams keep the driver testable
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    sweep: SweepConfig,
    *,
    train_uri: str,
    val_uri: str | None = None,
    evaluate_fn: Callable[[str], dict[str, Any]],
    launch_fn: Callable[[Any, RunSpec, str, str | None], Any] | None = None,
    wait_fn: Callable[[Any, str], Any] = wait_for_tuning_job,
    find_fn: Callable[..., Any | None] = find_tuning_job_by_display_name,
    experiment: str | None = None,
) -> list[RunResult]:
    """Run every grid point of ``sweep``, reusing jobs and logging to Experiments.

    For each :class:`RunSpec`: reuse a job matching its display name if one
    exists (else launch via ``launch_fn``, defaulting to an SFT launch), wait for
    completion, resolve the tuned endpoint, and score it with ``evaluate_fn``.
    When ``experiment`` is set, each run's params + numeric metrics are logged to
    Vertex AI Experiments (the caller must have called
    :func:`geap_tuning.experiments.init_experiment` first). Returns one
    :class:`RunResult` per spec, ready for :func:`aggregate_results` /
    :func:`select_best_run`.
    """
    launch = launch_fn or _default_launch
    results: list[RunResult] = []
    for spec in build_run_specs(sweep):
        existing = find_fn(client, spec.display_name)
        reused = existing is not None
        job = existing if reused else launch(client, spec, train_uri, val_uri)
        job = wait_fn(client, job.name)
        endpoint = tuned_endpoint(job)
        metrics = evaluate_fn(endpoint)
        if experiment is not None:
            params = {"base_model": spec.base_model, **spec.params}
            with track_run(spec.display_name, params=params):
                log_summary_metrics(_numeric_metrics(metrics))
        results.append(
            RunResult(
                spec=spec,
                job_name=job.name,
                endpoint=endpoint,
                metrics=metrics,
                reused=reused,
            )
        )
    return results


def collect_checkpoint_curve(
    job: Any,  # noqa: ANN401 - SDK job type is dynamic
    evaluate_fn: Callable[[str], dict[str, Any]],
    *,
    metric: str = "accuracy",
) -> list[tuple[int, float]]:
    """Return ``(epoch, metric)`` points by evaluating each exported checkpoint.

    Requires the job to have run with ``export_last_checkpoint_only=False`` (see
    :func:`geap_tuning.jobs.list_checkpoints`). Points are sorted by epoch so the
    result plots left-to-right as a learning curve.
    """
    points: list[tuple[int, float]] = []
    for checkpoint in list_checkpoints(job):
        endpoint = checkpoint_endpoint(job, checkpoint.checkpoint_id)
        points.append((checkpoint.epoch, evaluate_fn(endpoint)[metric]))
    return sorted(points)
