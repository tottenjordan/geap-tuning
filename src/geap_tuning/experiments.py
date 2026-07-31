"""Vertex AI Experiments + Managed TensorBoard tracking (opt-in "Layer 2").

Thin wrapper over ``google.cloud.aiplatform`` for logging tuning-run params and
your own offline metrics, so many runs can be compared side by side. This is the
one sanctioned place to mix SDKs: Experiments only *orchestrates around* a job —
the tuning call itself stays on the ``google.genai`` path. See
``docs/notes/experiment-tracking.md``.

Two independent layers exist. **Layer 1** is the automatic per-job train/val
metrics that stream to the console Monitor tab with no code — nothing here. This
module is **Layer 2**: opt-in, cross-run comparison plus your own metrics.

- **Summary metrics** (:func:`log_summary_metrics`) — one value per key per run;
  logged to Vertex AI Experiments with **no** TensorBoard required.
- **Time-series metrics** (:func:`log_timeseries_metrics`) — per-step curves;
  these live in a **Managed TensorBoard** instance and therefore require one to
  be attached to the experiment (:func:`get_or_create_tensorboard` +
  ``tensorboard=`` on :func:`init_experiment`). TensorBoard carries cost and a
  provisioning wait, so it is opt-in.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from google.cloud import aiplatform

if TYPE_CHECKING:
    from collections.abc import Iterator


def init_experiment(
    experiment: str,
    *,
    project: str,
    location: str,
    tensorboard: str | None = None,
) -> None:
    """Create/select the experiment context so later ``track_run`` calls attach to it.

    Preferred over ``aiplatform.Experiment.create`` because it also sets the
    global context. Pass ``tensorboard`` (a TensorBoard resource name from
    :func:`get_or_create_tensorboard`) to attach Managed TensorBoard and enable
    time-series metrics; leave it ``None`` for a TensorBoard-less experiment that
    still records params and summary metrics. Keep ``location`` aligned with the
    tuning region — Experiments is a regional resource.
    """
    aiplatform.init(
        project=project,
        location=location,
        experiment=experiment,
        experiment_tensorboard=tensorboard,
    )


def get_or_create_tensorboard(
    display_name: str,
    *,
    project: str,
    location: str,
    labels: dict[str, str] | None = None,
) -> str:
    """Return the resource name of a Managed TensorBoard, reusing one by display name.

    Mirrors ``find_tuning_job_by_display_name`` for cost control: reuse an
    existing instance instead of provisioning (and paying for) a duplicate.
    Only needed on the opt-in time-series path. ``labels`` are resource labels
    applied only when a new instance is created (a reused one keeps its existing
    labels); ``None`` omits the field. Unlike tuning jobs, ``aiplatform.init`` /
    ``start_run`` accept no label param, so TensorBoard is the only Experiments
    resource that can be labeled here. See ``docs/notes/resource-labels.md``.
    """
    for tb in aiplatform.Tensorboard.list(project=project, location=location):
        if tb.display_name == display_name:
            return tb.resource_name
    created = aiplatform.Tensorboard.create(
        display_name=display_name,
        project=project,
        location=location,
        labels=labels,
    )
    return created.resource_name


@contextlib.contextmanager
def track_run(
    run_name: str,
    *,
    params: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Open one trackable run, logging ``params`` on entry.

    ``params`` values must be scalar (str/int/float/bool) — Experiments params
    are a flat input snapshot (base_model, adapter_size, epochs, …).
    """
    with aiplatform.start_run(run_name) as run:
        if params:
            aiplatform.log_params(params)
        yield run


def log_summary_metrics(metrics: dict[str, float]) -> None:
    """Log summary metrics: one value per key per run (e.g. held-out accuracy).

    No TensorBoard required. Use for offline eval outputs.
    """
    aiplatform.log_metrics(metrics)


def log_timeseries_metrics(metrics: dict[str, float], *, step: int) -> None:
    """Log per-step time-series metrics at ``step``.

    Stored in Vertex AI / Managed TensorBoard, so the experiment must have a
    TensorBoard attached (see :func:`init_experiment` ``tensorboard=``).
    """
    aiplatform.log_time_series_metrics(metrics, step=step)


def experiment_dataframe(experiment: str) -> Any:  # noqa: ANN401 - pandas is a transitive dep
    """Return a cross-run comparison table (params + summary metrics) for ``experiment``."""
    return aiplatform.Experiment(experiment).get_data_frame()
