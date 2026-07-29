"""Multi-run visualization helpers for tuning sweeps.

Turns already-shaped rows (the output of :func:`geap_tuning.doe.aggregate_results`
or :func:`dataframe_to_rows` over an Experiments dataframe) into matplotlib
figures for comparing several tuning runs side by side. matplotlib and pandas
live in the optional ``viz`` dependency group and are **lazy-imported** behind
:func:`_import_matplotlib` / :func:`_import_pandas` (modeled on
``sft_vision.data._import_kagglehub``), so importing this module — and the whole
package — never requires the group. Install it with ``uv sync --group viz``.

Every plot function takes rows/series and returns a :class:`matplotlib.figure.Figure`
the caller can ``savefig`` or display inline in a notebook; none of them touch GCP.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_DEFAULT_METRICS = ("accuracy", "macro_f1")
_VIZ_HINT = "run: uv sync --group viz"


def _import_matplotlib() -> Any:  # noqa: ANN401 - returns the pyplot module
    """Return ``matplotlib.pyplot``, or raise an actionable error if uninstalled."""
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        msg = f"matplotlib is required for plotting; {_VIZ_HINT}"
        raise RuntimeError(msg) from exc


def _import_pandas() -> Any:  # noqa: ANN401 - returns the pandas module
    """Return ``pandas``, or raise an actionable error if uninstalled."""
    try:
        return importlib.import_module("pandas")
    except ImportError as exc:
        msg = f"pandas is required for tabular viz; {_VIZ_HINT}"
        raise RuntimeError(msg) from exc


def rows_to_dataframe(rows: Sequence[Mapping[str, Any]]) -> Any:  # noqa: ANN401 - pandas.DataFrame
    """Build a ``pandas.DataFrame`` from aggregate rows (one row per run)."""
    pd = _import_pandas()
    return pd.DataFrame(list(rows))


def dataframe_to_rows(frame: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - pandas.DataFrame
    """Return a DataFrame as a list of row dicts.

    Bridges :func:`geap_tuning.experiments.experiment_dataframe` output into the
    same row shape the plot functions consume, so the read-only viz path can
    reuse them without a live sweep.
    """
    return frame.to_dict(orient="records")


def normalize_experiment_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rename Vertex Experiments columns into the shape the plot functions expect.

    :func:`geap_tuning.experiments.experiment_dataframe` yields columns prefixed
    ``metric.`` / ``param.`` plus a ``run_name`` column. This strips those
    prefixes and renames ``run_name`` → ``run`` so a read-back dataframe plots
    the same way as :func:`geap_tuning.doe.aggregate_results` rows. Unprefixed
    keys pass through unchanged; on a name clash the later key wins.
    """
    normalized: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key == "run_name":
                out["run"] = value
            elif key.startswith("metric."):
                out[key[len("metric.") :]] = value
            elif key.startswith("param."):
                out[key[len("param.") :]] = value
            else:
                out[key] = value
        normalized.append(out)
    return normalized


def _is_missing(value: Any) -> bool:  # noqa: ANN401 - accepts any cell value
    """True if ``value`` is absent-like: ``None`` or a NaN float (``v != v``)."""
    return value is None or value != value  # noqa: PLR0124 - NaN self-inequality check


def drop_rows_missing_metrics(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str] = _DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    """Drop rows that have **no** usable value for any of ``metrics``.

    A read-back experiment can mix run kinds — e.g. a time-series-only run stores
    its values under ``time_series_metric.*`` and leaves the summary ``accuracy`` /
    ``macro_f1`` as ``NaN``. Such a row would plot as an empty bar cluster, so we
    keep a row only when at least one requested metric is present and non-NaN.
    """
    return [
        dict(row)
        for row in rows
        if any(metric in row and not _is_missing(row[metric]) for metric in metrics)
    ]


def plot_metric_bars(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    label_key: str = "run",
) -> Any:  # noqa: ANN401 - matplotlib Figure
    """Return a bar chart of ``metric`` across runs, one bar per row."""
    plt = _import_matplotlib()
    labels = [str(row[label_key]) for row in rows]
    values = [row[metric] for row in rows]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_xlabel(label_key)
    ax.set_title(f"{metric} by {label_key}")
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_grouped_metric_bars(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = _DEFAULT_METRICS,
    label_key: str = "run",
) -> Any:  # noqa: ANN401 - matplotlib Figure
    """Return a grouped bar chart comparing several ``metrics`` across runs.

    Each run is a cluster on the x-axis with one bar per metric. Metrics absent
    from a row contribute ``0`` for that run so the clusters stay aligned.
    """
    plt = _import_matplotlib()
    labels = [str(row[label_key]) for row in rows]
    positions = list(range(len(labels)))
    n_metrics = len(metrics) or 1
    width = 0.8 / n_metrics
    fig, ax = plt.subplots()
    for index, metric in enumerate(metrics):
        offset = (index - (n_metrics - 1) / 2) * width
        shifted = [pos + offset for pos in positions]
        values = [row.get(metric, 0) for row in rows]
        ax.bar(shifted, values, width=width, label=metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("metric value")
    ax.set_xlabel(label_key)
    ax.set_title("Metrics by " + label_key)
    ax.legend()
    return fig


def plot_curves(
    series: Mapping[str, Sequence[tuple[float, float]]],
    *,
    metric: str,
    xlabel: str = "epoch",
) -> Any:  # noqa: ANN401 - matplotlib Figure
    """Return a line chart overlaying one curve per run.

    ``series`` maps a run label to its ``(x, value)`` points (e.g. the output of
    :func:`geap_tuning.doe.collect_checkpoint_curve`).
    """
    plt = _import_matplotlib()
    fig, ax = plt.subplots()
    for label, points in series.items():
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {xlabel}")
    ax.legend()
    return fig
