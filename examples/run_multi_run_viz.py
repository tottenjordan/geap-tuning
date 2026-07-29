"""Read-only multi-run visualization: chart runs already tracked in Experiments.

NO TUNING COST. This reads an existing Vertex AI Experiment (populated by an
earlier ``examples/run_doe.py`` or ``examples/run_experiment_tracking.py`` run),
so it launches no jobs and calls no endpoints — a safe teaching artifact. It
still needs live GCP read access (``.env`` + ``gcloud auth``) and the optional
viz group:

    uv run --group viz python examples/run_multi_run_viz.py
    uv run --group viz python examples/run_multi_run_viz.py --experiment my-exp --out charts

Flow: point the SDK at the region, pull the experiment's cross-run dataframe,
normalize its ``metric.``/``param.`` columns into plot rows, and save a
grouped-bar and a single-metric chart. Experiments is regional — the experiment
must live in ``cfg.location``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from geap_tuning.config import load_config
from geap_tuning.experiments import experiment_dataframe, init_experiment
from geap_tuning.viz import (
    dataframe_to_rows,
    normalize_experiment_rows,
    plot_grouped_metric_bars,
    plot_metric_bars,
)

DEFAULT_EXPERIMENT = "geap-doe-sft"


def _arg(flag: str, default: str) -> str:
    """Return the value following ``flag`` in argv, or ``default``."""
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def main() -> None:
    """Render charts from an already-tracked experiment (no tuning)."""
    cfg = load_config()
    experiment = _arg("--experiment", DEFAULT_EXPERIMENT)
    out_dir = Path(_arg("--out", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Point the SDK at the region so the experiment resolves (no jobs launched).
    init_experiment(experiment, project=cfg.project, location=cfg.location)
    frame = experiment_dataframe(experiment)
    rows = normalize_experiment_rows(dataframe_to_rows(frame))
    print(f"Experiment '{experiment}': {len(rows)} runs")
    for row in rows:
        print(f"  {row}")

    grouped = plot_grouped_metric_bars(rows)
    grouped_path = out_dir / "multi_run_metrics.png"
    grouped.savefig(grouped_path, bbox_inches="tight")

    accuracy = plot_metric_bars(rows, metric="accuracy")
    accuracy_path = out_dir / "multi_run_accuracy.png"
    accuracy.savefig(accuracy_path, bbox_inches="tight")

    print(f"Saved {grouped_path} and {accuracy_path}")


if __name__ == "__main__":
    main()
