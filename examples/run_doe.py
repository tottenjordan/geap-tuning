"""DOE demo: run a declarative SFT hyperparameter sweep and compare the runs.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units in
``geap_tuning.doe`` / ``geap_tuning.viz``). Run it with a real ``.env`` and
``gcloud auth`` in place:

    uv run python examples/run_doe.py           # run the sweep, print a comparison table
    uv run --group viz python examples/run_doe.py --plot   # + save a grouped-bar PNG

The ``--plot`` flag needs the optional viz group (``uv sync --group viz``); the
default path has no extra dependency.

This crosses a small **SFT** grid (epochs x adapter_size, 4 runs) on the text
support-intent dataset with :func:`geap_tuning.doe.run_sweep`. Each grid point
becomes a job with a deterministic display name, so re-running **reuses** finished
jobs instead of paying to re-tune. Every run is scored offline on the held-out
test split and logged as one Vertex AI Experiments run, then aggregated into a
cross-run table (and optional chart). Tuning stays regional (``cfg.location``);
Experiments is regional too — keep them aligned.
"""

from __future__ import annotations

import sys
from pathlib import Path

from geap_tuning.config import genai_client, load_config
from geap_tuning.doe import (
    SweepConfig,
    aggregate_results,
    run_sweep,
    select_best_run,
)
from geap_tuning.experiments import experiment_dataframe, init_experiment
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.sft.data import SUPPORT_TICKETS, build_records, build_sft_dataset, split_dataset
from geap_tuning.sft.evaluate import run_eval

EXPERIMENT_NAME = "geap-doe-sft"
DATA_DIR = Path("datasets/sft_support_intent")
GCS_PREFIX = "doe_sft"
BASE_MODEL = "gemini-2.5-flash-lite"
PLOT_PATH = Path("doe_metrics.png")

SWEEP = SweepConfig(
    name="cheap",
    base_model=BASE_MODEL,
    grid={"epochs": [1, 2], "adapter_size": [4, 8]},  # 4 runs
)


def main() -> None:
    """Run the DOE sweep against live GEAP and print a cross-run comparison."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only
    plot = "--plot" in sys.argv
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")
    print(f"Sweep '{SWEEP.name}': grid={SWEEP.grid} → {2 * 2} runs")

    # 1. Build and stage the SFT dataset (train/val staged; test held out locally).
    paths = build_sft_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")
    _, _, test_pairs = split_dataset(SUPPORT_TICKETS)
    test_records = build_records(test_pairs)

    # 2. Point Experiments at the tuning region (init before run_sweep logs to it).
    init_experiment(EXPERIMENT_NAME, project=cfg.project, location=cfg.location)

    # 3. Offline scorer: score a tuned endpoint on the held-out test split.
    def evaluate_fn(endpoint: str) -> dict[str, object]:
        return run_eval(
            test_records,
            predict_fn=lambda user_text, e=endpoint: generate(client, e, user_text),
        )

    # 4. Run every grid point (reuse-or-launch), scoring + logging each run.
    results = run_sweep(
        client,
        SWEEP,
        train_uri=train_uri,
        val_uri=val_uri,
        evaluate_fn=evaluate_fn,
        experiment=EXPERIMENT_NAME,
    )
    for result in results:
        tag = "reused" if result.reused else "launched"
        print(f"  {result.spec.name}: acc={result.metrics['accuracy']:.3f} ({tag})")

    # 5. Aggregate into comparable rows; report the winner.
    rows = aggregate_results(results)
    best = select_best_run({r.spec.name: r.metrics for r in results})
    print("\nCross-run comparison:")
    for row in rows:
        print(f"  {row}")
    print(f"\nBest run (accuracy): {best}")

    # 6. Optional chart (needs the viz group).
    if plot:
        from geap_tuning.viz import plot_grouped_metric_bars  # noqa: PLC0415 - opt-in dep

        fig = plot_grouped_metric_bars(rows)
        fig.savefig(PLOT_PATH, bbox_inches="tight")
        print(f"Saved chart to {PLOT_PATH}")

    # 7. The same runs, read back from Experiments.
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio → Experiments.")


if __name__ == "__main__":
    main()
