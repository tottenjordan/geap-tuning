"""DOE demo that shows a real metric improvement: banking77 SFT sweep + baseline.

REQUIRES LIVE GCP AND INCURS TUNING COST (4 SFT jobs + baseline/eval inference).
This is an integration entrypoint, not covered by the test suite (pytest exercises the
mocked units in ``geap_tuning.doe`` / ``geap_tuning.sft.banking``). Run it with a real
``.env`` and ``gcloud auth`` in place:

    uv run --group viz python examples/run_doe_banking77.py
    uv run --group viz python examples/run_doe_banking77.py --csv-dir path/to/csvs

Unlike the demo ``run_doe.py`` sweep — which saturates at accuracy = 1.0 on the tiny
5-intent support-ticket set, so no grid cell can be told apart — this crosses the same
small **SFT** grid (epochs x adapter_size, 4 runs) on **banking77** (77 fine-grained
banking intents; PolyAI/banking77, CC-BY-4.0). The task has real headroom, so the runs
separate and every tuned run is compared against an **untuned baseline** for a
before -> after story.

Metric values are our own offline :func:`geap_tuning.sft.evaluate.run_eval` on the
held-out test split (Layer-1 Monitor loss curves are console-only, not SDK-fetchable).
Each grid point becomes a job with a deterministic display name, so re-running **reuses**
finished jobs instead of paying to re-tune (change ``SWEEP.name`` or the grid to force a
re-tune). Tuning is regional-only (``cfg.location``); Experiments is regional too — keep
them aligned. ``--csv-dir`` points at a pre-downloaded ``train.csv``/``test.csv`` for
offline runs; otherwise the CSVs are fetched and cached under ``DATA_DIR/raw``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from geap_tuning.config import genai_client, load_config
from geap_tuning.doe import (
    HEADLINE_METRIC,
    METRICS_BY_METHOD,
    SweepConfig,
    aggregate_results,
    run_sweep,
    select_best_run,
)
from geap_tuning.experiments import (
    experiment_dataframe,
    init_experiment,
    log_summary_metrics,
    track_run,
)
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.sft.banking import (
    banking_labels,
    build_banking_dataset,
    build_system_instruction,
    load_pairs_from_csv,
    parse_banking_prediction,
)
from geap_tuning.sft.evaluate import run_eval

EXPERIMENT_NAME = "geap-doe-banking77"
DATA_DIR = Path("datasets/banking77_sft")
GCS_PREFIX = "doe_banking77"
BASE_MODEL = "gemini-2.5-flash-lite"
PLOT_PATH = Path("doe_banking77_metrics.png")
PER_CLASS = {"train": 10, "val": 2, "test": 5}  # 770 / 154 / 385 examples

SWEEP = SweepConfig(
    name="banking77",
    base_model=BASE_MODEL,
    grid={"epochs": [2, 8], "adapter_size": [4, 16]},  # 4 runs
)


def _csv_dir_arg() -> str | None:
    """Return the value after ``--csv-dir`` on the command line, if present."""
    if "--csv-dir" in sys.argv:
        return sys.argv[sys.argv.index("--csv-dir") + 1]
    return None


def main() -> None:
    """Run the banking77 DOE against live GEAP and print a before/after comparison."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only
    csv_dir = _csv_dir_arg()
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")
    print(f"Sweep '{SWEEP.name}': grid={SWEEP.grid} -> {2 * 2} runs + 1 baseline")

    # 1. Build and stage the SFT dataset (train/val staged; test held out locally).
    paths = build_banking_dataset(DATA_DIR, csv_dir=csv_dir, per_class=PER_CLASS)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 2. Held-out test records + the shared label space / system instruction.
    test_records = [json.loads(line) for line in Path(paths["test"]).read_text().splitlines()]
    raw_train = Path(csv_dir) / "train.csv" if csv_dir else DATA_DIR / "raw" / "train.csv"
    labels = banking_labels(load_pairs_from_csv(raw_train))
    system_instruction = build_system_instruction(labels)

    # 3. Point Experiments at the tuning region (init before any run logs to it).
    init_experiment(EXPERIMENT_NAME, project=cfg.project, location=cfg.location)

    # 4. Offline scorer: constrain the model with the label list, canonicalize its reply.
    def make_predict(endpoint: str):  # noqa: ANN202 - returns a local closure
        def predict(user_text: str) -> str:
            reply = generate(client, endpoint, user_text, system_instruction=system_instruction)
            return parse_banking_prediction(reply, labels)

        return predict

    def evaluate_fn(endpoint: str) -> dict[str, object]:
        return run_eval(test_records, predict_fn=make_predict(endpoint))

    # 5. Baseline (before): score the untuned base model and log it as its own run.
    base_metrics = evaluate_fn(BASE_MODEL)
    baseline_run = f"geap-doe-{SWEEP.name}-baseline"
    with track_run(baseline_run, params={"base_model": BASE_MODEL, "epochs": 0, "adapter_size": 0}):
        log_summary_metrics(
            {"accuracy": base_metrics["accuracy"], "macro_f1": base_metrics["macro_f1"]}
        )
    print(f"\nBaseline (untuned {BASE_MODEL}): acc={base_metrics['accuracy']:.3f}")

    # 6. Sweep (after): run every grid point (reuse-or-launch), scoring + logging each.
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

    # 7. Aggregate baseline + tuned runs into comparable rows; report the improvement.
    metrics = METRICS_BY_METHOD["SFT"]
    baseline_row = {
        "run": "baseline",
        "base_model": BASE_MODEL,
        "epochs": 0,
        "adapter_size": 0,
        "accuracy": base_metrics["accuracy"],
        "macro_f1": base_metrics["macro_f1"],
    }
    rows = [baseline_row, *aggregate_results(results, metrics=metrics)]
    by_name = {r.spec.name: r.metrics for r in results}
    best_name = select_best_run(by_name, metric=HEADLINE_METRIC["SFT"])
    improvement = by_name[best_name]["accuracy"] - base_metrics["accuracy"]
    print("\nCross-run comparison (baseline first):")
    for row in rows:
        print(f"  {row}")
    print(f"\nBest run: {best_name} (accuracy={by_name[best_name]['accuracy']:.3f})")
    print(f"Improvement over baseline: {improvement:+.3f}")

    # 8. Chart baseline + runs (needs the viz group).
    try:
        from geap_tuning.viz import plot_grouped_metric_bars  # noqa: PLC0415 - opt-in dep

        fig = plot_grouped_metric_bars(rows, metrics=metrics)
        fig.savefig(PLOT_PATH, bbox_inches="tight")
        print(f"\nSaved chart to {PLOT_PATH}")
    except RuntimeError as exc:  # viz group not installed
        print(f"\nSkipped chart ({exc}). Install with: uv sync --group viz")

    # 9. The same runs, read back from Experiments.
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio -> Experiments.")


if __name__ == "__main__":
    main()
