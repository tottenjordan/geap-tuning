"""DOE demo: run a declarative **RLFT** hyperparameter sweep and compare the runs.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units in
``geap_tuning.doe`` / ``geap_tuning.viz``). Run it with a real ``.env`` and
``gcloud auth`` in place:

    uv run python examples/run_doe_rlft.py           # run the sweep, print a table
    uv run --group viz python examples/run_doe_rlft.py --plot   # + save a bar PNG

The ``--plot`` flag needs the optional viz group (``uv sync --group viz``); the
default path has no extra dependency.

This crosses a small **RLFT** grid (epochs x samples_per_prompt, 4 runs) on the
verifiable-math dataset with :func:`geap_tuning.doe.run_sweep` — the *same* driver
as the SFT/DPO sweeps, selected by ``method="RLFT"`` (dispatches
``launch_rlft_job``). The reward is a **declarative string-match** scorer carried
in ``sweep.fixed`` (a non-scalar object kept out of the run slug, the aggregate
rows, and the Experiments params by ``doe._scalar_params``); it is preflighted
once via ``validate_reward``. Each grid point becomes a job with a deterministic
display name, so re-running **reuses** finished jobs instead of paying to re-tune.
Every run is scored offline on the held-out test split (reward > 0 ⇒ correct;
headline metric ``accuracy``) and logged as one Vertex AI Experiments run.

NOTE: the RLFT docs call out ``gemini-3.5-flash`` as the supported base model —
verify availability in your region before running live. The client stays
**regional** (``cfg.location``): the ``global`` endpoint that serves Gemini 3.x
*inference* does not support tuning, so tuning + ``validate_reward`` must run in a
region.
"""

from __future__ import annotations

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
from geap_tuning.experiments import experiment_dataframe, init_experiment
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.rlft.data import (
    MATH_PROBLEMS,
    build_rlft_dataset,
    build_rlft_records,
    split_dataset,
)
from geap_tuning.rlft.evaluate import run_rlft_eval
from geap_tuning.rlft.tune import build_string_match_reward_config, validate_reward_config

EXPERIMENT_NAME = "geap-doe-rlft"
DATA_DIR = Path("datasets/rlft_math")
GCS_PREFIX = "doe_rlft"
BASE_MODEL = "gemini-3.5-flash"  # verify region availability before a live run
PLOT_PATH = Path("doe_rlft_metrics.png")

SWEEP = SweepConfig(
    name="rlft",
    method="RLFT",
    base_model=BASE_MODEL,
    grid={"epochs": [2, 4], "samples_per_prompt": [4, 8]},  # 4 runs
    fixed={"reward_config": build_string_match_reward_config()},
)


def main() -> None:
    """Run the RLFT DOE sweep against live GEAP and print a cross-run comparison."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; global excludes tuning
    plot = "--plot" in sys.argv
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")
    print(f"Sweep '{SWEEP.name}' (RLFT): grid={SWEEP.grid} → {2 * 2} runs")

    # 1. Build and stage the RLFT dataset (train/val staged; test held out).
    paths = build_rlft_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")
    train_problems, _, test_problems = split_dataset(MATH_PROBLEMS)
    train_records = build_rlft_records(train_problems)
    test_records = build_rlft_records(test_problems)

    # 2. Preflight the string-match reward on one record before spending money.
    preflight = validate_reward_config(
        client,
        project=cfg.project,
        location=cfg.location,
        sample_answer="Answer: 4",
        example_record=train_records[0],
        reward_config=build_string_match_reward_config(),
    )
    print(f"Reward preflight: {preflight}")

    # 3. Point Experiments at the tuning region (init before run_sweep logs to it).
    init_experiment(EXPERIMENT_NAME, project=cfg.project, location=cfg.location)

    # 4. Offline scorer: held-out answer accuracy (reward > 0 ⇒ correct).
    def evaluate_fn(endpoint: str) -> dict[str, object]:
        return run_rlft_eval(
            test_records,
            generate_fn=lambda user_text, e=endpoint: generate(client, e, user_text),
        )

    # 5. Run every grid point (reuse-or-launch), scoring + logging each run.
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
        print(f"  {result.spec.name}: accuracy={result.metrics['accuracy']:.3f} ({tag})")

    # 6. Aggregate into comparable rows; report the winner (headline = accuracy).
    metric = HEADLINE_METRIC["RLFT"]
    rows = aggregate_results(results, metrics=METRICS_BY_METHOD["RLFT"])
    best = select_best_run({r.spec.name: r.metrics for r in results}, metric=metric)
    print("\nCross-run comparison:")
    for row in rows:
        print(f"  {row}")
    print(f"\nBest run ({metric}): {best}")

    # 7. Optional chart (needs the viz group).
    if plot:
        from geap_tuning.viz import plot_metric_bars  # noqa: PLC0415 - opt-in dep

        fig = plot_metric_bars(rows, metric=metric)
        fig.savefig(PLOT_PATH, bbox_inches="tight")
        print(f"Saved chart to {PLOT_PATH}")

    # 8. The same runs, read back from Experiments.
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio → Experiments.")


if __name__ == "__main__":
    main()
