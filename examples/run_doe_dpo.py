"""DOE demo: run a declarative **DPO** hyperparameter sweep and compare the runs.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units in
``geap_tuning.doe`` / ``geap_tuning.viz``). Run it with a real ``.env`` and
``gcloud auth`` in place:

    uv run python examples/run_doe_dpo.py           # run the sweep, print a table
    uv run --group viz python examples/run_doe_dpo.py --plot   # + save a bar PNG

The ``--plot`` flag needs the optional viz group (``uv sync --group viz``); the
default path has no extra dependency.

This crosses a small **DPO** grid (beta x epochs, 4 runs) on the support-reply
preference dataset with :func:`geap_tuning.doe.run_sweep` — the *same* driver as
the SFT sweep, selected by ``method="DPO"`` (dispatches ``launch_preference_job``).
Each grid point becomes a job with a deterministic display name, so re-running
**reuses** finished jobs instead of paying to re-tune. Every run is scored offline
on the held-out test split with a base-model autorater (tuned reply vs. the
dispreferred reference; DPO's headline metric is ``win_rate``, **not** accuracy)
and logged as one Vertex AI Experiments run, then aggregated into a cross-run
table (and optional chart). Tuning stays regional (``cfg.location``); Experiments
is regional too — keep them aligned.
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
from geap_tuning.preference.data import (
    SUPPORT_REPLIES,
    build_preference_dataset,
    build_preference_records,
    split_dataset,
)
from geap_tuning.preference.evaluate import run_preference_eval

EXPERIMENT_NAME = "geap-doe-dpo"
DATA_DIR = Path("datasets/preference_support_style")
GCS_PREFIX = "doe_dpo"
BASE_MODEL = "gemini-2.5-flash"  # flash-lite may not support preference tuning
JUDGE_MODEL = "gemini-2.5-flash"
PLOT_PATH = Path("doe_dpo_metrics.png")

_JUDGE_PROMPT = (
    "You are judging two customer-support replies to the same message. Pick the "
    "reply that is warmer, more concise, and more helpful (acknowledges the issue "
    "and offers a next step). Answer with only the single letter 'A' or 'B'.\n\n"
    "Customer message: {user}\n\nReply A: {a}\n\nReply B: {b}\n\nBetter reply:"
)

SWEEP = SweepConfig(
    name="dpo",
    method="DPO",
    base_model=BASE_MODEL,
    grid={"beta": [0.05, 0.1], "epochs": [1, 2]},  # 4 runs
)


def main() -> None:
    """Run the DPO DOE sweep against live GEAP and print a cross-run comparison."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only
    plot = "--plot" in sys.argv
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")
    print(f"Sweep '{SWEEP.name}' (DPO): grid={SWEEP.grid} → {2 * 2} runs")

    # 1. Build and stage the preference dataset (train/val staged; test held out).
    paths = build_preference_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")
    _, _, test_triples = split_dataset(SUPPORT_REPLIES)
    test_records = build_preference_records(test_triples)

    # 2. Point Experiments at the tuning region (init before run_sweep logs to it).
    init_experiment(EXPERIMENT_NAME, project=cfg.project, location=cfg.location)

    # 3. Offline scorer: tuned-vs-dispreferred autorater win rate on the test split.
    def judge_fn(user_text: str, cand_a: str, cand_b: str) -> str:
        verdict = generate(
            client, JUDGE_MODEL, _JUDGE_PROMPT.format(user=user_text, a=cand_a, b=cand_b)
        )
        return verdict[:1].upper()

    def evaluate_fn(endpoint: str) -> dict[str, object]:
        return run_preference_eval(
            test_records,
            generate_fn=lambda user_text, e=endpoint: generate(client, e, user_text),
            judge_fn=judge_fn,
        )

    # 4. Run every grid point (reuse-or-launch), scoring + logging each run.
    results = run_sweep(
        client,
        SWEEP,
        train_uri=train_uri,
        val_uri=val_uri,
        evaluate_fn=evaluate_fn,
        experiment=EXPERIMENT_NAME,
        labels=cfg.labels,
    )
    for result in results:
        tag = "reused" if result.reused else "launched"
        print(f"  {result.spec.name}: win_rate={result.metrics['win_rate']:.3f} ({tag})")

    # 5. Aggregate into comparable rows; report the winner (headline = win_rate).
    metric = HEADLINE_METRIC["DPO"]
    rows = aggregate_results(results, metrics=METRICS_BY_METHOD["DPO"])
    best = select_best_run({r.spec.name: r.metrics for r in results}, metric=metric)
    print("\nCross-run comparison:")
    for row in rows:
        print(f"  {row}")
    print(f"\nBest run ({metric}): {best}")

    # 6. Optional chart (needs the viz group).
    if plot:
        from geap_tuning.viz import plot_metric_bars  # noqa: PLC0415 - opt-in dep

        fig = plot_metric_bars(rows, metric=metric)
        fig.savefig(PLOT_PATH, bbox_inches="tight")
        print(f"Saved chart to {PLOT_PATH}")

    # 7. The same runs, read back from Experiments.
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio → Experiments.")


if __name__ == "__main__":
    main()
