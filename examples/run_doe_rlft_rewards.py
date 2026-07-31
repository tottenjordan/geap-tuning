"""DOE demo: sweep the RLFT **reward shape** and compare the trained models.

REQUIRES LIVE GCP AND INCURS TUNING COST (four RLFT jobs). This is an integration
entrypoint, not covered by the test suite (pytest only exercises the mocked units
in ``geap_tuning.doe`` / ``geap_tuning.viz``). Run it with a real ``.env`` and
``gcloud auth`` in place:

    uv run python examples/run_doe_rlft_rewards.py            # run + print a table
    uv run --group viz python examples/run_doe_rlft_rewards.py --plot   # + bar PNG

The ``--plot`` flag needs the optional viz group (``uv sync --group viz``); the
default path has no extra dependency.

The other DOE demos sweep tuning **hyperparameters** (``run_doe_rlft.py`` crosses
``epochs x samples_per_prompt``). This one sweeps the axis unique to RLFT — the
**reward function** — tuning ``gemini-3.5-flash`` on the same verifiable-math
dataset once per reward shape and scoring every result on the same held-out split
against an **untuned baseline**, so the before→after lift is explicit:

- ``string-match`` — declarative ``Answer:\\s*-?\\d+`` format reward (no sandbox)
- ``code-exec``    — ships ``geap_tuning.rlft.reward`` to the sandbox (correctness)
- ``autorater``    — an LLM judge grading explanation quality
- ``composite``    — code-exec (0.8) + autorater (0.2), a weighted blend

A reward config is a non-scalar object, so it cannot be a grid axis (grid values
feed the run slug and Experiments params, which are scalars). Following
``run_doe_rlft.py``, each reward rides in ``sweep.fixed`` — kept out of the slug,
rows, and Experiments params by ``doe._scalar_params``. So each shape is its own
single-run ``SweepConfig`` (empty grid → one run), all logged to one shared
Experiment; the driver combines them under labels it controls, because every
empty-grid ``RunSpec.name`` is ``"default"`` and would otherwise collide.

NOTE: the RLFT docs call out ``gemini-3.5-flash`` as the supported base model —
verify availability in your region before running live. The tuning client stays
**regional** (``cfg.location``): the ``global`` endpoint that serves Gemini 3.x
*inference* does not support tuning. The untuned baseline is scored through a
separate ``global``-routed inference client (``genai_client(cfg, base_model=...)``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from geap_tuning.config import genai_client, genai_client_for_endpoint, load_config
from geap_tuning.doe import SweepConfig, run_sweep
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
from geap_tuning.rlft.tune import (
    build_autorater_reward_config,
    build_composite_reward_config,
    build_reward_config,
    build_string_match_reward_config,
    validate_reward_config,
)

EXPERIMENT_NAME = "geap-doe-rlft-rewards"
DATA_DIR = Path("datasets/rlft_math")
GCS_PREFIX = "doe_rlft_rewards"
BASE_MODEL = "gemini-3.5-flash"  # verify region availability before a live run
PLOT_PATH = Path("docs/doe/rlft-reward-shapes/metrics.png")
METRIC = "accuracy"  # headline: reward > 0 ⇒ correct (marker-gated)
CONTENT_METRIC = "content_accuracy"  # marker-agnostic: right number anywhere in reply


def main() -> None:
    """Run the reward-shape DOE against live GEAP and print a cross-shape table."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; global excludes tuning
    plot = "--plot" in sys.argv
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build and stage the RLFT dataset (train/val staged; test held out).
    paths = build_rlft_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")
    train_problems, _, test_problems = split_dataset(MATH_PROBLEMS)
    train_records = build_rlft_records(train_problems)
    test_records = build_rlft_records(test_problems)

    # 2. Build one reward object per shape. The autorater judge needs a
    #    fully-qualified publisher path (bare names fail with an opaque error).
    autorater_model = (
        f"projects/{cfg.project}/locations/{cfg.location}/publishers/google/models/gemini-2.5-flash"
    )
    shapes: list[tuple[str, SweepConfig]] = [
        (
            "string-match",
            SweepConfig(
                name="rew-string-match",
                method="RLFT",
                base_model=BASE_MODEL,
                fixed={"reward_config": build_string_match_reward_config()},
            ),
        ),
        (
            "code-exec",
            SweepConfig(
                name="rew-code-exec",
                method="RLFT",
                base_model=BASE_MODEL,
                fixed={"reward_config": build_reward_config()},
            ),
        ),
        (
            "autorater",
            SweepConfig(
                name="rew-autorater",
                method="RLFT",
                base_model=BASE_MODEL,
                fixed={
                    "reward_config": build_autorater_reward_config(autorater_model=autorater_model)
                },
            ),
        ),
        (
            "composite",
            SweepConfig(
                name="rew-composite",
                method="RLFT",
                base_model=BASE_MODEL,
                fixed={
                    "composite_reward_config": build_composite_reward_config(
                        [
                            (build_reward_config(), 0.8),
                            (build_autorater_reward_config(autorater_model=autorater_model), 0.2),
                        ]
                    )
                },
            ),
        ),
    ]

    # 3. Preflight every reward on one record before spending money.
    for label, sweep in shapes:
        preflight = validate_reward_config(
            client,
            project=cfg.project,
            location=cfg.location,
            sample_answer="Answer: 4",
            example_record=train_records[0],
            reward_config=sweep.fixed.get("reward_config"),
            composite_reward_config=sweep.fixed.get("composite_reward_config"),
        )
        print(f"Preflight [{label}]: {preflight}")

    # 4. Point Experiments at the tuning region (init before run_sweep logs to it).
    init_experiment(EXPERIMENT_NAME, project=cfg.project, location=cfg.location)

    # 5. Offline scorer: held-out answer accuracy (reward > 0 ⇒ correct). Shared.
    #    A tuned Gemini 3.x model lands on a us/eu multi-region endpoint, so route
    #    inference to the endpoint's own location (a us-central1 client 404s it).
    def evaluate_fn(endpoint: str) -> dict[str, object]:
        eval_client = genai_client_for_endpoint(cfg, endpoint)
        return run_rlft_eval(
            test_records,
            generate_fn=lambda user_text, e=endpoint: generate(eval_client, e, user_text),
        )

    # 6. Run each reward shape as its own single-run sweep (reuse-or-launch).
    results = []
    for label, sweep in shapes:
        result = run_sweep(
            client,
            sweep,
            train_uri=train_uri,
            val_uri=val_uri,
            evaluate_fn=evaluate_fn,
            experiment=EXPERIMENT_NAME,
            labels=cfg.labels,
        )[0]
        results.append((label, result))
        tag = "reused" if result.reused else "launched"
        print(
            f"  {label}: {METRIC}={result.metrics[METRIC]:.3f} "
            f"{CONTENT_METRIC}={result.metrics[CONTENT_METRIC]:.3f} ({tag})"
        )

    # 7. Untuned baseline through a global-routed inference client (3.x inference).
    base_client = genai_client(cfg, base_model=BASE_MODEL)
    baseline = run_rlft_eval(
        test_records,
        generate_fn=lambda user_text: generate(base_client, BASE_MODEL, user_text),
    )
    print(
        f"  untuned baseline: {METRIC}={baseline[METRIC]:.3f} "
        f"{CONTENT_METRIC}={baseline[CONTENT_METRIC]:.3f}"
    )

    # 8. Combine into driver-labeled rows (bypassing aggregate_results, whose keys
    #    would all be "default"). Two metrics per run: reward-based accuracy (marker
    #    contract) and marker-agnostic content_accuracy (right number in prose) — a
    #    format-only reward can leave the first low while the second stays high.
    def _row(run: str, metrics: dict[str, object]) -> dict[str, object]:
        return {"run": run, METRIC: metrics[METRIC], CONTENT_METRIC: metrics[CONTENT_METRIC]}

    rows = [_row("untuned", baseline)]
    rows += [_row(label, result.metrics) for label, result in results]
    best_label, best_result = max(results, key=lambda lr: lr[1].metrics[METRIC])
    print("\nCross-reward comparison:")
    for row in rows:
        print(f"  {row}")
    print(f"\nBest reward shape ({METRIC}): {best_label} = {best_result.metrics[METRIC]:.3f}")

    # 9. Optional chart (needs the viz group): grouped bars for both metrics.
    if plot:
        from geap_tuning.viz import plot_grouped_metric_bars  # noqa: PLC0415 - opt-in dep

        fig = plot_grouped_metric_bars(rows, metrics=(METRIC, CONTENT_METRIC))
        fig.savefig(PLOT_PATH, bbox_inches="tight")
        print(f"Saved chart to {PLOT_PATH}")

    # 10. The tuned shapes, read back from Experiments (baseline is offline-only).
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio → Experiments.")


if __name__ == "__main__":
    main()
