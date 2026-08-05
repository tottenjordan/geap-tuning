"""DOE demo: a reward-shape sweep engineered to actually **rank** the shapes.

REQUIRES LIVE GCP AND INCURS TUNING COST (four RLFT jobs). This is an integration
entrypoint, not covered by the test suite (pytest exercises the mocked units in
``geap_tuning.doe`` / ``geap_tuning.viz`` and the pure helpers in
``geap_tuning.rlft.bench`` / ``geap_tuning.rlft.evaluate``). Run it with a real
``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_doe_rlft_reward_ranking.py --pilot-only  # gate only, no spend
    uv run python examples/run_doe_rlft_reward_ranking.py               # run + tables
    uv run --group viz python examples/run_doe_rlft_reward_ranking.py --plot   # + PNGs

The sibling demo ``run_doe_rlft_rewards.py`` swept the same four reward shapes and
returned a **flat null result** — every shape and the untuned baseline scored
1.000 — because the base model saturated the task and the ``Answer: <n>`` marker
was handed to it for free by the system instruction. See
``docs/doe/rlft-reward-shapes/README.md``. This demo redesigns the *experiment* so
the shapes measurably diverge, **reusing the orchestration unchanged**
(``doe.run_sweep`` + one single-run ``SweepConfig`` per shape):

1. **Neutral system instruction** — ``bench.NEUTRAL_SYSTEM_INSTRUCTION`` drops the
   ``Answer: <n>`` contract, so the format-only ``string-match`` reward has
   something real to teach. This is the load-bearing change: ``format_rate`` is the
   axis this sweep can actually rank on.
2. **Base model** — ``gemini-3.5-flash``, the **only** base GEAP documents as
   RLFT-supported. The original design called for a *weaker* base to open
   correctness headroom, but that lever does not exist: RLFT tunes only
   ``gemini-3.5-flash``, which saturates any verifiable-math task (both pilots on a
   ``gemini-2.5-flash-lite`` base still scored correctness ≈ 0.95 — see
   ``docs/doe/rlft-reward-ranking/README.md``). So ``correctness`` is expected to
   stay near ceiling: a documented finding, not a rankable axis here.
3. **Harder, tiered, larger bank** — ``rlft.bench.HARD_MATH_PROBLEMS`` (150 multi-
   step problems across easy/medium/hard, stratified split → test n≈30). Kept for a
   larger, balanced test split, though it cannot create correctness headroom against
   this base.
4. **Multi-objective scoring** — ``run_rlft_multimetric_eval`` scores three
   independent axes (``correctness`` / ``format_rate`` / ``explanation_quality``)
   plus a per-difficulty breakdown, and ``bootstrap_ci`` reports each with a 95%
   CI so "best shape" is a ranked, significance-aware claim, not an arbitrary
   ``max``.

A **pilot gate** scores the untuned base first and refuses to launch the four jobs
unless the **format axis** has headroom (baseline ``format_rate`` below ceiling —
the marker is no longer free). ``correctness`` is reported but *not* gated: against
``gemini-3.5-flash`` it is saturated by design. Pass ``--force`` to launch anyway.
The explanation-quality judge uses a **different** model (``gemini-2.5-pro``) than
the training autorater reward (``gemini-2.5-flash``) to avoid grading the model
with its own trainer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from geap_tuning.config import genai_client, genai_client_for_endpoint, load_config
from geap_tuning.doe import RunResult, SweepConfig, run_sweep
from geap_tuning.experiments import experiment_dataframe, init_experiment
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.rlft.bench import (
    HARD_MATH_PROBLEMS,
    NEUTRAL_SYSTEM_INSTRUCTION,
    build_bench_dataset,
    build_bench_records,
    split_stratified,
)
from geap_tuning.rlft.evaluate import bootstrap_ci, run_rlft_multimetric_eval
from geap_tuning.rlft.tune import (
    build_autorater_reward_config,
    build_composite_reward_config,
    build_reward_config,
    build_string_match_reward_config,
    validate_reward_config,
)

EXPERIMENT_NAME = "geap-doe-rlft-reward-ranking"
DATA_DIR = Path("datasets/rlft_bench")
GCS_PREFIX = "doe_rlft_reward_ranking"
BASE_MODEL = "gemini-3.5-flash"  # only RLFT-supported base; correctness saturates by design
JUDGE_MODEL = "gemini-2.5-pro"  # distinct from the training autorater (gemini-2.5-flash)
PLOT_PATH = Path("docs/doe/rlft-reward-ranking/metrics.png")
TIER_PLOT_PATH = Path("docs/doe/rlft-reward-ranking/metrics_by_tier.png")

# The three measured axes. correctness is reported but saturated on gemini-3.5-flash
# (the only RLFT base), so the pilot gate keys only on format_rate — the one axis the
# neutral instruction leaves open to rank.
AXES = ("correctness", "format_rate", "explanation_quality")
FORMAT_CEILING = 0.5  # baseline marker rate must be low (marker not free) to rank format

_SCORE_RE = re.compile(r"(?:0?\.\d+|[01](?:\.0+)?)")


def make_judge(client: object, model: str):  # noqa: ANN201 - returns a closure
    """Return a ``judge_fn(question, reply, truth) -> float`` scoring 0..1.

    Asks ``model`` to rate the explanation's clarity/correctness and parses the
    first number in ``[0, 1]`` from the reply; unparseable replies score 0.0.
    """

    def judge_fn(question: str, reply: str, truth: str) -> float:
        prompt = (
            "You are grading a math explanation. Rate how clear and correct the "
            "reasoning is on a scale from 0.0 to 1.0. Respond with ONLY the number.\n\n"
            f"Problem: {question}\nCorrect answer: {truth}\nStudent's answer: {reply}"
        )
        # gemini-2.5-pro cannot disable thinking (rejects thinking_budget=0, the
        # generate() default); -1 lets it think dynamically.
        raw = generate(client, model, prompt, thinking_budget=-1)
        match = _SCORE_RE.search(raw)
        if match is None:
            return 0.0
        return max(0.0, min(1.0, float(match.group())))

    return judge_fn


def _row(run: str, metrics: dict[str, object]) -> dict[str, object]:
    """Flatten a multimetric result into a viz/leaderboard row (one per run)."""
    row: dict[str, object] = {"run": run}
    for axis in AXES:
        row[axis] = metrics.get(axis, 0.0)
    return row


def _tier_row(run: str, metrics: dict[str, object]) -> dict[str, object]:
    """Row of per-difficulty correctness for the by-tier chart."""
    by_difficulty = metrics.get("by_difficulty", {})
    row: dict[str, object] = {"run": run}
    for tier in ("easy", "medium", "hard"):
        row[tier] = by_difficulty.get(tier, {}).get("correctness", 0.0)
    return row


def _print_leaderboard(rows: list[dict[str, object]], results_by_run: dict[str, dict]) -> None:
    """Print a per-axis leaderboard with a bootstrap 95% CI on correctness."""
    print("\nPer-axis leaderboard (rank by each axis, descending):")
    for axis in AXES:
        ranked = sorted(rows, key=lambda r: r[axis], reverse=True)
        winner = ranked[0]
        print(f"  {axis}: winner={winner['run']} ({winner[axis]:.3f})")
        for row in ranked:
            print(f"      {row['run']:>14}: {row[axis]:.3f}")

    print("\nCorrectness with bootstrap 95% CI:")
    for run, metrics in results_by_run.items():
        low, high = bootstrap_ci(int(metrics["content_hits"]), int(metrics["n"]))
        print(f"  {run:>14}: {metrics['correctness']:.3f}  CI[{low:.3f}, {high:.3f}]")


def _gate_ok(baseline: dict[str, object], *, force: bool) -> bool:
    """Print the pilot line and return whether the format axis clears the gate.

    correctness is saturated on ``gemini-3.5-flash`` by design, so the gate keys
    only on ``format_rate`` (the marker is no longer free under the neutral
    instruction). ``--force`` overrides a failing gate.
    """
    print(
        f"\nPilot gate — untuned {BASE_MODEL}: correctness={baseline['correctness']:.3f} "
        f"(saturated by design, not gated) format_rate={baseline['format_rate']:.3f} "
        f"explanation_quality={baseline.get('explanation_quality', 0.0):.3f}"
    )
    if baseline["format_rate"] < FORMAT_CEILING or force:
        print("Pilot gate PASSED — format headroom confirmed; launching the sweep.\n")
        return True
    print(
        f"\nPILOT GATE FAILED: baseline format_rate must be < {FORMAT_CEILING} for the "
        "sweep to measure format lift (the marker should not be emitted for free). "
        "Pass --force to launch anyway."
    )
    return False


def _build_shapes(autorater_model: str) -> list[tuple[str, SweepConfig]]:
    """One single-run ``SweepConfig`` per reward shape (reward object in ``fixed``)."""
    return [
        (
            "string-match",
            SweepConfig(
                name="rew-rank-string-match",
                method="RLFT",
                base_model=BASE_MODEL,
                fixed={"reward_config": build_string_match_reward_config()},
            ),
        ),
        (
            "code-exec",
            SweepConfig(
                name="rew-rank-code-exec",
                method="RLFT",
                base_model=BASE_MODEL,
                fixed={"reward_config": build_reward_config()},
            ),
        ),
        (
            "autorater",
            SweepConfig(
                name="rew-rank-autorater",
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
                name="rew-rank-composite",
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


def _run_shapes(
    shapes: list[tuple[str, SweepConfig]],
    baseline: dict[str, object],
    run_one: Callable[[SweepConfig], RunResult],
) -> dict[str, dict]:
    """Run each shape via ``run_one``, isolating failures.

    A single job's failure (e.g. a transient server-side INTERNAL error raised as
    a ``RuntimeError``) is reported and skipped so it can't discard the shapes that
    already succeeded.
    """
    results_by_run: dict[str, dict] = {"untuned": baseline}
    for label, sweep in shapes:
        try:
            result = run_one(sweep)
        except RuntimeError as exc:  # a failed/transient job must not abort the sweep
            print(f"  {label}: SKIPPED — {exc}")
            continue
        results_by_run[label] = result.metrics
        tag = "reused" if result.reused else "launched"
        print(
            f"  {label}: correctness={result.metrics['correctness']:.3f} "
            f"format_rate={result.metrics['format_rate']:.3f} "
            f"quality={result.metrics.get('explanation_quality', 0.0):.3f} ({tag})"
        )
    return results_by_run


def main() -> None:
    """Run the rank-capable reward-shape DOE against live GEAP and print tables."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; global excludes tuning
    plot = "--plot" in sys.argv
    force = "--force" in sys.argv
    pilot_only = "--pilot-only" in sys.argv  # score the gate, then stop (no tuning spend)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build and stage the harder tiered bench (neutral instruction — no marker).
    paths = build_bench_dataset(DATA_DIR, system_instruction=NEUTRAL_SYSTEM_INSTRUCTION)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")
    train_problems, _, test_problems = split_stratified(HARD_MATH_PROBLEMS)
    train_records = build_bench_records(
        train_problems, system_instruction=NEUTRAL_SYSTEM_INSTRUCTION
    )
    test_records = build_bench_records(test_problems, system_instruction=NEUTRAL_SYSTEM_INSTRUCTION)
    print(f"Bench: {len(train_records)} train, {len(test_records)} test problems")

    # 2. Pilot gate — score the UNTUNED base before spending on four jobs. Gemini
    #    2.x inference runs on the global endpoint, so route the baseline there.
    base_client = genai_client(cfg, base_model=BASE_MODEL)
    judge_client = genai_client(cfg, base_model=JUDGE_MODEL)
    judge_fn = make_judge(judge_client, JUDGE_MODEL)
    baseline = run_rlft_multimetric_eval(
        test_records,
        generate_fn=lambda user_text, system_instruction: generate(
            base_client, BASE_MODEL, user_text, system_instruction=system_instruction
        ),
        judge_fn=judge_fn,
    )
    if not _gate_ok(baseline, force=force):
        return
    if pilot_only:
        print("--pilot-only: stopping before any tuning spend.")
        return

    # 3. Build one reward object per shape (autorater needs a fully-qualified path).
    autorater_model = (
        f"projects/{cfg.project}/locations/{cfg.location}/publishers/google/models/gemini-2.5-flash"
    )
    shapes = _build_shapes(autorater_model)

    # 4. Preflight every reward on one record before spending money.
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

    # 5. Point Experiments at the tuning region (init before run_sweep logs to it).
    init_experiment(EXPERIMENT_NAME, project=cfg.project, location=cfg.location)

    # 6. Shared offline scorer: multi-axis, replaying each record's systemInstruction.
    #    A tuned model lands on the endpoint's own (multi-)region, so route there.
    def evaluate_fn(endpoint: str) -> dict[str, object]:
        eval_client = genai_client_for_endpoint(cfg, endpoint)
        return run_rlft_multimetric_eval(
            test_records,
            generate_fn=lambda user_text, system_instruction, e=endpoint: generate(
                eval_client, e, user_text, system_instruction=system_instruction
            ),
            judge_fn=judge_fn,
        )

    # 7. Run each reward shape as its own single-run sweep (reuse-or-launch),
    #    isolating each so one failed job can't discard the shapes that succeeded.
    def run_one(sweep: SweepConfig) -> RunResult:
        return run_sweep(
            client,
            sweep,
            train_uri=train_uri,
            val_uri=val_uri,
            evaluate_fn=evaluate_fn,
            experiment=EXPERIMENT_NAME,
            labels=cfg.labels,
        )[0]

    results_by_run = _run_shapes(shapes, baseline, run_one)

    # 8. Leaderboard + CIs (driver-owned labels — every empty-grid spec name is
    #    "default", so aggregate_results keys would collide).
    rows = [_row(run, metrics) for run, metrics in results_by_run.items()]
    _print_leaderboard(rows, results_by_run)

    # 9. Optional charts (needs the viz group): cross-axis + per-tier correctness.
    if plot:
        from geap_tuning.viz import plot_grouped_metric_bars  # noqa: PLC0415 - opt-in dep

        fig = plot_grouped_metric_bars(rows, metrics=AXES)
        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_PATH, bbox_inches="tight")
        print(f"\nSaved chart to {PLOT_PATH}")

        tier_rows = [_tier_row(run, metrics) for run, metrics in results_by_run.items()]
        tier_fig = plot_grouped_metric_bars(tier_rows, metrics=("easy", "medium", "hard"))
        tier_fig.savefig(TIER_PLOT_PATH, bbox_inches="tight")
        print(f"Saved per-tier chart to {TIER_PLOT_PATH}")

    # 10. The tuned shapes, read back from Experiments (baseline is offline-only).
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio → Experiments.")


if __name__ == "__main__":
    main()
