"""End-to-end RLFT example: constrained generation with a GRADED reward.

REQUIRES LIVE GCP AND INCURS TUNING COST (one RLFT job). This is an integration
entrypoint, not covered by the test suite (pytest exercises the mocked units in
``geap_tuning.rlft.constrained`` / ``constraint_reward`` / ``constraint_eval`` and
the launcher/preflight in ``geap_tuning.rlft.tune``). Run it with a real ``.env``
and ``gcloud auth`` in place:

    uv run python examples/run_rlft_constrained.py --pilot-only  # gate only, no spend
    uv run python examples/run_rlft_constrained.py               # run + before/after

Each held-out prompt spells out four kinds of constraint at once (required
keywords, forbidden filler words, a word-count band, a sentence-count band). The
reward in :mod:`geap_tuning.rlft.constraint_reward` is *graded* — the fraction of
independently-checked components satisfied — which answers the two prior RLFT null
results (see ``docs/doe/rlft-reward-ranking/README.md``): the graded signal has
variance even for a mediocre rollout, and satisfying four constraint types at once
leaves headroom even for a strong base.

A **pilot gate** scores the untuned base first and refuses to launch unless the
base has headroom (constraint ``accuracy`` below ``SAT_CEILING``). ``--pilot-only``
stops after the gate (no tuning spend); ``--force`` launches even if the gate fails.

NOTE: ``gemini-3.5-flash`` is the only base GEAP documents as RLFT-supported. The
tuning client stays **regional** (``cfg.location``) — the ``global`` endpoint that
serves Gemini 3.x *inference* does not support tuning — while the untuned baseline
runs against that global inference endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

from geap_tuning.config import genai_client, genai_client_for_endpoint, load_config
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.jobs import (
    find_tuning_job_by_display_name,
    tuned_endpoint,
    wait_for_tuning_job,
)
from geap_tuning.rlft import constraint_reward
from geap_tuning.rlft.constrained import (
    CONSTRAINT_SPECS,
    build_constrained_dataset,
    build_records,
    split_dataset,
)
from geap_tuning.rlft.constraint_eval import run_eval
from geap_tuning.rlft.evaluate import bootstrap_ci
from geap_tuning.rlft.tune import build_reward_config, launch_rlft_job, validate_reward_config

DISPLAY_NAME = "geap-rlft-constrained"
DATA_DIR = Path("datasets/rlft_constrained")
GCS_PREFIX = "rlft_constrained"
BASE_MODEL = "gemini-3.5-flash"  # only RLFT-supported base
SAT_CEILING = 0.85  # base constraint accuracy must be below this to have headroom


def _print_metrics(label: str, metrics: dict[str, object]) -> None:
    """Print the headline accuracy + full-satisfaction rate + per-type rates."""
    print(
        f"{label}: accuracy={metrics['accuracy']:.3f} "
        f"full_satisfaction_rate={metrics['full_satisfaction_rate']:.3f} "
        f"(n={metrics['n']})"
    )
    for kind, stats in metrics["by_constraint_type"].items():
        print(f"    {kind:>15}: {stats['rate']:.3f} ({stats['satisfied']}/{stats['total']})")


def _gate_ok(baseline: dict[str, object], *, force: bool) -> bool:
    """Print the pilot line and return whether the base has constraint headroom."""
    print(
        f"\nPilot gate — untuned {BASE_MODEL}: accuracy={baseline['accuracy']:.3f} "
        f"(ceiling {SAT_CEILING})"
    )
    if baseline["accuracy"] < SAT_CEILING or force:
        print("Pilot gate PASSED — constraint headroom confirmed; launching the job.\n")
        return True
    print(
        f"\nPILOT GATE FAILED: baseline accuracy must be < {SAT_CEILING} for tuning to show "
        "a lift (the base already satisfies most constraints). Tighten the constraints "
        "(more simultaneous types / narrower bands) or pass --force to launch anyway."
    )
    return False


def main() -> None:
    """Run the constrained-generation RLFT before→after against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; global excludes tuning
    force = "--force" in sys.argv
    pilot_only = "--pilot-only" in sys.argv  # score the gate, then stop (no tuning spend)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build the local dataset splits (neutral instruction; graded-reward references).
    paths = build_constrained_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")
    train_specs, _, test_specs = split_dataset(CONSTRAINT_SPECS)
    train_records = build_records(train_specs)
    test_records = build_records(test_specs)
    print(f"Bench: {len(train_records)} train, {len(test_records)} test prompts")

    # 2. Preflight the graded reward on one training record before spending money.
    reward_cfg = build_reward_config("constraint_satisfaction", module=constraint_reward)
    preflight = validate_reward_config(
        client,
        project=cfg.project,
        location=cfg.location,
        sample_answer="A short reply that mentions the required keywords.",
        example_record=train_records[0],
        reward_config=reward_cfg,
    )
    print(f"Reward preflight: {preflight}")

    # 3. Pilot gate — score the UNTUNED base before spending on a job. Gemini 3.x
    #    inference runs on the global endpoint, so route the baseline there.
    base_client = genai_client(cfg, base_model=BASE_MODEL)
    baseline = run_eval(
        test_records,
        generate_fn=lambda user_text, system_instruction: generate(
            base_client, BASE_MODEL, user_text, system_instruction=system_instruction
        ),
    )
    _print_metrics(f"Untuned {BASE_MODEL}", baseline)
    if not _gate_ok(baseline, force=force):
        return
    if pilot_only:
        print("--pilot-only: stopping before any tuning spend.")
        return

    # 4. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 5. Reuse an existing job if one exists; otherwise launch.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_rlft_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            base_model=BASE_MODEL,
            reward_config=reward_cfg,
            labels=cfg.labels,
        )
        print(f"Launched tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")

    # 6. Wait for completion and resolve the tuned endpoint.
    job = wait_for_tuning_job(client, job.name)
    endpoint = tuned_endpoint(job)
    print(f"Tuned endpoint: {endpoint}")

    # 7. Evaluate the tuned endpoint on the held-out split (its own (multi-)region).
    eval_client = genai_client_for_endpoint(cfg, endpoint)
    tuned = run_eval(
        test_records,
        generate_fn=lambda user_text, system_instruction: generate(
            eval_client, endpoint, user_text, system_instruction=system_instruction
        ),
    )
    _print_metrics("Tuned", tuned)

    # 8. Before→after headline lift + a bootstrap CI on the full-satisfaction rate.
    lift = tuned["accuracy"] - baseline["accuracy"]
    print(
        f"\nLIFT: accuracy base={baseline['accuracy']:.3f} tuned={tuned['accuracy']:.3f} "
        f"(+{lift:.3f})"
    )
    base_low, base_high = bootstrap_ci(int(baseline["full_satisfaction_hits"]), int(baseline["n"]))
    tuned_low, tuned_high = bootstrap_ci(int(tuned["full_satisfaction_hits"]), int(tuned["n"]))
    print(
        f"full_satisfaction_rate base={baseline['full_satisfaction_rate']:.3f} "
        f"CI[{base_low:.3f}, {base_high:.3f}] -> tuned={tuned['full_satisfaction_rate']:.3f} "
        f"CI[{tuned_low:.3f}, {tuned_high:.3f}]"
    )


if __name__ == "__main__":
    main()
