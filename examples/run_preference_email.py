"""End-to-end DPO example: concise professional email rewriting.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint,
not covered by the test suite (pytest only exercises the mocked units). Run it
with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_preference_email.py --pilot-only  # gate only, no spend
    uv run python examples/run_preference_email.py               # gate, then tune

This teaches concise + professional email rewriting and shows an **honest**
before → after lift with two disciplines borrowed from the RLFT constrained demo:

1. A **pilot gate** (``--pilot-only`` stops for free). It judges the untuned base
   rewrite against the *gold preferred* reference; if the base already beats the
   concise gold at/above ``SAT_CEILING`` there is no headroom and it refuses to
   tune (``--force`` overrides).
2. A **head-to-head** before → after win-rate: the tuned rewrite versus the
   *base* rewrite of the same draft (not a fixed strawman), judged blind with a
   randomized A/B position, with a ``bootstrap_ci`` on the win-rate.

The judge prompt is distinct from the generator's system instruction, and both
completions in the dataset carry the same facts so the judge grades style.
"""

from __future__ import annotations

import sys
from pathlib import Path

from geap_tuning.config import genai_client, load_config
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.jobs import (
    find_tuning_job_by_display_name,
    tuned_endpoint,
    wait_for_tuning_job,
)
from geap_tuning.preference.email import (
    EMAIL_DRAFTS,
    SYSTEM_INSTRUCTION,
    build_preference_dataset,
    build_preference_records,
    split_dataset,
)
from geap_tuning.preference.email_eval import run_head_to_head_eval, run_pilot_eval
from geap_tuning.preference.tune import launch_preference_job
from geap_tuning.rlft.evaluate import bootstrap_ci

DISPLAY_NAME = "geap-dpo-concise-email-v2"  # v2: bigger/verbose bank + head-to-head eval
BASE_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-2.5-flash"
DATA_DIR = Path("datasets/preference_concise_email")
GCS_PREFIX = "preference_concise_email_v2"
# The base must NOT already beat the gold preferred rewrite this often — above this
# it writes as well as the human gold and tuning has little to teach.
SAT_CEILING = 0.6

_JUDGE_PROMPT = (
    "You are judging two versions of the same work email. Pick the version that "
    "is more professional and concise (clear, brief, free of filler and hedging) "
    "while keeping the same information. Answer with only the single letter 'A' "
    "or 'B'.\n\n"
    "Original draft: {user}\n\nEmail A: {a}\n\nEmail B: {b}\n\nBetter email:"
)


def _gate_ok(pilot: dict[str, object], *, force: bool) -> bool:
    """Print the pilot line and return whether the base has headroom vs the gold."""
    print(
        f"\nPilot gate — untuned {BASE_MODEL} vs gold: win_rate={pilot['win_rate']:.3f} "
        f"(ceiling {SAT_CEILING}) mean_compression={pilot['mean_compression']:.2f} (n={pilot['n']})"
    )
    if pilot["win_rate"] < SAT_CEILING or force:
        print("Pilot gate PASSED — the base trails the concise gold; headroom confirmed.\n")
        return True
    print(
        f"\nPILOT GATE FAILED: base win_rate vs gold must be < {SAT_CEILING} for tuning to show "
        "a lift (the base already writes as well as the gold). Author more verbose drafts or "
        "pass --force to launch anyway."
    )
    return False


def main() -> None:
    """Run the concise-email DPO before → after against live GEAP with a pilot gate."""
    cfg = load_config()
    client = genai_client(cfg)
    force = "--force" in sys.argv
    pilot_only = "--pilot-only" in sys.argv  # score the gate, then stop (no tuning spend)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build the local dataset splits.
    paths = build_preference_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")

    # A blind A/B autorater judge (base model, prompt distinct from the generator).
    def judge_fn(draft: str, cand_a: str, cand_b: str) -> str:
        verdict = generate(
            client, JUDGE_MODEL, _JUDGE_PROMPT.format(user=draft, a=cand_a, b=cand_b)
        )
        return verdict[:1].upper()

    def base_rewrite(draft: str) -> str:
        return generate(client, BASE_MODEL, draft, system_instruction=SYSTEM_INSTRUCTION)

    _, _, test_triples = split_dataset(EMAIL_DRAFTS)
    test_records = build_preference_records(test_triples)

    # 2. Pilot gate — base rewrite vs the gold preferred reference (the "before").
    pilot = run_pilot_eval(test_records, base_rewrite, judge_fn)
    if not _gate_ok(pilot, force=force):
        sys.exit(1)
    if pilot_only:
        print("--pilot-only: stopping before any tuning spend.")
        return

    # 3. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 4. Reuse an existing job if one exists; otherwise launch. A firmer pull toward
    # the preferred (shorter) completion than the defaults (epochs=2, beta=0.1).
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_preference_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            base_model=BASE_MODEL,
            epochs=3,
            beta=0.2,
            labels=cfg.labels,
        )
        print(f"Launched tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")

    # 5. Wait for completion and resolve the tuned endpoint.
    job = wait_for_tuning_job(client, job.name)
    endpoint = tuned_endpoint(job)
    print(f"Tuned endpoint: {endpoint}")

    def tuned_rewrite(draft: str) -> str:
        return generate(client, endpoint, draft, system_instruction=SYSTEM_INSTRUCTION)

    # 6. Head-to-head (the "after"): tuned rewrite vs base rewrite, blind A/B.
    h2h = run_head_to_head_eval(test_records, base_rewrite, tuned_rewrite, judge_fn)
    # And the tuned model's own standing against the gold, to show it closed the gap.
    tuned_pilot = run_pilot_eval(test_records, tuned_rewrite, judge_fn)
    low, high = bootstrap_ci(int(h2h["hits"]), int(h2h["n"]))
    print(
        f"\nHEAD-TO-HEAD tuned vs base: win_rate={h2h['win_rate']:.3f} "
        f"CI[{low:.3f}, {high:.3f}] (n={h2h['n']}); >0.5 means tuning helped"
    )
    print(
        f"compression base={h2h['base_mean_compression']:.2f} "
        f"tuned={h2h['tuned_mean_compression']:.2f}"
    )
    print(
        f"vs gold: base win_rate={pilot['win_rate']:.3f} -> "
        f"tuned win_rate={tuned_pilot['win_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
