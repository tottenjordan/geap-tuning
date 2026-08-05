"""End-to-end DPO example: concise professional email rewriting.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint,
not covered by the test suite (pytest only exercises the mocked units). Run it
with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_preference_email.py --pilot-only  # gate only, no spend
    uv run python examples/run_preference_email.py               # gate, then tune

This teaches concise + professional email rewriting and shows an **honest**
before → after lift with two disciplines borrowed from the RLFT constrained demo:

1. A **pilot gate** (``--pilot-only`` stops for free). It scores the untuned base's
   objective concision (``mean_compression`` = rewrite/draft word ratio); if the
   base already compresses aggressively (below ``MIN_BASE_COMPRESSION``) there is
   no headroom and it refuses to tune (``--force`` overrides).
2. A **head-to-head** before → after. The **headline is objective concision**: the
   base vs tuned ``mean_compression`` and a ``compression_win_rate`` (fraction of
   drafts where the tuned rewrite is strictly shorter than the base rewrite) with a
   ``bootstrap_ci`` — this is the exact axis the preference pairs train, so it moves
   reliably. A blind A/B **subjective** judge win-rate rides along as a secondary
   signal; a strong base can saturate it (ours already writes emails the judge
   prefers to our gold ~87% of the time while *expanding* drafts), so it can stay
   flat even as compression clearly improves — an honest lesson in what DPO moves.

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
# Headroom gate on OBJECTIVE concision: the base's rewrite/draft word ratio must be
# at least this high (i.e. it is not already compressing hard) for DPO toward the
# shorter preferred completion to have room to teach. Our base sits at ~1.14 (it
# expands drafts), so this passes; a base already at ~0.6 would have little to gain.
MIN_BASE_COMPRESSION = 0.9

_JUDGE_PROMPT = (
    "You are judging two versions of the same work email. Pick the version that "
    "is more professional and concise (clear, brief, free of filler and hedging) "
    "while keeping the same information. Answer with only the single letter 'A' "
    "or 'B'.\n\n"
    "Original draft: {user}\n\nEmail A: {a}\n\nEmail B: {b}\n\nBetter email:"
)


def _gate_ok(pilot: dict[str, object], *, force: bool) -> bool:
    """Print the pilot line and return whether the base has objective concision headroom."""
    compression = float(pilot["mean_compression"])  # type: ignore[arg-type]
    print(
        f"\nPilot gate — untuned {BASE_MODEL}: mean_compression={compression:.2f} "
        f"(floor {MIN_BASE_COMPRESSION}); subjective base-vs-gold win_rate="
        f"{pilot['win_rate']:.3f} (context only) (n={pilot['n']})"
    )
    if compression >= MIN_BASE_COMPRESSION or force:
        print(
            "Pilot gate PASSED — the base is not concise (it barely shortens, or expands, "
            "the draft); real concision headroom.\n"
        )
        return True
    print(
        f"\nPILOT GATE FAILED: base mean_compression must be >= {MIN_BASE_COMPRESSION} for DPO to "
        "have concision to teach (the base already compresses aggressively). Pass --force to "
        "launch anyway."
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

    # 6. Head-to-head (the "after"): tuned rewrite vs base rewrite.
    h2h = run_head_to_head_eval(test_records, base_rewrite, tuned_rewrite, judge_fn)
    # HEADLINE: objective concision. compression_hits (tuned shorter than base) is binomial.
    low, high = bootstrap_ci(int(h2h["compression_hits"]), int(h2h["n"]))
    print(
        f"\nHEADLINE (objective concision): mean_compression "
        f"base={h2h['base_mean_compression']:.2f} -> tuned={h2h['tuned_mean_compression']:.2f} "
        f"(lower is more concise)"
    )
    print(
        f"tuned shorter than base in {int(h2h['compression_hits'])}/{int(h2h['n'])} "
        f"(compression_win_rate={h2h['compression_win_rate']:.3f} CI[{low:.3f}, {high:.3f}])"
    )
    print(
        f"SECONDARY (subjective judge): tuned-vs-base win_rate={h2h['win_rate']:.3f} "
        "(a strong base can hold this flat even as concision improves)"
    )


if __name__ == "__main__":
    main()
