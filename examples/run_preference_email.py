"""End-to-end DPO example: concise professional email rewriting.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint,
not covered by the test suite (pytest only exercises the mocked units). Run it
with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_preference_email.py

Unlike ``run_preference.py`` (support-reply warmth), this teaches concise +
professional email rewriting and shows a **before → after** lift: it scores the
untuned base's win-rate against the dispreferred reference first, then tunes,
then scores the tuned endpoint. A blind A/B autorater (a different prompt from
the generator) grades tone/concision, corroborated by an objective compression
ratio. Candidate A is always the model under test; B is the dispreferred ref.
"""

from __future__ import annotations

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
from geap_tuning.preference.email_eval import run_email_eval
from geap_tuning.preference.tune import launch_preference_job

DISPLAY_NAME = "geap-dpo-concise-email"
BASE_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-2.5-flash"
DATA_DIR = Path("datasets/preference_concise_email")
GCS_PREFIX = "preference_concise_email"

_JUDGE_PROMPT = (
    "You are judging two versions of the same work email. Pick the version that "
    "is more professional and concise (clear, brief, free of filler and hedging) "
    "while keeping the same information. Answer with only the single letter 'A' "
    "or 'B'.\n\n"
    "Original draft: {user}\n\nEmail A: {a}\n\nEmail B: {b}\n\nBetter email:"
)


def main() -> None:
    """Run the full concise-email DPO workflow with a before → after comparison."""
    cfg = load_config()
    client = genai_client(cfg)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build the local dataset splits.
    paths = build_preference_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")

    # 2. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # A blind A/B autorater judge (base model, prompt distinct from the generator).
    def judge_fn(draft: str, cand_a: str, cand_b: str) -> str:
        verdict = generate(
            client,
            JUDGE_MODEL,
            _JUDGE_PROMPT.format(user=draft, a=cand_a, b=cand_b),
        )
        return verdict[:1].upper()

    _, _, test_triples = split_dataset(EMAIL_DRAFTS)
    test_records = build_preference_records(test_triples)

    # 3. Score the untuned base model first (the "before").
    base = run_email_eval(
        test_records,
        generate_fn=lambda draft: generate(
            client, BASE_MODEL, draft, system_instruction=SYSTEM_INSTRUCTION
        ),
        judge_fn=judge_fn,
    )
    print(
        f"BASE  win_rate={base['win_rate']:.3f} mean_compression={base['mean_compression']:.2f} "
        f"(n={base['n']})"
    )

    # 4. Reuse an existing job if one exists; otherwise launch.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_preference_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            base_model=BASE_MODEL,
            labels=cfg.labels,
        )
        print(f"Launched tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")

    # 5. Wait for completion and resolve the tuned endpoint.
    job = wait_for_tuning_job(client, job.name)
    endpoint = tuned_endpoint(job)
    print(f"Tuned endpoint: {endpoint}")

    # 6. Score the tuned endpoint (the "after") and report the lift.
    tuned = run_email_eval(
        test_records,
        generate_fn=lambda draft: generate(
            client, endpoint, draft, system_instruction=SYSTEM_INSTRUCTION
        ),
        judge_fn=judge_fn,
    )
    print(
        f"TUNED win_rate={tuned['win_rate']:.3f} mean_compression={tuned['mean_compression']:.2f} "
        f"(n={tuned['n']})"
    )
    print(
        f"LIFT  win_rate {base['win_rate']:.3f}→{tuned['win_rate']:.3f}; "
        f"compression {base['mean_compression']:.2f}→{tuned['mean_compression']:.2f}"
    )


if __name__ == "__main__":
    main()
