"""End-to-end DPO example: data -> GCS -> tune -> wait -> evaluate.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint,
not covered by the test suite (pytest only exercises the mocked units). Run it
with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_preference.py

It builds the support-reply *style* preference dataset, stages the splits to GCS,
reuses an existing tuning job with the same display name if one exists (otherwise
launches a fresh DPO job with ``method="PREFERENCE_TUNING"`` and ``beta``), waits
for completion, then reports an autorater win-rate: how often the tuned model's
reply beats the dispreferred reference in a blind A/B judgment.
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
from geap_tuning.preference.data import (
    SUPPORT_REPLIES,
    build_preference_dataset,
    build_preference_records,
    split_dataset,
)
from geap_tuning.preference.evaluate import run_preference_eval
from geap_tuning.preference.tune import launch_preference_job

DISPLAY_NAME = "geap-dpo-support-style"
DATA_DIR = Path("datasets/preference_support_style")
GCS_PREFIX = "preference_support_style"
JUDGE_MODEL = "gemini-2.5-flash"

_JUDGE_PROMPT = (
    "You are judging two customer-support replies to the same message. Pick the "
    "reply that is warmer, more concise, and more helpful (acknowledges the issue "
    "and offers a next step). Answer with only the single letter 'A' or 'B'.\n\n"
    "Customer message: {user}\n\nReply A: {a}\n\nReply B: {b}\n\nBetter reply:"
)


def main() -> None:
    """Run the full DPO workflow against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")

    # 1. Build the local dataset splits.
    paths = build_preference_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")

    # 2. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 3. Reuse an existing job if one exists; otherwise launch.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_preference_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
        )
        print(f"Launched tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")

    # 4. Wait for completion and resolve the tuned endpoint.
    job = wait_for_tuning_job(client, job.name)
    endpoint = tuned_endpoint(job)
    print(f"Tuned endpoint: {endpoint}")

    # 5. Evaluate on the held-out test split with a base-model autorater judge.
    def judge_fn(user_text: str, cand_a: str, cand_b: str) -> str:
        verdict = generate(
            client,
            JUDGE_MODEL,
            _JUDGE_PROMPT.format(user=user_text, a=cand_a, b=cand_b),
        )
        return verdict[:1].upper()

    _, _, test_triples = split_dataset(SUPPORT_REPLIES)
    test_records = build_preference_records(test_triples)
    metrics = run_preference_eval(
        test_records,
        generate_fn=lambda user_text: generate(client, endpoint, user_text),
        judge_fn=judge_fn,
    )
    print(f"Tuned-vs-dispreferred win rate: {metrics['win_rate']:.3f} (n={metrics['n']})")


if __name__ == "__main__":
    main()
