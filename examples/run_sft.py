"""End-to-end SFT example: data → GCS → tune → wait → evaluate.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint,
not covered by the test suite (pytest only exercises the mocked units). Run it
with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_sft.py

It builds the support-intent dataset, stages the splits to GCS, reuses an
existing tuning job with the same display name if one exists (otherwise launches
a fresh one), waits for completion, then reports classification accuracy from
the tuned endpoint on the held-out test split.
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
from geap_tuning.sft.data import (
    SUPPORT_TICKETS,
    build_records,
    build_sft_dataset,
    split_dataset,
)
from geap_tuning.sft.evaluate import run_eval
from geap_tuning.sft.tune import launch_sft_job

DISPLAY_NAME = "geap-sft-support-intent"
DATA_DIR = Path("datasets/sft_support_intent")
GCS_PREFIX = "sft_support_intent"


def main() -> None:
    """Run the full SFT workflow against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")

    # 1. Build the local dataset splits.
    paths = build_sft_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")

    # 2. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 3. Reuse an existing job if one exists; otherwise launch.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_sft_job(
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

    # 5. Evaluate on the held-out test split.
    _, _, test_pairs = split_dataset(SUPPORT_TICKETS)
    test_records = build_records(test_pairs)
    metrics = run_eval(
        test_records,
        predict_fn=lambda user_text: generate(client, endpoint, user_text),
    )
    print(f"Test accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
