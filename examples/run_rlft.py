"""End-to-end RLFT example: data -> validate reward -> GCS -> tune -> wait -> evaluate.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units). Run it with a
real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_rlft.py

It builds the verifiable-math dataset, preflights the code-execution reward on one
example via ``tunings.validate_reward`` (aborting if the reward errors), stages the
splits to GCS, reuses an existing tuning job with the same display name if one
exists (otherwise launches a fresh RLFT job with ``method="REINFORCEMENT_TUNING"``
and a code-execution ``reward_config``), waits for completion, then reports
held-out answer accuracy scored by the same reward function.

NOTE: the RLFT docs call out ``gemini-3.5-flash`` as the supported base model —
verify availability in your region before running live. This example tunes
``gemini-3.5-flash``, but the client stays **regional** (``cfg.location``): the
``global`` endpoint that serves Gemini 3.x *inference* does not support tuning, so
tuning jobs and ``validate_reward`` must run in a region (``us-central1`` /
``europe-west4``).
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
from geap_tuning.rlft.data import (
    MATH_PROBLEMS,
    build_rlft_dataset,
    build_rlft_records,
    split_dataset,
)
from geap_tuning.rlft.evaluate import run_rlft_eval
from geap_tuning.rlft.tune import launch_rlft_job, validate_reward_config

DISPLAY_NAME = "geap-rlft-math"
DATA_DIR = Path("datasets/rlft_math")
GCS_PREFIX = "rlft_math"
BASE_MODEL = "gemini-3.5-flash"


def main() -> None:
    """Run the full RLFT workflow against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; global excludes tuning
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build the local dataset splits.
    paths = build_rlft_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")

    # 2. Preflight the reward on the first training record before spending money.
    train_records = build_rlft_records(split_dataset(MATH_PROBLEMS)[0])
    preflight = validate_reward_config(
        client,
        project=cfg.project,
        location=cfg.location,
        sample_answer="Answer: 4",
        example_record=train_records[0],
    )
    print(f"Reward preflight: {preflight}")

    # 3. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 4. Reuse an existing job if one exists; otherwise launch.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_rlft_job(
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

    # 6. Evaluate on the held-out test split, reward-scoring each answer.
    _, _, test_problems = split_dataset(MATH_PROBLEMS)
    test_records = build_rlft_records(test_problems)
    metrics = run_rlft_eval(
        test_records,
        generate_fn=lambda user_text: generate(client, endpoint, user_text),
    )
    print(f"Held-out answer accuracy: {metrics['accuracy']:.3f} (n={metrics['n']})")


if __name__ == "__main__":
    main()
