"""End-to-end generative SFT example: messy text → strict JSON extraction.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint,
not covered by the test suite (pytest only exercises the mocked units). Run it
with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_sft_extraction.py

Unlike ``run_sft.py`` (classification), this demonstrates a **before → after**
lift on a generative task: it first scores the untuned base model on the
held-out test split, then tunes, then scores the tuned endpoint, and prints the
field-exact-match accuracy gain. The base tends to add prose / code fences and
emit ``quantity`` as a string, so there is real headroom.
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
from geap_tuning.sft.extraction import (
    EXTRACTION_EXAMPLES,
    SYSTEM_INSTRUCTION,
    build_extraction_dataset,
    build_records,
    split_dataset,
)
from geap_tuning.sft.extraction_eval import run_eval
from geap_tuning.sft.tune import launch_sft_job

DISPLAY_NAME = "geap-sft-json-extraction"
BASE_MODEL = "gemini-2.5-flash"
DATA_DIR = Path("datasets/sft_json_extraction")
GCS_PREFIX = "sft_json_extraction"


def _predict_fn(client: object, target: str):  # noqa: ANN202 - returns a closure
    """Return a ``predict_fn`` that queries ``target`` (base model or endpoint)."""
    return lambda user_text: generate(
        client, target, user_text, system_instruction=SYSTEM_INSTRUCTION
    )


def main() -> None:
    """Run the full generative-SFT workflow with a before → after comparison."""
    cfg = load_config()
    client = genai_client(cfg)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build the local dataset splits.
    paths = build_extraction_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")

    # 2. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 3. Score the untuned base model first (the "before").
    _, _, test = split_dataset(EXTRACTION_EXAMPLES)
    test_records = build_records(test)
    base = run_eval(test_records, predict_fn=_predict_fn(client, BASE_MODEL))
    print(
        f"BASE  accuracy={base['accuracy']:.3f} exact_match={base['exact_match']:.3f} "
        f"json_validity={base['json_validity']:.3f}"
    )

    # 4. Reuse an existing job if one exists; otherwise launch.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_sft_job(
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
    tuned = run_eval(test_records, predict_fn=_predict_fn(client, endpoint))
    print(
        f"TUNED accuracy={tuned['accuracy']:.3f} exact_match={tuned['exact_match']:.3f} "
        f"json_validity={tuned['json_validity']:.3f}"
    )
    print(
        f"LIFT  accuracy +{tuned['accuracy'] - base['accuracy']:.3f}; "
        f"exact_match +{tuned['exact_match'] - base['exact_match']:.3f}; "
        f"json_validity {base['json_validity']:.3f}→{tuned['json_validity']:.3f}"
    )


if __name__ == "__main__":
    main()
