"""End-to-end generative SFT example: messy text → normalized JSON extraction.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint,
not covered by the test suite (pytest only exercises the mocked units). Run it
with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_sft_extraction.py --pilot-only  # gate only, no spend
    uv run python examples/run_sft_extraction.py               # gate, then tune

Unlike ``run_sft.py`` (classification), this demonstrates a **before → after**
lift on a generative task. Plain field extraction saturates a modern base (it
scored a perfect ``accuracy`` in a live run), so the task instead teaches a
**house normalization standard the base cannot guess** (P-code priorities, city
abbreviation expansion, ``ord-`` prefix stripping, spelled-out quantities). A
**pilot gate** scores the untuned base first (``--pilot-only`` stops for free);
it proceeds only if the base ``accuracy`` is below ``SAT_CEILING`` — real headroom
to teach — and ``--force`` overrides. Then it tunes, scores the tuned endpoint,
and prints the field-exact-match gain plus a per-field breakdown.
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
from geap_tuning.sft.extraction import (
    EXTRACTION_EXAMPLES,
    SYSTEM_INSTRUCTION,
    build_extraction_dataset,
    build_records,
    split_dataset,
)
from geap_tuning.sft.extraction_eval import run_eval
from geap_tuning.sft.tune import launch_sft_job

# v2: the normalization-convention redesign (the v1 plain-extraction task saturated).
DISPLAY_NAME = "geap-sft-json-extraction-v2"
BASE_MODEL = "gemini-2.5-flash"
DATA_DIR = Path("datasets/sft_json_extraction_v2")
GCS_PREFIX = "sft_json_extraction_v2"
# The base must score BELOW this field-exact-match accuracy to have headroom worth
# tuning. Above it, the base already applies our convention and there is little to teach.
SAT_CEILING = 0.85


def _predict_fn(client: object, target: str):  # noqa: ANN202 - returns a closure
    """Return a ``predict_fn`` that queries ``target`` (base model or endpoint)."""
    return lambda user_text: generate(
        client, target, user_text, system_instruction=SYSTEM_INSTRUCTION
    )


def _gate_ok(base: dict[str, object], *, force: bool) -> bool:
    """Print the pilot line and return whether the base has headroom to tune."""
    accuracy = float(base["accuracy"])  # type: ignore[arg-type]
    print(
        f"\nPilot gate — untuned {BASE_MODEL}: accuracy={accuracy:.3f} (ceiling {SAT_CEILING}) "
        f"exact_match={base['exact_match']:.3f} json_validity={base['json_validity']:.3f}"
    )
    print(f"  per_field: {base['per_field']}")
    if accuracy < SAT_CEILING or force:
        print("Pilot gate PASSED — the base does not know our convention; headroom confirmed.\n")
        return True
    print(
        f"\nPILOT GATE FAILED: base accuracy must be < {SAT_CEILING} for SFT to show a lift "
        "(the base already applies our normalization standard). Tighten the convention or "
        "pass --force to launch anyway."
    )
    return False


def main() -> None:
    """Run the full generative-SFT workflow with a pilot gate and before → after."""
    cfg = load_config()
    client = genai_client(cfg)
    force = "--force" in sys.argv
    pilot_only = "--pilot-only" in sys.argv  # score the gate, then stop (no tuning spend)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")

    # 1. Build the local dataset splits.
    paths = build_extraction_dataset(DATA_DIR)
    print(f"Wrote splits: {paths}")

    # 2. Score the untuned base model first (the "before") and gate on headroom.
    _, _, test = split_dataset(EXTRACTION_EXAMPLES)
    test_records = build_records(test)
    base = run_eval(test_records, predict_fn=_predict_fn(client, BASE_MODEL))
    if not _gate_ok(base, force=force):
        sys.exit(1)
    if pilot_only:
        print("--pilot-only: stopping before any tuning spend.")
        return

    # 3. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

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
    print(f"  per_field: {tuned['per_field']}")
    print(
        f"LIFT  accuracy +{tuned['accuracy'] - base['accuracy']:.3f}; "
        f"exact_match +{tuned['exact_match'] - base['exact_match']:.3f}; "
        f"json_validity {base['json_validity']:.3f}→{tuned['json_validity']:.3f}"
    )


if __name__ == "__main__":
    main()
