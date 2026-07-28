"""Checkpointing demo: SFT with intermediate checkpoints → compare → reassign default.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units in
``geap_tuning.jobs``). Run it with a real ``.env`` and ``gcloud auth`` in place:

    uv run python examples/run_checkpoints.py

It reuses the SFT support-intent dataset and tunes ``gemini-2.5-flash`` with
``export_last_checkpoint_only=False`` (the default) over a few epochs, so GEAP
exports one checkpoint per epoch. It then:

1. lists every checkpoint (``checkpoint_id``/``epoch``/``step``/``endpoint``),
2. runs inference against two different checkpoints on a held-out ticket to show
   they can diverge, and
3. reads the model's default checkpoint and reassigns it to an earlier one — the
   default checkpoint is the one the model's base endpoint serves.

See ``docs/notes/checkpoints-and-continuous-tuning.md``. Tuning stays regional
(``cfg.location``); ``gemini-2.5-flash`` supports intermediate checkpoints.
"""

from __future__ import annotations

from pathlib import Path

from geap_tuning.config import genai_client, load_config
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.jobs import (
    checkpoint_endpoint,
    find_tuning_job_by_display_name,
    get_default_checkpoint_id,
    list_checkpoints,
    set_default_checkpoint,
    wait_for_tuning_job,
)
from geap_tuning.sft.data import SUPPORT_TICKETS, build_records, build_sft_dataset, split_dataset
from geap_tuning.sft.tune import launch_sft_job

DISPLAY_NAME = "geap-checkpoints-sft"
DATA_DIR = Path("datasets/sft_support_intent")
GCS_PREFIX = "checkpoints_sft"
BASE_MODEL = "gemini-2.5-flash"
EPOCHS = 3  # a few epochs → a few intermediate checkpoints to compare


def main() -> None:
    """Run the full checkpointing workflow against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")

    # 1. Build and stage the SFT dataset.
    paths = build_sft_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 2. Reuse or launch an SFT job that exports intermediate checkpoints.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_sft_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            base_model=BASE_MODEL,
            epochs=EPOCHS,
            export_last_checkpoint_only=False,  # keep every checkpoint (default)
        )
        print(f"Launched tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")
    job = wait_for_tuning_job(client, job.name)

    # 3. List the checkpoints the job exported.
    checkpoints = list_checkpoints(job)
    print(f"\n{len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        print(f"  id={cp.checkpoint_id} epoch={cp.epoch} step={cp.step} endpoint={cp.endpoint}")
    if len(checkpoints) < 2:  # noqa: PLR2004 - need two to compare
        print("Fewer than two checkpoints — nothing to compare. Done.")
        return

    # 4. Compare the first vs. last checkpoint on a held-out ticket.
    _, _, test_pairs = split_dataset(SUPPORT_TICKETS)
    sample_ticket = build_records(test_pairs[:1])[0]["contents"][0]["parts"][0]["text"]
    first_id, last_id = checkpoints[0].checkpoint_id, checkpoints[-1].checkpoint_id
    for cid in (first_id, last_id):
        reply = generate(client, checkpoint_endpoint(job, cid), sample_ticket)
        print(f"\n[checkpoint {cid}] {reply!r}")

    # 5. Read the default checkpoint, then reassign it to the earliest one.
    current = get_default_checkpoint_id(client, job)
    print(f"\nCurrent default checkpoint: {current}")
    set_default_checkpoint(client, job, first_id)
    print(f"Reassigned default checkpoint to: {get_default_checkpoint_id(client, job)}")


if __name__ == "__main__":
    main()
