"""Experiment-tracking demo: log tuning params + offline metrics to Vertex AI Experiments.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units in
``geap_tuning.experiments``). Run it with a real ``.env`` and ``gcloud auth`` in
place:

    uv run python examples/run_experiment_tracking.py                # summary metrics only (free)
    uv run python examples/run_experiment_tracking.py --tensorboard  # + Managed TensorBoard curves

This is the opt-in "Layer 2" of experiment tracking (see
``docs/notes/experiment-tracking.md``): the automatic per-job train/val curves in
the console Monitor tab ("Layer 1") need no code. Here we reuse a **single**
SFT-with-checkpoints job (via its display name, so re-runs don't pay for a
duplicate), evaluate each exported checkpoint on the held-out test split, and:

1. log one Experiments **run per checkpoint** with its params + summary accuracy /
   macro-F1 — a cross-run comparison table, no TensorBoard required; and
2. with ``--tensorboard``, provision/attach a **Managed TensorBoard** (cost +
   ~10-20 min provisioning) and log an accuracy-vs-epoch **time-series** curve.

Tuning stays regional (``cfg.location``); ``gemini-2.5-flash`` supports
intermediate checkpoints. Experiments is a regional resource — keep its location
aligned with the tuning region.
"""

from __future__ import annotations

import sys
from pathlib import Path

from geap_tuning.config import genai_client, load_config
from geap_tuning.experiments import (
    experiment_dataframe,
    get_or_create_tensorboard,
    init_experiment,
    log_summary_metrics,
    log_timeseries_metrics,
    track_run,
)
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.jobs import (
    checkpoint_endpoint,
    find_tuning_job_by_display_name,
    list_checkpoints,
    wait_for_tuning_job,
)
from geap_tuning.sft.data import SUPPORT_TICKETS, build_records, build_sft_dataset, split_dataset
from geap_tuning.sft.evaluate import run_eval
from geap_tuning.sft.tune import launch_sft_job

DISPLAY_NAME = "geap-exp-tracking-sft"
EXPERIMENT_NAME = "geap-sft-checkpoint-eval"
TENSORBOARD_DISPLAY_NAME = "geap-tuning-tb"
DATA_DIR = Path("datasets/sft_support_intent")
GCS_PREFIX = "experiment_tracking_sft"
BASE_MODEL = "gemini-2.5-flash"
ADAPTER_SIZE = 8
EPOCHS = 3  # a few epochs → a few checkpoints → a few runs to compare


def main() -> None:
    """Run the full experiment-tracking workflow against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only
    use_tb = "--tensorboard" in sys.argv
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")
    if use_tb:
        print("TensorBoard: ON (provisioning may take ~10-20 min and incurs cost)")
    else:
        print("TensorBoard: OFF (summary metrics only; pass --tensorboard for time-series)")

    # 1. Build and stage the SFT dataset.
    paths = build_sft_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 2. Reuse or launch a single SFT job that exports intermediate checkpoints.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_sft_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            base_model=BASE_MODEL,
            epochs=EPOCHS,
            adapter_size=ADAPTER_SIZE,
            export_last_checkpoint_only=False,  # keep every checkpoint (default)
            labels=cfg.labels,
        )
        print(f"Launched tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")
    job = wait_for_tuning_job(client, job.name)

    # 3. Point Experiments (optionally + Managed TensorBoard) at the tuning region.
    tensorboard = None
    if use_tb:
        tensorboard = get_or_create_tensorboard(
            TENSORBOARD_DISPLAY_NAME, project=cfg.project, location=cfg.location, labels=cfg.labels
        )
        print(f"TensorBoard: {tensorboard}")
    init_experiment(
        EXPERIMENT_NAME, project=cfg.project, location=cfg.location, tensorboard=tensorboard
    )

    # 4. Evaluate every checkpoint once on the held-out test split.
    checkpoints = list_checkpoints(job)
    if not checkpoints:
        print("No intermediate checkpoints exported — nothing to track. Done.")
        return
    _, _, test_pairs = split_dataset(SUPPORT_TICKETS)
    test_records = build_records(test_pairs)
    results = []
    for cp in checkpoints:
        endpoint = checkpoint_endpoint(job, cp.checkpoint_id)
        metrics = run_eval(
            test_records,
            predict_fn=lambda user_text, e=endpoint: generate(client, e, user_text),
        )
        results.append((cp, metrics))
        print(f"checkpoint {cp.checkpoint_id} (epoch {cp.epoch}): acc={metrics['accuracy']:.3f}")

    # 5. Log one summary run per checkpoint (cross-run table; no TensorBoard needed).
    for cp, metrics in results:
        params = {
            "base_model": BASE_MODEL,
            "epochs": EPOCHS,
            "adapter_size": ADAPTER_SIZE,
            "checkpoint_id": cp.checkpoint_id,
            "epoch": cp.epoch,
            "step": cp.step,
        }
        with track_run(f"{DISPLAY_NAME}-cp-{cp.checkpoint_id}", params=params):
            log_summary_metrics({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})

    # 6. With TensorBoard, also log the accuracy-vs-epoch curve as one time-series run.
    if use_tb:
        with track_run(f"{DISPLAY_NAME}-curve"):
            for cp, metrics in sorted(results, key=lambda r: r[0].epoch):
                log_timeseries_metrics(
                    {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
                    step=cp.epoch,
                )

    # 7. Print the cross-run comparison table.
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio → Experiments.")
    if use_tb:
        print("Time-series curves live in the attached Managed TensorBoard.")


if __name__ == "__main__":
    main()
