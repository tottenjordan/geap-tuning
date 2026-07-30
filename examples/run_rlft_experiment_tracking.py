"""RLFT experiment-tracking demo: log tuning params + offline metrics to Vertex AI.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units in
``geap_tuning.experiments`` / ``geap_tuning.jobs``). Run it with a real ``.env`` and
``gcloud auth`` in place:

    uv run python examples/run_rlft_experiment_tracking.py                # summary metrics only
    uv run python examples/run_rlft_experiment_tracking.py --tensorboard  # + TensorBoard curves

This is the **RLFT** counterpart of ``examples/run_experiment_tracking.py`` (SFT) —
the opt-in "Layer 2" of experiment tracking (see
``docs/notes/experiment-tracking.md``): the automatic per-job train/val curves in the
console Monitor tab ("Layer 1") need no code and are not SDK-fetchable, so the metric
**values** here come from our own offline per-checkpoint eval, not job-emitted reward.

We reuse a **single** cheap RLFT-with-checkpoints job (via its display name, so re-runs
don't pay for a duplicate), preflight the declarative string-match reward, evaluate each
exported checkpoint on the held-out test split (reward-scored answer accuracy), and:

1. log one Experiments **run per checkpoint** with its params + summary accuracy — a
   cross-run comparison table, no TensorBoard required; and
2. with ``--tensorboard``, provision/attach a **Managed TensorBoard** (cost +
   ~10-20 min provisioning) and log an accuracy-vs-epoch **time-series** curve.

The job stays **regional** (``cfg.location``): the ``global`` endpoint that serves
Gemini inference does not support tuning, so tuning and ``validate_reward`` must run in
a region. ``export_last_checkpoint_only=False`` is what keeps per-checkpoint endpoints
around for the curve — with only the final checkpoint the curve is a single point.
``gemini-2.5-flash`` at a few epochs is chosen for cost; swap the reward via the
builders in ``geap_tuning.rlft.tune`` if you want a correctness-based signal.
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
from geap_tuning.rlft.data import (
    MATH_PROBLEMS,
    build_rlft_dataset,
    build_rlft_records,
    split_dataset,
)
from geap_tuning.rlft.evaluate import run_rlft_eval
from geap_tuning.rlft.tune import (
    build_string_match_reward_config,
    launch_rlft_job,
    validate_reward_config,
)

DISPLAY_NAME = "geap-rlft-exp-tracking"
EXPERIMENT_NAME = "geap-rlft-checkpoint-eval"
TENSORBOARD_DISPLAY_NAME = "geap-tuning-tb"  # shared with the SFT tracking demo
DATA_DIR = Path("datasets/rlft_math")
GCS_PREFIX = "experiment_tracking_rlft"
BASE_MODEL = "gemini-2.5-flash"  # cheap; a few epochs -> a few checkpoints to compare
EPOCHS = 3
ADAPTER_SIZE = 16
SAMPLES_PER_PROMPT = 4


def main() -> None:
    """Run the full RLFT experiment-tracking workflow against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; global excludes tuning
    use_tb = "--tensorboard" in sys.argv
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")
    if use_tb:
        print("TensorBoard: ON (provisioning may take ~10-20 min and incurs cost)")
    else:
        print("TensorBoard: OFF (summary metrics only; pass --tensorboard for time-series)")

    # 1. Build and stage the RLFT dataset.
    paths = build_rlft_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 2. Preflight the declarative string-match reward before spending money.
    reward = build_string_match_reward_config()
    train_records = build_rlft_records(split_dataset(MATH_PROBLEMS)[0])
    preflight = validate_reward_config(
        client,
        project=cfg.project,
        location=cfg.location,
        sample_answer="Answer: 4",
        example_record=train_records[0],
        reward_config=reward,
    )
    print(f"Reward preflight: {preflight}")

    # 3. Reuse or launch a single RLFT job that exports intermediate checkpoints.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_rlft_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            base_model=BASE_MODEL,
            epochs=EPOCHS,
            adapter_size=ADAPTER_SIZE,
            samples_per_prompt=SAMPLES_PER_PROMPT,
            reward_config=reward,
            export_last_checkpoint_only=False,  # keep every checkpoint for the curve
            labels=cfg.labels,
        )
        print(f"Launched tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")
    job = wait_for_tuning_job(client, job.name)

    # 4. Point Experiments (optionally + Managed TensorBoard) at the tuning region.
    tensorboard = None
    if use_tb:
        tensorboard = get_or_create_tensorboard(
            TENSORBOARD_DISPLAY_NAME, project=cfg.project, location=cfg.location, labels=cfg.labels
        )
        print(f"TensorBoard: {tensorboard}")
    init_experiment(
        EXPERIMENT_NAME, project=cfg.project, location=cfg.location, tensorboard=tensorboard
    )

    # 5. Evaluate every checkpoint once on the held-out test split (reward-scored).
    checkpoints = list_checkpoints(job)
    if not checkpoints:
        print("No intermediate checkpoints exported — nothing to track. Done.")
        return
    _, _, test_problems = split_dataset(MATH_PROBLEMS)
    test_records = build_rlft_records(test_problems)
    results = []
    for cp in checkpoints:
        endpoint = checkpoint_endpoint(job, cp.checkpoint_id)
        metrics = run_rlft_eval(
            test_records,
            generate_fn=lambda user_text, e=endpoint: generate(client, e, user_text),
        )
        results.append((cp, metrics))
        print(f"checkpoint {cp.checkpoint_id} (epoch {cp.epoch}): acc={metrics['accuracy']:.3f}")

    # 6. Log one summary run per checkpoint (cross-run table; no TensorBoard needed).
    for cp, metrics in results:
        params = {
            "base_model": BASE_MODEL,
            "epochs": EPOCHS,
            "adapter_size": ADAPTER_SIZE,
            "samples_per_prompt": SAMPLES_PER_PROMPT,
            "reward": "string_match",  # scalar label, never the reward object
            "checkpoint_id": cp.checkpoint_id,
            "epoch": cp.epoch,
            "step": cp.step,
        }
        with track_run(f"{DISPLAY_NAME}-cp-{cp.checkpoint_id}", params=params):
            log_summary_metrics({"accuracy": metrics["accuracy"]})

    # 7. With TensorBoard, also log the accuracy-vs-epoch curve as one time-series run.
    if use_tb:
        with track_run(f"{DISPLAY_NAME}-curve"):
            for cp, metrics in sorted(results, key=lambda r: r[0].epoch):
                log_timeseries_metrics({"accuracy": metrics["accuracy"]}, step=cp.epoch)

    # 8. Print the cross-run comparison table.
    print(f"\nExperiment '{EXPERIMENT_NAME}' runs:")
    print(experiment_dataframe(EXPERIMENT_NAME))
    print("\nView in Agent Platform Studio → Experiments.")
    if use_tb:
        print("Time-series curves live in the attached Managed TensorBoard.")


if __name__ == "__main__":
    main()
