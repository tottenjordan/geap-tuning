"""Advanced GEAP managed-evaluation example: multi-metric auto-eval during SFT.

REQUIRES LIVE GCP AND INCURS TUNING COST. Like the other ``run_*`` scripts, this
is an integration entrypoint, not covered by the test suite. Run with a real
``.env`` and ``gcloud auth``:

    uv run python examples/run_advanced_eval.py

GEAP's managed **Evaluation** service runs *inside* a tuning job: attach an
``EvaluationConfig`` (there is no standalone eval client in google-genai 2.14.0)
and GEAP evaluates each checkpoint and writes results to Cloud Storage. This
example reuses the SFT support-intent dataset (see ``run_sft.py``) and attaches a
**comprehensive** config that exercises every metric kind plus the judge and
inference knobs:

- **LLM-judge** metric — a judge model scores each response against a prompt
  (``llm_judge_metric``).
- **Computation** metrics — deterministic ``EXACT_MATCH`` and ``ROUGE`` against
  the reference text (``computation_metric``).
- **Predefined catalog** metric — a managed metric by name
  (``predefined_metric("text_quality_v1")``).
- **Autorater config** — tunes the shared judge model (``sampling_count`` etc.).
- **Inference generation config** — how the *tuned model* generates the responses
  being scored (``temperature=0.0`` for deterministic eval).

GEAP evaluates each exported checkpoint, so eval cadence follows checkpointing
(keep ``export_last_checkpoint_only=False``). NOTE: there is **no**
``evaluate_interval`` here — in google-genai 2.14.0 that field serializes only
under the reinforcement spec, so passing it on an SFT job 400s; it is RLFT-only
(see ``run_rlft_reward_types.py``).

NOTE: the eval service is **Preview and available in ``us-central1`` only**; the
SDK **lowercases** ``Metric.name``; and predefined metric names must exist in the
live catalog — verify before running.
"""

from __future__ import annotations

from pathlib import Path

from google.genai import types

from geap_tuning.autoeval import (
    build_autorater_config,
    build_evaluation_config,
    computation_metric,
    llm_judge_metric,
    predefined_metric,
)
from geap_tuning.config import genai_client, load_config
from geap_tuning.gcs import upload_file
from geap_tuning.jobs import (
    find_tuning_job_by_display_name,
    wait_for_tuning_job,
)
from geap_tuning.sft.data import build_sft_dataset
from geap_tuning.sft.tune import launch_sft_job

VERSION = "v1"
DISPLAY_NAME = f"geap-eval-{VERSION}"
DATA_DIR = Path("datasets/sft_support_intent")
GCS_PREFIX = "sft_support_intent"
EVAL_PREFIX = "advanced_eval"
EVAL_REGION = "us-central1"  # the managed eval service is us-central1-only (Preview)


def build_advanced_eval_config(bucket: str) -> types.EvaluationConfig:
    """Assemble the comprehensive multi-metric eval config used by this example."""
    metrics = [
        llm_judge_metric(
            "intent_correctness",
            "Does the response state the correct support intent for the ticket? "
            "Respond with a score from 1 (wrong) to 5 (exactly right).\n{prediction}",
        ),
        computation_metric(types.ComputationBasedMetricType.EXACT_MATCH),
        computation_metric(types.ComputationBasedMetricType.ROUGE),
        predefined_metric("text_quality_v1"),
    ]
    return build_evaluation_config(
        bucket,
        prefix=EVAL_PREFIX,
        metrics=metrics,
        autorater_config=build_autorater_config(sampling_count=4),
        # Deterministic generation so eval scores are comparable across checkpoints.
        inference_generation_config=types.GenerationConfig(temperature=0.0),
    )


def main() -> None:
    """Launch (or reuse) an SFT job with a comprehensive managed-eval config."""
    cfg = load_config()
    client = genai_client(cfg)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket} labels={cfg.labels}")
    if cfg.location != EVAL_REGION:
        print(
            f"WARNING: managed eval is {EVAL_REGION}-only (Preview); "
            f"cfg.location={cfg.location} may not run auto-eval."
        )

    # 1. Build + stage the SFT dataset (same as run_sft.py).
    paths = build_sft_dataset(DATA_DIR)
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 2. Assemble the comprehensive eval config.
    eval_config = build_advanced_eval_config(cfg.bucket)
    print(f"Eval metrics: {len(eval_config.metrics)}; results → {EVAL_PREFIX}/")

    # 3. Reuse an existing job if one exists; otherwise launch with eval attached.
    #    export_last_checkpoint_only=False (default) so eval runs per checkpoint.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_sft_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            evaluation_config=eval_config,
            labels=cfg.labels,
        )
        print(f"Launched tuning job with managed eval: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")

    # 4. Wait for completion and point at where eval results landed.
    job = wait_for_tuning_job(client, job.name)
    prefix = eval_config.output_config.gcs_destination.output_uri_prefix
    print(f"Tuning complete. Eval results under: {prefix}")
    print(f"Inspect with: gcloud storage ls {prefix}")


if __name__ == "__main__":
    main()
