"""Continuous tuning: SFT seed -> RLFT continued from it (same math domain).

REQUIRES LIVE GCP AND INCURS TUNING COST (two jobs). This is an integration
entrypoint, not covered by the test suite. Run it with a real ``.env`` and
``gcloud auth`` in place:

    uv run python examples/run_continuous_tuning.py

The documented GEAP best practice is **SFT first, then continuous-tune** — teach
the skill/format supervised, then refine with RL. This demo:

1. **Stage 1 (SFT seed):** tunes ``gemini-3.5-flash`` on the math SFT seed
   (``build_math_sft_dataset`` — the same ``MATH_PROBLEMS``, with a gold
   ``Answer: <n>`` turn) to teach the answer format.
2. **Stage 2 (RLFT continues):** launches an RLFT job whose ``base_model`` is the
   *stage-1 tuned model's resource name* (``projects/.../models/id@ver``). The
   Gen AI SDK auto-detects the ``projects/`` prefix as a pre-tuned model, so no
   extra plumbing is needed. RLFT then refines *correctness* on the references
   dataset.
3. **Compare:** reports held-out math accuracy for the SFT-seed model vs. the
   continuously-tuned RLFT model to show the lift.

Constraints (see ``docs/notes/checkpoints-and-continuous-tuning.md``): continuous
tuning is **Gen AI SDK / Vertex only**; the base SFT model must have been tuned
on/after 2025-07-11; both stages run in the **same region**; tuning stays
regional (never ``global``). ``VERSION`` parameterizes the display names so reruns
reuse the same jobs.
"""

from __future__ import annotations

from pathlib import Path

from geap_tuning.config import genai_client, load_config
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.jobs import (
    find_tuning_job_by_display_name,
    tuned_endpoint,
    tuned_model_name,
    wait_for_tuning_job,
)
from geap_tuning.rlft.data import (
    MATH_PROBLEMS,
    build_math_sft_dataset,
    build_rlft_dataset,
    build_rlft_records,
    split_dataset,
)
from geap_tuning.rlft.evaluate import run_rlft_eval
from geap_tuning.rlft.tune import launch_rlft_job, validate_reward_config
from geap_tuning.sft.tune import launch_sft_job

VERSION = "v1"
BASE_MODEL = "gemini-3.5-flash"
SFT_DIR = Path("datasets/math_sft")
RLFT_DIR = Path("datasets/rlft_math")


def _accuracy(client: object, endpoint: str) -> dict[str, object]:
    """Held-out math accuracy of ``endpoint`` (reward-scored, same as RLFT eval)."""
    _, _, test_problems = split_dataset(MATH_PROBLEMS)
    return run_rlft_eval(
        build_rlft_records(test_problems),
        generate_fn=lambda user_text: generate(client, endpoint, user_text),
    )


def main() -> None:
    """Run the SFT -> RLFT continuous-tuning workflow against live GEAP."""
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; both stages share it
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")

    # === Stage 1: SFT seed (teach the "Answer: <n>" format) ===
    sft_paths = build_math_sft_dataset(SFT_DIR)
    sft_train = upload_file(sft_paths["train"], f"{cfg.bucket}/cont_sft/train.jsonl")
    sft_val = upload_file(sft_paths["val"], f"{cfg.bucket}/cont_sft/val.jsonl")
    sft_name = f"geap-cont-sft-{VERSION}"

    sft_job = find_tuning_job_by_display_name(client, sft_name)
    if sft_job is None:
        sft_job = launch_sft_job(
            client,
            train_uri=sft_train,
            val_uri=sft_val,
            display_name=sft_name,
            base_model=BASE_MODEL,
        )
        print(f"Launched SFT seed job: {sft_job.name}")
    else:
        print(f"Reusing SFT seed job: {sft_job.name} ({sft_job.state})")
    sft_job = wait_for_tuning_job(client, sft_job.name)
    sft_model = tuned_model_name(sft_job)  # projects/.../models/id@ver
    print(f"SFT seed model: {sft_model}")

    # === Stage 2: RLFT continued from the SFT model ===
    rlft_paths = build_rlft_dataset(RLFT_DIR)
    train_records = build_rlft_records(split_dataset(MATH_PROBLEMS)[0])
    preflight = validate_reward_config(
        client,
        project=cfg.project,
        location=cfg.location,
        sample_answer="Answer: 4",
        example_record=train_records[0],
    )
    print(f"Reward preflight: {preflight}")

    rlft_train = upload_file(rlft_paths["train"], f"{cfg.bucket}/cont_rlft/train.jsonl")
    rlft_val = upload_file(rlft_paths["val"], f"{cfg.bucket}/cont_rlft/val.jsonl")
    rlft_name = f"geap-cont-rlft-{VERSION}"

    rlft_job = find_tuning_job_by_display_name(client, rlft_name)
    if rlft_job is None:
        rlft_job = launch_rlft_job(
            client,
            train_uri=rlft_train,
            val_uri=rlft_val,
            display_name=rlft_name,
            base_model=sft_model,  # continue-tune from the SFT model
        )
        print(f"Launched RLFT continuation job: {rlft_job.name}")
    else:
        print(f"Reusing RLFT continuation job: {rlft_job.name} ({rlft_job.state})")
    rlft_job = wait_for_tuning_job(client, rlft_job.name)

    # === Compare: SFT seed vs. continuously-tuned RLFT model ===
    sft_metrics = _accuracy(client, tuned_endpoint(sft_job))
    rlft_metrics = _accuracy(client, tuned_endpoint(rlft_job))
    print(f"SFT seed accuracy:        {sft_metrics['accuracy']:.3f} (n={sft_metrics['n']})")
    print(f"Continuous RLFT accuracy: {rlft_metrics['accuracy']:.3f} (n={rlft_metrics['n']})")


if __name__ == "__main__":
    main()
