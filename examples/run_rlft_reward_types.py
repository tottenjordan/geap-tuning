"""RLFT reward-type tour: preflight four reward shapes, then tune on a composite.

REQUIRES LIVE GCP AND INCURS TUNING COST. Like ``run_rlft.py``, this is an
integration entrypoint, not covered by the test suite. Run with a real ``.env``
and ``gcloud auth``:

    uv run python examples/run_rlft_reward_types.py

It reuses the verifiable-math dataset from ``run_rlft.py`` and walks the four RLFT
reward-scorer shapes the SDK exposes, so you can see when to reach for each:

- **code-execution** (``build_reward_config``) — ships tested Python that verifies
  correctness against the record's ``references`` (the strongest signal here).
- **string-match** (``build_string_match_reward_config``) — a cheap, declarative
  format/keyword reward; no gold answer, no sandbox. Here: does the response end
  with an ``Answer: <number>`` line?
- **autorater** (``build_autorater_reward_config``) — an LLM judge scores
  subjective quality (clear step-by-step reasoning).
- **cloud-run** (``build_cloud_run_reward_config``) — delegates to an external
  service; **documented only** (needs a deployed endpoint), so it is built and
  printed but not preflighted or tuned on.

The **composite** reward combines verifiable correctness (code-execution, 0.8)
with subjective quality (autorater, 0.2) — a common RLFT recipe — and is the one
we actually launch a job with.

NOTE: the RLFT docs call out ``gemini-3.5-flash`` as the supported base model.
Tuning stays **regional** (``cfg.location``); the ``global`` endpoint excludes
tuning and ``validate_reward``.
"""

from __future__ import annotations

import argparse
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
from geap_tuning.rlft.tune import (
    build_autorater_reward_config,
    build_cloud_run_reward_config,
    build_composite_reward_config,
    build_reward_config,
    build_string_match_reward_config,
    launch_rlft_job,
    validate_reward_config,
)

VERSION = "v1"
DISPLAY_NAME = f"geap-rlft-rewards-{VERSION}"
DATA_DIR = Path("datasets/rlft_math")
GCS_PREFIX = "rlft_rewards"
BASE_MODEL = "gemini-3.5-flash"
# A placeholder Cloud Run URL — the cloud-run scorer is documented-only here.
CLOUD_RUN_URI = "https://reward-scorer-xxxxxxxx-uc.a.run.app"


def main(*, preflight_only: bool = False) -> None:
    """Preflight each reward shape, then launch one composite-reward RLFT job.

    With ``preflight_only=True`` the function stops after the (free)
    ``validate_reward`` calls — no GCS upload, no tuning job, no cost — so you can
    smoke-test the new reward shapes against the live API before spending.
    """
    cfg = load_config()
    client = genai_client(cfg)  # tuning is regional-only; global excludes tuning
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")

    # 1. Build the local dataset splits and one preflight record.
    paths = build_rlft_dataset(DATA_DIR)
    train_records = build_rlft_records(split_dataset(MATH_PROBLEMS)[0])
    record = train_records[0]
    print(f"Wrote splits: {paths}")

    # 2. Build each single reward shape (cloud-run is documented-only).
    code_reward = build_reward_config(reward_name="math_correctness")
    string_reward = build_string_match_reward_config()
    autorater_reward = build_autorater_reward_config()
    cloud_run_reward = build_cloud_run_reward_config(
        reward_name="external_scorer", cloud_run_uri=CLOUD_RUN_URI
    )
    print(
        "Built rewards: code-execution, string-match, autorater, and a "
        f"cloud-run builder for {cloud_run_reward.cloud_run_reward_scorer.cloud_run_uri} "
        "(not preflighted — needs a deployed service)."
    )

    # 3. Composite = verifiable correctness (0.8) + subjective quality (0.2).
    composite = build_composite_reward_config([(code_reward, 0.8), (autorater_reward, 0.2)])

    # 4. Preflight each preflightable reward on one example — no tuning cost.
    #    A sample answer that satisfies both the code-exec and format rewards.
    sample_answer = "Let me add them: 2 + 2 = 4.\nAnswer: 4"
    for name, single, composite_cfg in (
        ("code-execution", code_reward, None),
        ("string-match", string_reward, None),
        ("autorater", autorater_reward, None),
        ("composite", None, composite),
    ):
        result = validate_reward_config(
            client,
            project=cfg.project,
            location=cfg.location,
            sample_answer=sample_answer,
            example_record=record,
            reward_config=single,
            composite_reward_config=composite_cfg,
        )
        print(f"Preflight [{name}]: {result}")

    if preflight_only:
        print("Preflight-only: skipping GCS upload and tuning job (no cost incurred).")
        return

    # 5. Stage train/val to GCS.
    train_uri = upload_file(paths["train"], f"{cfg.bucket}/{GCS_PREFIX}/train.jsonl")
    val_uri = upload_file(paths["val"], f"{cfg.bucket}/{GCS_PREFIX}/val.jsonl")
    print(f"Uploaded train={train_uri} val={val_uri}")

    # 6. Reuse an existing job if one exists; otherwise launch on the composite.
    job = find_tuning_job_by_display_name(client, DISPLAY_NAME)
    if job is None:
        job = launch_rlft_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=DISPLAY_NAME,
            base_model=BASE_MODEL,
            composite_reward_config=composite,
        )
        print(f"Launched composite-reward tuning job: {job.name}")
    else:
        print(f"Reusing existing job: {job.name} ({job.state})")

    # 7. Wait, resolve the tuned endpoint, and report held-out accuracy.
    job = wait_for_tuning_job(client, job.name)
    endpoint = tuned_endpoint(job)
    print(f"Tuned endpoint: {endpoint}")

    _, _, test_problems = split_dataset(MATH_PROBLEMS)
    test_records = build_rlft_records(test_problems)
    metrics = run_rlft_eval(
        test_records,
        generate_fn=lambda user_text: generate(client, endpoint, user_text),
    )
    print(f"Held-out answer accuracy: {metrics['accuracy']:.3f} (n={metrics['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only run the free validate_reward preflights; skip GCS upload and the tuning job.",
    )
    args = parser.parse_args()
    main(preflight_only=args.preflight_only)
