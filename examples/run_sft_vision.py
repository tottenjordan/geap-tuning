"""Multimodal (image) SFT: oral-disease classification with a hyperparameter sweep.

REQUIRES LIVE GCP AND INCURS TUNING COST. This is an integration entrypoint, not
covered by the test suite (pytest only exercises the mocked units in
``geap_tuning.sft_vision``). Run it with a real ``.env`` and ``gcloud auth`` in
place, plus a Kaggle token (``KAGGLE_API_TOKEN``) for the dataset download and the
optional ``vision`` dependency group installed::

    uv sync --group vision
    uv run python examples/run_sft_vision.py

Or point it at a pre-downloaded copy of the dataset to skip the Kaggle call::

    uv run python examples/run_sft_vision.py --data-dir /path/to/oral-disease

Dataset: Kaggle *Multi-Class Oral Disease Detection Dataset*
(``singh868/multi-class-oral-disease-detection-dataset``), by Rahul Singh,
licensed **CC BY-SA 4.0**.

Flow: download → balanced per-class downsample → stage images + JSONL to GCS →
launch two SFT experiments (reusing any job with the same display name) → wait →
evaluate each on the **validation** split → pick the best config → evaluate the
winner on the **test** split → print a comparison table.

The tune call is the *same* ``launch_sft_job`` the text SFT example uses — the
only difference is multimodal ``contents`` records carrying a ``fileData`` image
part. See ``docs/notes/multimodal-sft.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.genai import types

from geap_tuning.config import genai_client, load_config
from geap_tuning.gcs import upload_file
from geap_tuning.inference import generate
from geap_tuning.jobs import (
    find_tuning_job_by_display_name,
    tuned_endpoint,
    wait_for_tuning_job,
)
from geap_tuning.sft.tune import launch_sft_job
from geap_tuning.sft_vision.data import (
    PROMPT,
    build_vision_dataset,
    download_dataset,
    image_gcs_uri,
)
from geap_tuning.sft_vision.evaluate import (
    image_gcs_uri_of,
    resolve_local_path,
    run_image_eval,
    select_best_experiment,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from geap_tuning.schemas import Record

# Local staging root (also the ``local_root`` eval maps GCS URIs back to) and the
# GCS prefix under the configured bucket. Images land at
# ``{bucket}/{GCS_PREFIX}/data/{split}/{class}/{file}``; JSONL at
# ``{bucket}/{GCS_PREFIX}/{split}.jsonl``.
OUT_DIR = Path("datasets/sft_vision_oral")
GCS_PREFIX = "sft_vision_oral"
BASE_MODEL = "gemini-2.5-flash-lite"

# Balanced per-class sample. Small by default to bound tuning cost (the reference
# uses 200/40/40 across 2 experiments); raise for a more faithful run.
PER_CLASS = {"train": 50, "val": 10, "test": 10}

# Two configs to compare. Names must be filesystem/display-name safe.
EXPERIMENTS: list[dict[str, Any]] = [
    {"name": "baseline", "epochs": 2, "learning_rate_multiplier": 1.0, "adapter_size": 8},
    {"name": "wide", "epochs": 3, "learning_rate_multiplier": 2.0, "adapter_size": 16},
]


def _make_predict(client: Any, endpoint: str) -> Callable[[Record], str]:  # noqa: ANN401
    """Return a ``predict_fn`` that sends each record's local image to ``endpoint``."""

    def predict(record: Record) -> str:
        uri = image_gcs_uri_of(record)
        mime_type = record["contents"][0]["parts"][0]["fileData"]["mimeType"]
        local_path = resolve_local_path(uri, OUT_DIR)
        part = types.Part.from_bytes(data=local_path.read_bytes(), mime_type=mime_type)
        return generate(client, endpoint, [part, PROMPT])

    return predict


def _stage_to_gcs(bucket: str, dataset: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Upload every selected image and each split's JSONL; return split→JSONL URI."""
    jsonl_uris: dict[str, str] = {}
    for split, payload in dataset.items():
        for item in payload["items"]:
            dest = image_gcs_uri(bucket, GCS_PREFIX, item.split, item.class_name, item.filename)
            upload_file(item.local_path, dest)
        jsonl_uris[split] = upload_file(payload["jsonl"], f"{bucket}/{GCS_PREFIX}/{split}.jsonl")
        print(f"Staged {len(payload['items'])} images + JSONL for split={split}")
    return jsonl_uris


def _run_experiment(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    experiment: dict[str, Any],
    *,
    train_uri: str,
    val_uri: str,
) -> Any:  # noqa: ANN401 - returns the SDK tuning job
    """Reuse-or-launch one SFT job for ``experiment`` and wait for completion."""
    display_name = f"geap-sft-vision-{experiment['name']}"
    job = find_tuning_job_by_display_name(client, display_name)
    if job is None:
        job = launch_sft_job(
            client,
            train_uri=train_uri,
            val_uri=val_uri,
            display_name=display_name,
            base_model=BASE_MODEL,
            epochs=experiment["epochs"],
            adapter_size=experiment["adapter_size"],
            learning_rate_multiplier=experiment["learning_rate_multiplier"],
        )
        print(f"[{experiment['name']}] launched {job.name}")
    else:
        print(f"[{experiment['name']}] reusing {job.name} ({job.state})")
    return wait_for_tuning_job(client, job.name)


def _print_table(
    experiments: list[dict[str, Any]],
    val_results: dict[str, dict[str, Any]],
    best_name: str,
    test_metrics: dict[str, Any],
) -> None:
    """Print a per-experiment comparison table (val metrics; winner's test metrics)."""
    print("\n=== Experiment comparison (validation split) ===")
    header = f"{'experiment':<12}{'epochs':>7}{'lr':>6}{'adapter':>9}{'val_acc':>9}{'val_f1':>9}"
    print(header)
    print("-" * len(header))
    by_name = {e["name"]: e for e in experiments}
    for name in sorted(val_results):
        exp = by_name[name]
        metrics = val_results[name]
        marker = "  <- best" if name == best_name else ""
        print(
            f"{name:<12}{exp['epochs']:>7}{exp['learning_rate_multiplier']:>6}"
            f"{exp['adapter_size']:>9}{metrics['accuracy']:>9.3f}{metrics['macro_f1']:>9.3f}"
            f"{marker}",
        )
    print(
        f"\nBest config '{best_name}' on TEST split: "
        f"accuracy={test_metrics['accuracy']:.3f} macro_f1={test_metrics['macro_f1']:.3f}",
    )


def main() -> None:
    """Run the multimodal SFT sweep against live GEAP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Pre-downloaded dataset root (skips the Kaggle download).",
    )
    args = parser.parse_args()

    cfg = load_config()
    client = genai_client(cfg)
    print(f"Project={cfg.project} location={cfg.location} bucket={cfg.bucket}")

    # 1. Acquire the raw dataset (Kaggle download unless a local root is given).
    source_dir = args.data_dir or download_dataset()
    print(f"Dataset source: {source_dir}")

    # 2. Downsample + build multimodal contents JSONL per split under OUT_DIR.
    dataset = build_vision_dataset(
        source_dir,
        OUT_DIR,
        bucket=cfg.bucket,
        gcs_prefix=GCS_PREFIX,
        per_class=PER_CLASS,
    )

    # 3. Stage images + JSONL to GCS.
    jsonl_uris = _stage_to_gcs(cfg.bucket, dataset)

    # 4. Launch/await each experiment, then evaluate on the validation split.
    val_results: dict[str, dict[str, Any]] = {}
    endpoints: dict[str, str] = {}
    val_records = dataset["val"]["records"]
    for experiment in EXPERIMENTS:
        job = _run_experiment(
            client,
            experiment,
            train_uri=jsonl_uris["train"],
            val_uri=jsonl_uris["val"],
        )
        endpoint = tuned_endpoint(job)
        endpoints[experiment["name"]] = endpoint
        metrics = run_image_eval(val_records, _make_predict(client, endpoint))
        val_results[experiment["name"]] = metrics
        print(f"[{experiment['name']}] val accuracy={metrics['accuracy']:.3f}")

    # 5. Select the best config on validation, then score it on the test split.
    best_name = select_best_experiment(val_results)
    test_metrics = run_image_eval(
        dataset["test"]["records"],
        _make_predict(client, endpoints[best_name]),
    )
    _print_table(EXPERIMENTS, val_results, best_name, test_metrics)


if __name__ == "__main__":
    main()
