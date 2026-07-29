"""Multimodal (image) SFT dataset: oral disease classification.

Ports the data pipeline of https://github.com/jswortz/dental-fine-tune-26 into the
repo's tested-package shape. The source is the Kaggle *Multi-Class Oral Disease
Detection Dataset* (``singh868/multi-class-oral-disease-detection-dataset``,
CC BY-SA 4.0, by Rahul Singh): YOLO-structured clinical images across five
classes. We infer the class from each filename prefix and the split from its
path, take a balanced per-class sample, and emit ``contents``-format JSONL where
each example pairs an image (a ``fileData`` part pointing at a GCS object) with a
text prompt and the ground-truth label.

The tuning call itself is identical to text SFT — multimodal records only differ
in carrying a ``fileData`` part — so the example reuses
:func:`geap_tuning.sft.tune.launch_sft_job`. See ``docs/notes/multimodal-sft.md``.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geap_tuning.gcs import build_gcs_uri
from geap_tuning.schemas import Record, file_part, model_turn, text_part, user_turn, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping

# Kaggle dataset slug (CC BY-SA 4.0). See module docstring for citation.
KAGGLE_DATASET = "singh868/multi-class-oral-disease-detection-dataset"

# Raw class prefixes (from the dataset's ``data.yaml``) → the display labels the
# model is trained to emit. The prompt lists the display labels verbatim.
LABEL_MAP: dict[str, str] = {
    "calculus": "Dental Calculus",
    "cancer": "Oral Cancer",
    "caries": "Dental Caries",
    "gingivitis": "Gingivitis",
    "ulcer": "Oral Ulcer",
}
CLASSES: tuple[str, ...] = tuple(LABEL_MAP)

PROMPT = (
    "Classify the oral disease depicted in this image. "
    "Classes: Dental Calculus, Oral Cancer, Dental Caries, Gingivitis, Oral Ulcer."
)

# Image extension → MIME type for the ``fileData`` part.
MIME_BY_EXT: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
SUPPORTED_EXTS = frozenset(MIME_BY_EXT)

# Splits and the path substrings that identify them in the YOLO-structured source.
SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_PER_CLASS: dict[str, int] = {"train": 50, "val": 10, "test": 10}

# Placed between the GCS prefix and the split so eval can map a fileUri back to a
# local image by splitting on it (see ``evaluate.resolve_local_path``).
DATA_SEGMENT = "data"


@dataclass(frozen=True, slots=True)
class SelectedImage:
    """One sampled image: where it lives locally and how to label/serve it."""

    split: str
    class_name: str
    filename: str
    local_path: Path
    mime_type: str


def _configure_kaggle_auth(env: MutableMapping[str, str]) -> None:
    """Translate a ``KAGGLE_API_TOKEN`` into the vars kagglehub expects.

    kagglehub authenticates with ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` (or
    ``~/.kaggle/kaggle.json``). The token downloaded from Kaggle is a JSON blob
    ``{"username": ..., "key": ...}``; if ``KAGGLE_API_TOKEN`` holds that JSON we
    split it into both vars, otherwise we treat the whole value as the key
    (``KAGGLE_USERNAME`` must then already be set). No-op if unset.
    """
    token = env.get("KAGGLE_API_TOKEN")
    if not token:
        return
    try:
        parsed = json.loads(token)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and "username" in parsed and "key" in parsed:
        env["KAGGLE_USERNAME"] = str(parsed["username"])
        env["KAGGLE_KEY"] = str(parsed["key"])
    else:
        env.setdefault("KAGGLE_KEY", token)


def _import_kagglehub() -> Any:  # noqa: ANN401 - optional dep, dynamically typed
    """Import kagglehub, raising an actionable error when the extra is missing.

    Uses :func:`importlib.import_module` so a missing optional dep is a clean
    runtime error (kagglehub is only in the ``vision`` group). Tests monkeypatch
    this function to avoid the network entirely.
    """
    try:
        return importlib.import_module("kagglehub")
    except ImportError as exc:
        msg = "kagglehub is required to download the dataset; run: uv sync --group vision"
        raise RuntimeError(msg) from exc


def download_dataset() -> Path:
    """Download the Kaggle oral-disease dataset and return its local root.

    Live-only (network + Kaggle credentials): reads ``KAGGLE_API_TOKEN`` from the
    environment (see :func:`_configure_kaggle_auth`) and delegates to
    ``kagglehub.dataset_download``. Not exercised by the test suite.
    """
    _configure_kaggle_auth(os.environ)
    return Path(_import_kagglehub().dataset_download(KAGGLE_DATASET))


def _classify(filename: str) -> str | None:
    """Return the class whose name prefixes ``filename`` (case-insensitive), else None."""
    lower = filename.lower()
    for class_name in CLASSES:
        if lower.startswith(class_name):
            return class_name
    return None


def _split_of(path: Path) -> str | None:
    """Infer the split from a path: ``train`` / ``valid|val`` / ``test``, else None."""
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if parts & {"valid", "val"}:
        return "val"
    if "test" in parts:
        return "test"
    return None


def _discover(source_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Group image paths under ``source_dir`` by ``{split: {class: [paths]}}``."""
    buckets: dict[str, dict[str, list[Path]]] = {s: {c: [] for c in CLASSES} for s in SPLITS}
    for dirpath, _, filenames in source_dir.walk():
        for name in filenames:
            if Path(name).suffix.lower() not in SUPPORTED_EXTS:
                continue
            class_name = _classify(name)
            split = _split_of(dirpath / name)
            if class_name is None or split is None:
                continue
            buckets[split][class_name].append(dirpath / name)
    return buckets


def prepare_dataset(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    per_class: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, list[SelectedImage]]:
    """Downsample the source into a balanced per-class subset staged under ``out_dir``.

    Recursively finds images, infers class (filename prefix) and split (path),
    deterministically shuffles and takes ``per_class[split]`` images per class,
    and copies each into ``out_dir/{split}/{class}/{filename}``. Returns the
    selected images per split. ``out_dir`` is cleaned first to avoid stale files.
    """
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    per_class = per_class or DEFAULT_PER_CLASS
    if out_dir.exists():
        shutil.rmtree(out_dir)

    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    buckets = _discover(source_dir)
    selected: dict[str, list[SelectedImage]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        limit = per_class.get(split, 0)
        for class_name in CLASSES:
            paths = list(buckets[split][class_name])
            rng.shuffle(paths)
            for src in paths[:limit]:
                dest = out_dir / split / class_name / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                mime = MIME_BY_EXT[src.suffix.lower()]
                selected[split].append(
                    SelectedImage(split, class_name, src.name, dest, mime),
                )
    return selected


def image_gcs_uri(bucket: str, gcs_prefix: str, split: str, class_name: str, filename: str) -> str:
    """Return the ``gs://`` URI an image is staged at (mirrors the local layout)."""
    return build_gcs_uri(bucket, gcs_prefix, DATA_SEGMENT, split, class_name, filename)


def build_image_records(
    items: Iterable[SelectedImage],
    *,
    bucket: str,
    gcs_prefix: str,
) -> list[Record]:
    """Build multimodal SFT ``contents`` records for ``items``.

    Each record's user turn carries the image (a ``fileData`` part pointing at the
    staged GCS object) plus the text :data:`PROMPT`; the model turn is the display
    label. Uses the shared ``contents`` builders in :mod:`geap_tuning.schemas`.
    """
    records: list[Record] = []
    for item in items:
        uri = image_gcs_uri(bucket, gcs_prefix, item.split, item.class_name, item.filename)
        records.append(
            {
                "contents": [
                    user_turn(file_part(item.mime_type, uri), text_part(PROMPT)),
                    model_turn(text_part(LABEL_MAP[item.class_name])),
                ],
            },
        )
    return records


def build_vision_dataset(  # noqa: PLR0913 - staging params, all keyword-only after the dirs
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    bucket: str,
    gcs_prefix: str,
    per_class: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Prepare, build records, and write one JSONL per split under ``out_dir``.

    Returns ``{split: {"jsonl": path, "records": [...], "items": [...]}}``. The
    ``items`` (each a :class:`SelectedImage`) drive image upload in the driver;
    the ``jsonl`` files are the tuning manifests.
    """
    out_dir = Path(out_dir)
    selected = prepare_dataset(source_dir, out_dir, per_class=per_class, seed=seed)
    result: dict[str, dict[str, Any]] = {}
    for split, items in selected.items():
        records = build_image_records(items, bucket=bucket, gcs_prefix=gcs_prefix)
        jsonl_path = out_dir / f"{split}.jsonl"
        write_jsonl(records, jsonl_path)
        result[split] = {"jsonl": str(jsonl_path), "records": records, "items": items}
    return result
