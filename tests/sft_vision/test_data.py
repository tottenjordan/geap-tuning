"""Tests for the multimodal (image) SFT dataset builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from geap_tuning.sft_vision.data import (
    CLASSES,
    DATA_SEGMENT,
    LABEL_MAP,
    PROMPT,
    SelectedImage,
    _classify,
    _configure_kaggle_auth,
    _import_kagglehub,
    _split_of,
    build_image_records,
    build_vision_dataset,
    download_dataset,
    image_gcs_uri,
    prepare_dataset,
)

# Filename extension per class in the synthetic tree — exercises MIME mapping.
_EXT_BY_CLASS = {
    "calculus": ".jpg",
    "cancer": ".png",
    "caries": ".jpeg",
    "gingivitis": ".webp",
    "ulcer": ".JPG",  # uppercase: classification/MIME must be case-insensitive
}
_SOURCE_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def _make_source_tree(root: Path, *, per_class: int = 6) -> Path:
    """Create a YOLO-style tree of empty image files under ``root``."""
    for split, split_dir in _SOURCE_SPLIT_DIRS.items():
        for class_name in CLASSES:
            folder = root / split_dir / "images"
            folder.mkdir(parents=True, exist_ok=True)
            ext = _EXT_BY_CLASS[class_name]
            for i in range(per_class):
                (folder / f"{class_name}_{split}_{i}{ext}").write_bytes(b"fake-image")
    return root


def test_classes_match_label_map() -> None:
    assert tuple(LABEL_MAP) == CLASSES
    assert set(CLASSES) == {"calculus", "cancer", "caries", "gingivitis", "ulcer"}


def test_prompt_lists_every_display_label() -> None:
    for label in LABEL_MAP.values():
        assert label in PROMPT


def test_classify_matches_prefix_case_insensitively() -> None:
    assert _classify("gingivitis_train_1.webp") == "gingivitis"
    assert _classify("ULCER_5.JPG") == "ulcer"
    assert _classify("random_file.jpg") is None


def test_split_of_infers_from_path() -> None:
    assert _split_of(Path("a/train/images/x.jpg")) == "train"
    assert _split_of(Path("a/valid/images/x.jpg")) == "val"
    assert _split_of(Path("a/val/images/x.jpg")) == "val"
    assert _split_of(Path("a/test/images/x.jpg")) == "test"
    assert _split_of(Path("a/b/x.jpg")) is None


def test_prepare_dataset_downsamples_and_copies(tmp_path: Path) -> None:
    source = _make_source_tree(tmp_path / "src", per_class=6)
    out = tmp_path / "out"
    per_class = {"train": 3, "val": 2, "test": 2}

    selected = prepare_dataset(source, out, per_class=per_class, seed=7)

    for split, limit in per_class.items():
        items = selected[split]
        assert len(items) == limit * len(CLASSES)
        for item in items:
            assert isinstance(item, SelectedImage)
            assert item.split == split
            assert item.class_name in CLASSES
            # File was copied into out/{split}/{class}/{filename}.
            assert item.local_path == out / split / item.class_name / item.filename
            assert item.local_path.exists()
            assert item.local_path.read_bytes() == b"fake-image"


def test_prepare_dataset_infers_mime_type(tmp_path: Path) -> None:
    source = _make_source_tree(tmp_path / "src", per_class=2)
    selected = prepare_dataset(source, tmp_path / "out", per_class={"train": 2}, seed=1)
    mime_by_class = {item.class_name: item.mime_type for item in selected["train"]}
    assert mime_by_class["cancer"] == "image/png"
    assert mime_by_class["ulcer"] == "image/jpeg"  # uppercase .JPG
    assert mime_by_class["gingivitis"] == "image/webp"


def test_prepare_dataset_is_deterministic(tmp_path: Path) -> None:
    source = _make_source_tree(tmp_path / "src", per_class=6)

    def names(out: Path) -> dict[str, set[str]]:
        selected = prepare_dataset(source, out, per_class={"train": 3}, seed=42)
        return {"train": {item.filename for item in selected["train"]}}

    assert names(tmp_path / "a") == names(tmp_path / "b")


def test_prepare_dataset_cleans_stale_output(tmp_path: Path) -> None:
    source = _make_source_tree(tmp_path / "src", per_class=3)
    out = tmp_path / "out"
    stale = out / "train" / "calculus" / "stale.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    prepare_dataset(source, out, per_class={"train": 1}, seed=1)

    assert not stale.exists()


def test_image_gcs_uri_layout() -> None:
    uri = image_gcs_uri("gs://bucket", "prefix", "train", "caries", "caries_1.jpg")
    assert uri == f"gs://bucket/prefix/{DATA_SEGMENT}/train/caries/caries_1.jpg"


def test_build_image_records_shape() -> None:
    item = SelectedImage(
        split="train",
        class_name="gingivitis",
        filename="gingivitis_1.webp",
        local_path=Path("/tmp/gingivitis_1.webp"),  # noqa: S108 - not touched, label only
        mime_type="image/webp",
    )
    [record] = build_image_records([item], bucket="gs://bucket", gcs_prefix="prefix")

    user_turn = record["contents"][0]
    assert user_turn["role"] == "user"
    file_part = user_turn["parts"][0]["fileData"]
    assert file_part["mimeType"] == "image/webp"
    assert file_part["fileUri"] == image_gcs_uri(
        "gs://bucket", "prefix", "train", "gingivitis", "gingivitis_1.webp"
    )
    assert user_turn["parts"][1]["text"] == PROMPT

    model_turn = record["contents"][1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["text"] == LABEL_MAP["gingivitis"]


def test_build_vision_dataset_writes_jsonl_per_split(tmp_path: Path) -> None:
    source = _make_source_tree(tmp_path / "src", per_class=4)
    out = tmp_path / "out"

    result = build_vision_dataset(
        source,
        out,
        bucket="gs://bucket",
        gcs_prefix="prefix",
        per_class={"train": 2, "val": 1, "test": 1},
        seed=3,
    )

    assert set(result) == {"train", "val", "test"}
    for split, payload in result.items():
        jsonl_path = Path(payload["jsonl"])
        assert jsonl_path == out / f"{split}.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(payload["records"]) == len(payload["items"])


def test_configure_kaggle_auth_json_token() -> None:
    env: dict[str, str] = {"KAGGLE_API_TOKEN": '{"username": "alice", "key": "secret"}'}
    _configure_kaggle_auth(env)
    assert env["KAGGLE_USERNAME"] == "alice"
    assert env["KAGGLE_KEY"] == "secret"


def test_configure_kaggle_auth_plain_token() -> None:
    env: dict[str, str] = {"KAGGLE_API_TOKEN": "rawkey"}
    _configure_kaggle_auth(env)
    assert env["KAGGLE_KEY"] == "rawkey"
    assert "KAGGLE_USERNAME" not in env


def test_configure_kaggle_auth_noop_when_unset() -> None:
    env: dict[str, str] = {}
    _configure_kaggle_auth(env)
    assert env == {}


def test_import_kagglehub_missing_raises_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the missing-optional-dep branch regardless of whether the ``vision``
    # group happens to be installed, so the error guidance stays covered.
    def _raise(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("geap_tuning.sft_vision.data.importlib.import_module", _raise)
    with pytest.raises(RuntimeError, match="uv sync --group vision"):
        _import_kagglehub()


def test_download_dataset_delegates_to_kagglehub(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeKaggle:
        def dataset_download(self, slug: str) -> str:
            assert "oral-disease" in slug
            return "/tmp/kaggle-cache/oral"  # noqa: S108 - fake path, not touched

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(
        "geap_tuning.sft_vision.data._import_kagglehub",
        _FakeKaggle,
    )
    assert download_dataset() == Path("/tmp/kaggle-cache/oral")  # noqa: S108 - fake path
