"""Tests for Cloud Storage helpers."""

import base64
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import NotFound

from geap_tuning import gcs
from geap_tuning.gcs import build_gcs_uri, upload_file, upload_jsonl


def test_build_gcs_uri_joins_and_strips() -> None:
    assert build_gcs_uri("gs://b", "sft", "train.jsonl") == "gs://b/sft/train.jsonl"
    assert build_gcs_uri("gs://b/", "/sft/", "x") == "gs://b/sft/x"


@patch("geap_tuning.gcs.storage.Client")
def test_upload_file_parses_bucket_and_blob(mock_client: object, tmp_path: Path) -> None:
    local = tmp_path / "t.jsonl"
    local.write_text("{}", encoding="utf-8")

    uri = upload_file(local, "gs://mybucket/data/t.jsonl")

    assert uri == "gs://mybucket/data/t.jsonl"
    client = mock_client.return_value  # type: ignore[attr-defined]
    client.bucket.assert_called_once_with("mybucket")
    client.bucket.return_value.blob.assert_called_once_with("data/t.jsonl")
    client.bucket.return_value.blob.return_value.upload_from_filename.assert_called_once_with(
        str(local)
    )


def test_upload_file_accepts_an_injected_client(tmp_path: Path) -> None:
    src = tmp_path / "d.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    client = MagicMock()

    uri = upload_file(src, "gs://b/p/d.jsonl", client=client)

    assert uri == "gs://b/p/d.jsonl"
    client.bucket.assert_called_once_with("b")
    client.bucket.return_value.blob.assert_called_once_with("p/d.jsonl")


def test_upload_file_reuses_one_default_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One authenticated client per uploaded file meant ~350 of them in the vision
    # example; the default client must be built once and shared.
    src = tmp_path / "d.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    built = []

    def fake_client() -> MagicMock:
        built.append(1)
        return MagicMock()

    gcs._default_client.cache_clear()  # noqa: SLF001 - clearing the cache under test
    monkeypatch.setattr(gcs.storage, "Client", fake_client)
    try:
        for i in range(3):
            upload_file(src, f"gs://b/p/{i}.jsonl")
        assert len(built) == 1
    finally:
        gcs._default_client.cache_clear()  # noqa: SLF001 - leave no cached mock


def test_upload_jsonl_threads_the_client(tmp_path: Path) -> None:
    src = tmp_path / "d.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    client = MagicMock()
    assert upload_jsonl(src, "gs://b", "p", "d.jsonl", client=client) == "gs://b/p/d.jsonl"
    client.bucket.assert_called_once_with("b")


# --- content fingerprint (W-6) ---------------------------------------------------


def test_object_fingerprint_reads_md5_without_downloading() -> None:
    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.md5_hash = base64.b64encode(bytes.fromhex("aabbccdd" * 4)).decode()

    fp = gcs.object_fingerprint("gs://b/train.jsonl", client=client)

    assert fp == "aabbccddaabbccdd"  # 16 hex chars
    blob.reload.assert_called_once()  # metadata only
    blob.download_as_bytes.assert_not_called()


def test_object_fingerprint_falls_back_to_crc32c() -> None:
    # Composite/resumable uploads can have no MD5.
    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.md5_hash = None
    blob.crc32c = base64.b64encode(bytes.fromhex("11223344")).decode()
    assert gcs.object_fingerprint("gs://b/t.jsonl", client=client) == "11223344"


def test_object_fingerprint_returns_none_for_a_missing_object() -> None:
    client = MagicMock()
    client.bucket.return_value.blob.return_value.reload.side_effect = NotFound("nope")
    assert gcs.object_fingerprint("gs://b/absent.jsonl", client=client) is None


def test_object_fingerprint_is_label_safe() -> None:
    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.md5_hash = base64.b64encode(bytes.fromhex("ff" * 16)).decode()
    fp = gcs.object_fingerprint("gs://b/t.jsonl", client=client)
    assert re.fullmatch(r"[a-z0-9_-]{1,63}", fp)  # GCP resource-label value grammar


def test_object_fingerprint_differs_for_different_content() -> None:
    # The whole point: same URI, edited bytes -> different fingerprint.
    def fp_for(hexdigest: str) -> str | None:
        client = MagicMock()
        client.bucket.return_value.blob.return_value.md5_hash = base64.b64encode(
            bytes.fromhex(hexdigest)
        ).decode()
        return gcs.object_fingerprint("gs://b/train.jsonl", client=client)

    assert fp_for("aa" * 16) != fp_for("bb" * 16)
