"""Tests for Cloud Storage helpers."""

from pathlib import Path
from unittest.mock import patch

from geap_tuning.gcs import build_gcs_uri, upload_file


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
