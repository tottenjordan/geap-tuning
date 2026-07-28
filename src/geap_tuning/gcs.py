"""Cloud Storage helpers for staging tuning datasets.

All GEAP tuning jobs read their datasets from ``gs://`` URIs, so every service
needs to build URIs and upload JSONL to a bucket. These helpers keep that in one
place; :func:`build_gcs_uri` is pure and the upload functions wrap the Cloud
Storage client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.cloud import storage

if TYPE_CHECKING:
    from pathlib import Path

_GCS_PREFIX = "gs://"


def build_gcs_uri(bucket: str, *parts: str) -> str:
    """Join a bucket and path segments into a normalized ``gs://`` URI."""
    base = bucket.removeprefix(_GCS_PREFIX).strip("/")
    cleaned = [segment.strip("/") for segment in parts if segment.strip("/")]
    return _GCS_PREFIX + "/".join([base, *cleaned])


def _split_uri(gcs_uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/path/to/blob`` into ``(bucket, blob)``."""
    if not gcs_uri.startswith(_GCS_PREFIX):
        msg = f"Not a gs:// URI: {gcs_uri}"
        raise ValueError(msg)
    bucket, _, blob = gcs_uri.removeprefix(_GCS_PREFIX).partition("/")
    if not blob:
        msg = f"URI has no object path: {gcs_uri}"
        raise ValueError(msg)
    return bucket, blob


def upload_file(local_path: str | Path, gcs_uri: str) -> str:
    """Upload a local file to ``gcs_uri`` and return that URI."""
    bucket_name, blob_name = _split_uri(gcs_uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
    return gcs_uri


def upload_jsonl(local_path: str | Path, bucket: str, *parts: str) -> str:
    """Upload a JSONL file under ``bucket``/``parts`` and return the resulting URI."""
    gcs_uri = build_gcs_uri(bucket, *parts)
    return upload_file(local_path, gcs_uri)
