"""Cloud Storage helpers for staging tuning datasets.

All GEAP tuning jobs read their datasets from ``gs://`` URIs, so every service
needs to build URIs and upload JSONL to a bucket. These helpers keep that in one
place; :func:`build_gcs_uri` is pure and the upload functions wrap the Cloud
Storage client.
"""

from __future__ import annotations

import base64
import functools
from typing import TYPE_CHECKING

from google.api_core.exceptions import NotFound
from google.cloud import storage

if TYPE_CHECKING:
    from pathlib import Path

_GCS_PREFIX = "gs://"
# Enough hex to make a collision implausible while fitting a resource-label value.
_FINGERPRINT_CHARS = 16


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


@functools.cache
def _default_client() -> storage.Client:
    """Return a process-wide Cloud Storage client, built on first use.

    Cached because a fresh ``storage.Client()`` per call meant one authenticated
    client and connection pool **per uploaded file** — the vision example stages
    ~350 images, so that was ~350 clients left to garbage collection. Callers that
    need their own credentials or project can pass ``client=`` instead.
    """
    return storage.Client()


def upload_file(
    local_path: str | Path,
    gcs_uri: str,
    *,
    client: storage.Client | None = None,
) -> str:
    """Upload a local file to ``gcs_uri`` and return that URI."""
    bucket_name, blob_name = _split_uri(gcs_uri)
    storage_client = client or _default_client()
    storage_client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
    return gcs_uri


def upload_jsonl(
    local_path: str | Path,
    bucket: str,
    *parts: str,
    client: storage.Client | None = None,
) -> str:
    """Upload a JSONL file under ``bucket``/``parts`` and return the resulting URI."""
    gcs_uri = build_gcs_uri(bucket, *parts)
    return upload_file(local_path, gcs_uri, client=client)


def object_fingerprint(gcs_uri: str, *, client: storage.Client | None = None) -> str | None:
    """Return a short, label-safe fingerprint of the object's **content**.

    Reads the checksum Cloud Storage already stores (``md5_hash``, falling back to
    ``crc32c`` for composite/resumable uploads that have no MD5) via a metadata-only
    ``reload()`` — the object is never downloaded.

    This exists because a tuning job records the dataset **URI**, not its content,
    and every driver stages to a fixed path. Without a content fingerprint,
    overwriting the data and re-running silently reuses the model trained on the
    old data. Returns ``None`` when the object does not exist.

    The result is lowercase hex truncated to 16 chars, which satisfies the GCP
    resource-label value grammar (``[a-z0-9_-]{0,63}``).
    """
    bucket_name, blob_name = _split_uri(gcs_uri)
    storage_client = client or _default_client()
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    try:
        blob.reload()
    except NotFound:
        return None
    digest = blob.md5_hash or blob.crc32c
    if not digest:
        return None
    return base64.b64decode(digest).hex()[:_FINGERPRINT_CHARS]
