"""Resolve GEAP config from the environment.

The project ``.env`` carries redundant aliases for the same values (see
``docs/notes/environment.md``): the project id under ``PROJECT_ID`` and
``GOOGLE_CLOUD_PROJECT``, the location under ``GOOGLE_CLOUD_LOCATION`` and
``GCP_REGION``, and the bucket under three different names. ``load_config``
collapses that sprawl into one :class:`TuningConfig` so every tuning service
reads config the same way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai

_PROJECT_KEYS = ("PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
_LOCATION_KEYS = ("GOOGLE_CLOUD_LOCATION", "GCP_REGION")
_BUCKET_KEYS = ("GCS_BUCKET_NAME", "BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
_DEFAULT_LOCATION = "us-central1"


@dataclass(frozen=True)
class TuningConfig:
    """Resolved GEAP settings shared by every tuning service."""

    project: str
    location: str
    bucket: str  # normalized to a gs:// URI


def _first(env: dict[str, str], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among ``keys`` in ``env``."""
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def load_config(env: dict[str, str] | None = None) -> TuningConfig:
    """Resolve a :class:`TuningConfig` from ``env`` (defaults to ``.env`` + ``os.environ``)."""
    if env is None:
        load_dotenv()
        env = dict(os.environ)

    project = _first(env, _PROJECT_KEYS)
    if not project:
        msg = f"No project set; expected one of {_PROJECT_KEYS}"
        raise ValueError(msg)

    bucket = _first(env, _BUCKET_KEYS)
    if not bucket:
        msg = f"No bucket set; expected one of {_BUCKET_KEYS}"
        raise ValueError(msg)
    if not bucket.startswith("gs://"):
        bucket = f"gs://{bucket}"

    location = _first(env, _LOCATION_KEYS) or _DEFAULT_LOCATION
    return TuningConfig(project=project, location=location, bucket=bucket)


def genai_client(cfg: TuningConfig | None = None) -> genai.Client:
    """Build a Gen AI SDK client wired to the Vertex/GEAP backend.

    Thin factory over ``genai.Client(vertexai=True, ...)``; not unit-tested
    because it only forwards resolved config to the SDK.
    """
    cfg = cfg or load_config()
    return genai.Client(vertexai=True, project=cfg.project, location=cfg.location)
