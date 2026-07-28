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
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai

_PROJECT_KEYS = ("PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
_LOCATION_KEYS = ("GOOGLE_CLOUD_LOCATION", "GCP_REGION")
_BUCKET_KEYS = ("GCS_BUCKET_NAME", "BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
_DEFAULT_LOCATION = "us-central1"

# Gemini 3.x models serve *inference* only from the ``global`` endpoint. NOTE:
# tuning is NOT available on ``global`` (see ``requires_global_endpoint``), so this
# applies to inference clients only — tuning jobs stay regional.
GLOBAL_LOCATION = "global"
_GEMINI_MAJOR_RE = re.compile(r"gemini-(\d+)")
_GEMINI_GLOBAL_MAJOR = 3


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


def requires_global_endpoint(model: str | None) -> bool:
    """Return ``True`` for Gemini 3.x+ models, which serve inference only from ``global``.

    Gemini 3.x models are served for **inference** (``generateContent``) only from
    the ``global`` endpoint, not regional ones. Older models (Gemini 1.x/2.x) and
    an unset model stay regional.

    IMPORTANT: this concerns *inference against the base model*. It does **not**
    apply to tuning — the ``global`` endpoint explicitly does **not** support
    tuning (or ``validate_reward``), so tuning jobs must always run in a region
    (``us-central1``/``europe-west4``), regardless of the base model. See
    ``docs/notes/environment.md``.
    """
    if not model:
        return False
    match = _GEMINI_MAJOR_RE.search(model)
    return bool(match) and int(match.group(1)) >= _GEMINI_GLOBAL_MAJOR


def resolve_location(cfg: TuningConfig, base_model: str | None = None) -> str:
    """Return the inference location, forcing ``global`` for Gemini 3.x+ models.

    For **inference** clients only. Do not use this to place a tuning job: tuning
    is not offered on ``global`` (see :func:`requires_global_endpoint`), so tuning
    code should pass ``cfg.location`` directly.
    """
    if requires_global_endpoint(base_model):
        return GLOBAL_LOCATION
    return cfg.location


def genai_client(cfg: TuningConfig | None = None, *, base_model: str | None = None) -> genai.Client:
    """Build a Gen AI SDK client wired to the Vertex/GEAP backend.

    Thin factory over ``genai.Client(vertexai=True, ...)``; not unit-tested
    because it only forwards resolved config to the SDK. Pass ``base_model`` to
    route an **inference** client for a Gemini 3.x model to the ``global`` endpoint
    (see :func:`resolve_location`) — the client's location is fixed at
    construction. Leave ``base_model`` unset for **tuning** clients: tuning is not
    available on ``global``, so those must stay regional.
    """
    cfg = cfg or load_config()
    location = resolve_location(cfg, base_model)
    return genai.Client(vertexai=True, project=cfg.project, location=location)
