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
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types

_PROJECT_KEYS = ("PROJECT_ID", "GOOGLE_CLOUD_PROJECT")
_LOCATION_KEYS = ("GOOGLE_CLOUD_LOCATION", "GCP_REGION")
_BUCKET_KEYS = ("GCS_BUCKET_NAME", "BUCKET", "GOOGLE_CLOUD_STORAGE_BUCKET")
_DEFAULT_LOCATION = "us-central1"

# The Gen AI SDK does NOT retry by default: ``HttpOptions.retry_options`` is
# ``None``, which the SDK resolves to ``stop_after_attempt(1)`` — i.e. never.
# That is a poor fit here: ``wait_for_tuning_job`` polls for 30-60 minutes and the
# eval loops make hundreds of ``generate_content`` calls, so a single transient
# 429/503 would discard an already-paid run. Setting this once on the client
# covers every SDK call the repo makes — polling, tuning and inference alike.
_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5,
    initial_delay=1.0,
    max_delay=60.0,
    exp_base=2.0,
    jitter=1.0,
    http_status_codes=[429, 500, 502, 503, 504],
)

# Without this httpx receives ``timeout=None``, which means *no* timeout — a
# half-open connection would hang a poll forever with no output.
_REQUEST_TIMEOUT_MS = 120_000


def _http_options() -> types.HttpOptions:
    """Return the shared transport policy (retry + timeout) for every client."""
    return types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS, retry_options=_RETRY_OPTIONS)


# A single resource label, sourced from one key/value env pair. Kept as a
# {key: value} map (not a scalar pair) so it drops straight into the SDK
# ``labels=`` kwargs; empty when either var is unset.
_LABEL_KEY_ENV = "LABEL_KEY"
_LABEL_VALUE_ENV = "LABEL_VALUE"

# Gemini 3.x models serve *inference* only from the ``global`` endpoint. NOTE:
# tuning is NOT available on ``global`` (see ``requires_global_endpoint``), so this
# applies to inference clients only — tuning jobs stay regional.
GLOBAL_LOCATION = "global"
_GEMINI_MAJOR_RE = re.compile(r"gemini-(\d+)")
_GEMINI_GLOBAL_MAJOR = 3

# A tuned model's endpoint/model resource name embeds its own location, e.g.
# ``projects/P/locations/us/endpoints/E``. For Gemini 3.x that location is the
# ``us``/``eu`` multi-region — NOT the regional location the tuning job ran in.
_ENDPOINT_LOCATION_RE = re.compile(r"/locations/([^/]+)/")


@dataclass(frozen=True)
class TuningConfig:
    """Resolved GEAP settings shared by every tuning service."""

    project: str
    location: str
    bucket: str  # normalized to a gs:// URI
    labels: dict[str, str] = field(default_factory=dict)  # env-driven; empty when unset


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
    if location == GLOBAL_LOCATION:
        # `global` serves Gemini 3.x *inference* but does not support tuning, so a
        # config resolving to it would fail later at job-launch time with a much
        # less obvious error. `resolve_location` still routes inference clients to
        # `global` when the model needs it; that is a per-call decision, not config.
        msg = (
            f"location {GLOBAL_LOCATION!r} does not support tuning; set "
            f"{_LOCATION_KEYS[0]} to a region such as {_DEFAULT_LOCATION!r}"
        )
        raise ValueError(msg)

    # One resource label, attached to every resource we create (tuning jobs and
    # their generated model/endpoint, plus Managed TensorBoard). Both vars must
    # be set for the label to apply; otherwise ``labels`` stays empty and label
    # attachment is a no-op.
    label_key = env.get(_LABEL_KEY_ENV)
    label_value = env.get(_LABEL_VALUE_ENV)
    labels = {label_key: label_value} if label_key and label_value else {}

    return TuningConfig(project=project, location=location, bucket=bucket, labels=labels)


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

    Thin factory over ``genai.Client(vertexai=True, ...)``, which also attaches
    the shared retry/timeout policy (see :func:`_http_options`). Pass ``base_model`` to
    route an **inference** client for a Gemini 3.x model to the ``global`` endpoint
    (see :func:`resolve_location`) — the client's location is fixed at
    construction. Leave ``base_model`` unset for **tuning** clients: tuning is not
    available on ``global``, so those must stay regional.
    """
    cfg = cfg or load_config()
    location = resolve_location(cfg, base_model)
    return genai.Client(
        vertexai=True,
        project=cfg.project,
        location=location,
        http_options=_http_options(),
    )


def endpoint_location(endpoint: str) -> str | None:
    """Return the location embedded in a tuned endpoint/model resource name.

    A tuned model is deployed to an endpoint whose resource name carries its own
    location (``projects/P/locations/<loc>/endpoints/E``). For Gemini 3.x that is
    the ``us``/``eu`` **multi-region**, not the regional location the tuning job
    ran in — so inference must target that location or it 404s. Returns ``None``
    for a bare endpoint id (no ``/locations/`` segment), letting callers fall back
    to a default.
    """
    match = _ENDPOINT_LOCATION_RE.search(endpoint)
    return match.group(1) if match else None


def genai_client_for_endpoint(cfg: TuningConfig, endpoint: str) -> genai.Client:
    """Build an **inference** client whose location matches a tuned ``endpoint``.

    Thin factory: reads the location out of the endpoint resource name (see
    :func:`endpoint_location`) so ``generate_content`` resolves it, falling back to
    ``cfg.location`` when the name has no location segment. Use this to call any
    tuned model — a regional (or ``global``) client cannot reach a tuned Gemini 3.x
    endpoint, which lives in the ``us``/``eu`` multi-region.
    """
    location = endpoint_location(endpoint) or cfg.location
    return genai.Client(
        vertexai=True,
        project=cfg.project,
        location=location,
        http_options=_http_options(),
    )
