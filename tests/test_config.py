"""Tests for environment/config resolution."""

from unittest.mock import MagicMock

import pytest

from geap_tuning import config as config_module
from geap_tuning.config import (
    TuningConfig,
    endpoint_location,
    genai_client,
    genai_client_for_endpoint,
    load_config,
    requires_global_endpoint,
    resolve_location,
)

_RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]


@pytest.fixture
def captured_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``genai.Client`` so the factories can be inspected without a real client."""
    fake = MagicMock()
    monkeypatch.setattr(config_module.genai, "Client", fake)
    return fake


def test_resolves_aliases_and_defaults() -> None:
    env = {"GOOGLE_CLOUD_PROJECT": "proj-x", "BUCKET": "gs://b", "GCP_REGION": "europe-west4"}
    cfg = load_config(env)
    assert cfg.project == "proj-x"
    assert cfg.bucket == "gs://b"
    assert cfg.location == "europe-west4"


def test_project_id_takes_precedence_and_location_defaults() -> None:
    env = {"PROJECT_ID": "p1", "GOOGLE_CLOUD_PROJECT": "p2", "GCS_BUCKET_NAME": "gs://b"}
    cfg = load_config(env)
    assert cfg.project == "p1"  # PROJECT_ID wins
    assert cfg.location == "us-central1"  # default


def test_bucket_name_is_normalized_to_uri() -> None:
    cfg = load_config({"PROJECT_ID": "p", "GCS_BUCKET_NAME": "plain-bucket"})
    assert cfg.bucket == "gs://plain-bucket"


def test_missing_project_raises() -> None:
    with pytest.raises(ValueError, match="project"):
        load_config({"BUCKET": "gs://b"})


def test_missing_bucket_raises() -> None:
    with pytest.raises(ValueError, match="bucket"):
        load_config({"PROJECT_ID": "p"})


def test_labels_resolved_when_both_env_vars_set() -> None:
    env = {
        "PROJECT_ID": "p",
        "GCS_BUCKET_NAME": "gs://b",
        "LABEL_KEY": "project",
        "LABEL_VALUE": "geap-tuning",
    }
    assert load_config(env).labels == {"project": "geap-tuning"}


def test_labels_empty_when_env_vars_absent() -> None:
    cfg = load_config({"PROJECT_ID": "p", "GCS_BUCKET_NAME": "gs://b"})
    assert cfg.labels == {}


@pytest.mark.parametrize(
    "env",
    [
        {"PROJECT_ID": "p", "GCS_BUCKET_NAME": "gs://b", "LABEL_KEY": "project"},
        {"PROJECT_ID": "p", "GCS_BUCKET_NAME": "gs://b", "LABEL_VALUE": "geap-tuning"},
    ],
)
def test_labels_empty_when_only_one_env_var_set(env: dict[str, str]) -> None:
    assert load_config(env).labels == {}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3.5-flash", True),
        ("gemini-3-pro", True),
        ("gemini-2.5-flash", False),
        ("gemini-2.5-flash-lite", False),
        ("gemini-1.5-pro", False),
        (None, False),
        ("", False),
    ],
)
def test_requires_global_endpoint(model: str | None, expected: bool) -> None:  # noqa: FBT001
    assert requires_global_endpoint(model) is expected


def test_resolve_location_uses_global_for_gemini_3() -> None:
    cfg = TuningConfig(project="p", location="us-central1", bucket="gs://b")
    assert resolve_location(cfg, "gemini-3.5-flash") == "global"


def test_resolve_location_keeps_region_for_older_models() -> None:
    cfg = TuningConfig(project="p", location="us-central1", bucket="gs://b")
    assert resolve_location(cfg, "gemini-2.5-flash") == "us-central1"
    assert resolve_location(cfg, None) == "us-central1"


def test_endpoint_location_reads_multi_region_from_resource_name() -> None:
    # Tuned Gemini 3.x endpoints land on the us/eu multi-region, not the region.
    ep = "projects/p/locations/us/endpoints/123"
    assert endpoint_location(ep) == "us"


def test_endpoint_location_reads_region_from_resource_name() -> None:
    ep = "projects/p/locations/us-central1/endpoints/123"
    assert endpoint_location(ep) == "us-central1"


def test_endpoint_location_returns_none_for_bare_id() -> None:
    assert endpoint_location("4327537029437980672") is None


# --- client factories: transport policy ----------------------------------------
# The SDK does not retry by default (HttpOptions.retry_options is None ->
# stop_after_attempt(1)), so these assert the policy is actually attached.


def test_genai_client_attaches_retry_and_timeout(captured_client: MagicMock) -> None:
    cfg = TuningConfig(project="p", location="us-central1", bucket="gs://b")
    genai_client(cfg)
    opts = captured_client.call_args.kwargs["http_options"]
    assert opts.retry_options.attempts == 5
    assert opts.retry_options.http_status_codes == _RETRYABLE_STATUS_CODES
    assert opts.timeout == 120_000


def test_genai_client_retry_backoff_is_exponential_with_jitter(
    captured_client: MagicMock,
) -> None:
    cfg = TuningConfig(project="p", location="us-central1", bucket="gs://b")
    genai_client(cfg)
    retry = captured_client.call_args.kwargs["http_options"].retry_options
    assert retry.exp_base > 1  # actually backs off rather than retrying flat
    assert retry.initial_delay < retry.max_delay
    assert retry.jitter  # avoids synchronized retries across a sweep


def test_genai_client_for_endpoint_attaches_the_same_policy(
    captured_client: MagicMock,
) -> None:
    cfg = TuningConfig(project="p", location="us-central1", bucket="gs://b")
    genai_client_for_endpoint(cfg, "projects/p/locations/us/endpoints/1")
    opts = captured_client.call_args.kwargs["http_options"]
    assert opts.retry_options.attempts == 5
    assert opts.timeout == 120_000


def test_genai_client_still_routes_location_and_project(captured_client: MagicMock) -> None:
    # The transport policy must not disturb the existing routing behaviour.
    cfg = TuningConfig(project="proj-x", location="europe-west4", bucket="gs://b")
    genai_client(cfg, base_model="gemini-3.5-flash")
    kwargs = captured_client.call_args.kwargs
    assert kwargs["project"] == "proj-x"
    assert kwargs["location"] == "global"  # Gemini 3.x inference
    assert kwargs["vertexai"] is True


def test_load_config_rejects_the_global_location() -> None:
    # `global` serves Gemini 3.x inference but does not support tuning, so a config
    # resolving to it must fail here rather than at job-launch time.
    env = {"PROJECT_ID": "p", "BUCKET": "gs://b", "GOOGLE_CLOUD_LOCATION": "global"}
    with pytest.raises(ValueError, match="does not support tuning"):
        load_config(env)


def test_resolve_location_still_routes_inference_to_global() -> None:
    # The config-level rejection must not break per-call inference routing.
    cfg = TuningConfig(project="p", location="us-central1", bucket="gs://b")
    assert resolve_location(cfg, "gemini-3.5-flash") == "global"


# --- run suffix (re-runnable display names) --------------------------------------


def _env(**extra: str) -> dict[str, str]:
    return {"PROJECT_ID": "p", "BUCKET": "gs://b", **extra}


def test_run_suffix_defaults_to_empty_and_leaves_names_unchanged() -> None:
    cfg = load_config(_env())
    assert cfg.run_suffix == ""
    assert cfg.display_name("geap-sft-support-intent") == "geap-sft-support-intent"


def test_run_suffix_is_appended_to_display_names() -> None:
    cfg = load_config(_env(GEAP_RUN_SUFFIX="-v2"))
    assert cfg.display_name("geap-sft-support-intent") == "geap-sft-support-intent-v2"


@pytest.mark.parametrize("bad", ["_v2", "V2", "v2!", "a" * 33, "-v 2"])
def test_run_suffix_rejects_names_that_are_not_resource_id_safe(bad: str) -> None:
    # Vertex resource IDs are [a-z0-9-] only. Failing here beats a 400 from the API
    # partway into a paid run.
    with pytest.raises(ValueError, match="not resource-ID-safe"):
        load_config(_env(GEAP_RUN_SUFFIX=bad))


def test_run_suffix_accepts_plain_alphanumeric() -> None:
    # A suffix need not start with '-'; "2026run" is legal too.
    cfg = load_config(_env(GEAP_RUN_SUFFIX="2026run"))
    assert cfg.display_name("geap-sft") == "geap-sft2026run"
