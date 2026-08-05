"""Tests for environment/config resolution."""

import pytest

from geap_tuning.config import (
    TuningConfig,
    endpoint_location,
    load_config,
    requires_global_endpoint,
    resolve_location,
)


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
    ep = "projects/934903580331/locations/us/endpoints/4327537029437980672"
    assert endpoint_location(ep) == "us"


def test_endpoint_location_reads_region_from_resource_name() -> None:
    ep = "projects/p/locations/us-central1/endpoints/123"
    assert endpoint_location(ep) == "us-central1"


def test_endpoint_location_returns_none_for_bare_id() -> None:
    assert endpoint_location("4327537029437980672") is None
