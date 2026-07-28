"""Tests for environment/config resolution."""

import pytest

from geap_tuning.config import (
    TuningConfig,
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
