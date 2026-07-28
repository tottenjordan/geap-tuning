"""Tests for environment/config resolution."""

import pytest

from geap_tuning.config import load_config


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
