"""Tests for the tuning auto-eval config builder."""

from google.genai import types

from geap_tuning.autoeval import build_evaluation_config


def test_build_evaluation_config_defaults() -> None:
    config = build_evaluation_config("gs://my-bucket")
    assert isinstance(config, types.EvaluationConfig)
    assert config.metrics
    assert config.metrics[0].name
    prefix = config.output_config.gcs_destination.output_uri_prefix
    assert prefix.startswith("gs://my-bucket/")
    assert "tuning_eval" in prefix


def test_build_evaluation_config_custom_prefix() -> None:
    config = build_evaluation_config("my-bucket", prefix="eval_out")
    prefix = config.output_config.gcs_destination.output_uri_prefix
    assert prefix == "gs://my-bucket/eval_out"


def test_build_evaluation_config_custom_metrics() -> None:
    metrics = [types.Metric(name="safety")]
    config = build_evaluation_config("gs://b", metrics=metrics)
    assert config.metrics[0].name == "safety"
