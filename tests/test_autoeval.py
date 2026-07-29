"""Tests for the tuning auto-eval config and metric builders."""

from google.genai import types

from geap_tuning.autoeval import (
    build_autorater_config,
    build_evaluation_config,
    computation_metric,
    llm_judge_metric,
    predefined_metric,
)


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


def test_build_evaluation_config_no_autorater_or_inference_by_default() -> None:
    config = build_evaluation_config("gs://b")
    assert config.autorater_config is None
    assert config.inference_generation_config is None


def test_build_evaluation_config_threads_autorater_and_inference() -> None:
    autorater = build_autorater_config(sampling_count=8)
    inference = types.GenerationConfig(temperature=0.0)
    config = build_evaluation_config(
        "gs://b",
        autorater_config=autorater,
        inference_generation_config=inference,
    )
    assert config.autorater_config is autorater
    assert config.inference_generation_config is inference


def test_llm_judge_metric() -> None:
    metric = llm_judge_metric("coherence", "Rate the coherence of {prediction}")
    assert isinstance(metric, types.Metric)
    assert metric.prompt_template == "Rate the coherence of {prediction}"
    # SDK lowercases the metric name.
    assert metric.name == "coherence"


def test_llm_judge_metric_with_system_instruction() -> None:
    metric = llm_judge_metric(
        "coherence",
        "Rate {prediction}",
        judge_model_system_instruction="You are a strict grader.",
    )
    assert metric.judge_model_system_instruction == "You are a strict grader."


def test_computation_metric_bleu() -> None:
    metric = computation_metric(types.ComputationBasedMetricType.BLEU)
    assert isinstance(metric, types.UnifiedMetric)
    assert metric.computation_based_metric_spec.type == types.ComputationBasedMetricType.BLEU
    assert metric.predefined_metric_spec is None


def test_computation_metric_exact_match() -> None:
    metric = computation_metric(types.ComputationBasedMetricType.EXACT_MATCH)
    assert metric.computation_based_metric_spec.type == types.ComputationBasedMetricType.EXACT_MATCH


def test_predefined_metric() -> None:
    metric = predefined_metric("text_quality_v1")
    assert isinstance(metric, types.UnifiedMetric)
    assert metric.predefined_metric_spec.metric_spec_name == "text_quality_v1"
    assert metric.computation_based_metric_spec is None


def test_build_autorater_config() -> None:
    config = build_autorater_config()
    assert isinstance(config, types.AutoraterConfig)
    assert config.sampling_count == 4
    assert config.flip_enabled is None
    assert config.autorater_model is None


def test_build_autorater_config_custom() -> None:
    config = build_autorater_config(
        sampling_count=8,
        flip_enabled=True,
        autorater_model="projects/p/locations/l/endpoints/e",
    )
    assert config.sampling_count == 8
    assert config.flip_enabled is True
    assert config.autorater_model == "projects/p/locations/l/endpoints/e"
