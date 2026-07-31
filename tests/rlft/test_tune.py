"""Tests for the RLFT job launcher and reward preflight (mocked client)."""

from unittest.mock import MagicMock

import pytest
from google.genai import types

from geap_tuning.rlft.tune import (
    build_autorater_reward_config,
    build_cloud_run_reward_config,
    build_composite_reward_config,
    build_reward_config,
    build_string_match_reward_config,
    launch_rlft_job,
    validate_reward_config,
)


def test_build_reward_config_ships_reward_source() -> None:
    cfg = build_reward_config(reward_name="math_correctness")
    assert cfg.reward_name == "math_correctness"
    snippet = cfg.code_execution_reward_scorer.python_code_snippet
    assert "def evaluate(" in snippet  # the real reward source is shipped verbatim


def test_build_string_match_reward_config() -> None:
    cfg = build_string_match_reward_config()
    assert cfg.reward_name == "answer_format"
    scorer = cfg.string_match_reward_scorer
    assert scorer is not None
    assert cfg.code_execution_reward_scorer is None
    assert cfg.autorater_scorer is None
    assert scorer.correct_answer_reward == 1.0
    assert scorer.wrong_answer_reward == -1.0
    assert scorer.string_match_expression.match_operation == types.MatchOperation.REGEX_CONTAINS
    assert scorer.string_match_expression.expression == r"Answer:\s*-?\d+"


def test_build_string_match_reward_config_custom() -> None:
    cfg = build_string_match_reward_config(
        reward_name="mentions_units",
        expression="km",
        match_operation=types.MatchOperation.PARTIAL_MATCH,
        correct_answer_reward=0.5,
        wrong_answer_reward=0.0,
    )
    scorer = cfg.string_match_reward_scorer
    assert cfg.reward_name == "mentions_units"
    assert scorer.string_match_expression.match_operation == types.MatchOperation.PARTIAL_MATCH
    assert scorer.string_match_expression.expression == "km"
    assert scorer.correct_answer_reward == 0.5
    assert scorer.wrong_answer_reward == 0.0


def test_build_autorater_reward_config() -> None:
    cfg = build_autorater_reward_config()
    assert cfg.reward_name == "explanation_quality"
    scorer = cfg.autorater_scorer
    assert scorer is not None
    assert cfg.code_execution_reward_scorer is None
    assert cfg.string_match_reward_scorer is None
    assert scorer.autorater_config.sampling_count == 4
    assert "SCORE:" in scorer.autorater_prompt
    assert (
        scorer.autorater_response_parse_config.parse_type == types.ResponseParseType.REGEX_EXTRACT
    )
    assert scorer.exact_match_scorer is not None


def test_build_autorater_reward_config_custom_model() -> None:
    cfg = build_autorater_reward_config(
        reward_name="clarity",
        autorater_model="projects/p/locations/l/endpoints/e",
        sampling_count=8,
    )
    scorer = cfg.autorater_scorer
    assert cfg.reward_name == "clarity"
    assert scorer.autorater_config.autorater_model == "projects/p/locations/l/endpoints/e"
    assert scorer.autorater_config.sampling_count == 8


def test_build_cloud_run_reward_config() -> None:
    cfg = build_cloud_run_reward_config(
        reward_name="external", cloud_run_uri="https://svc-abc.a.run.app"
    )
    assert cfg.reward_name == "external"
    assert cfg.cloud_run_reward_scorer.cloud_run_uri == "https://svc-abc.a.run.app"
    assert cfg.code_execution_reward_scorer is None


def test_build_composite_reward_config() -> None:
    code = build_reward_config()
    autorater = build_autorater_reward_config()
    composite = build_composite_reward_config([(code, 0.8), (autorater, 0.2)])
    weighted = composite.weighted_reward_configs
    assert len(weighted) == 2
    assert weighted[0].reward_config is code
    assert weighted[0].weight == 0.8
    assert weighted[1].reward_config is autorater
    assert weighted[1].weight == 0.2


def test_launch_rlft_job_builds_config() -> None:
    client = MagicMock()

    launch_rlft_job(
        client,
        train_uri="gs://b/train.jsonl",
        val_uri="gs://b/val.jsonl",
        display_name="d",
        base_model="gemini-2.5-flash",
        adapter_size=16,
        samples_per_prompt=8,
    )

    kwargs = client.tunings.tune.call_args.kwargs
    assert kwargs["base_model"] == "gemini-2.5-flash"
    assert kwargs["training_dataset"].gcs_uri == "gs://b/train.jsonl"
    cfg = kwargs["config"]
    assert cfg.method == "REINFORCEMENT_TUNING"
    assert cfg.tuned_model_display_name == "d"
    assert cfg.adapter_size == "ADAPTER_SIZE_SIXTEEN"
    assert cfg.samples_per_prompt == 8
    assert cfg.reward_config.code_execution_reward_scorer is not None
    assert cfg.validation_dataset.gcs_uri == "gs://b/val.jsonl"


def test_launch_rlft_job_without_validation() -> None:
    client = MagicMock()
    launch_rlft_job(client, train_uri="gs://b/t.jsonl", display_name="d")
    assert client.tunings.tune.call_args.kwargs["config"].validation_dataset is None


def test_launch_rlft_job_rejects_unknown_adapter_size() -> None:
    with pytest.raises(KeyError):
        launch_rlft_job(MagicMock(), train_uri="gs://b/t.jsonl", display_name="d", adapter_size=7)


def test_launch_rlft_job_exports_intermediate_checkpoints_by_default() -> None:
    client = MagicMock()
    launch_rlft_job(client, train_uri="gs://b/t.jsonl", display_name="d")
    assert client.tunings.tune.call_args.kwargs["config"].export_last_checkpoint_only is False


def test_launch_rlft_job_threads_evaluation_config() -> None:
    client = MagicMock()
    eval_cfg = types.EvaluationConfig()
    launch_rlft_job(
        client, train_uri="gs://b/t.jsonl", display_name="d", evaluation_config=eval_cfg
    )
    assert client.tunings.tune.call_args.kwargs["config"].evaluation_config is eval_cfg


def test_launch_rlft_job_sets_checkpoint_interval() -> None:
    client = MagicMock()
    launch_rlft_job(client, train_uri="gs://b/t.jsonl", display_name="d", checkpoint_interval=20)
    assert client.tunings.tune.call_args.kwargs["config"].checkpoint_interval == 20


def test_launch_rlft_job_continuous_from_pretuned() -> None:
    client = MagicMock()
    launch_rlft_job(
        client,
        train_uri="gs://b/t.jsonl",
        display_name="d",
        base_model="projects/p/locations/l/models/m@1",
        pre_tuned_model_checkpoint_id="2",
    )
    kwargs = client.tunings.tune.call_args.kwargs
    assert kwargs["base_model"] == "projects/p/locations/l/models/m@1"
    assert kwargs["config"].pre_tuned_model_checkpoint_id == "2"


def test_launch_rlft_job_with_composite_reward() -> None:
    client = MagicMock()
    composite = build_composite_reward_config(
        [(build_reward_config(), 0.8), (build_autorater_reward_config(), 0.2)]
    )
    launch_rlft_job(
        client,
        train_uri="gs://b/t.jsonl",
        display_name="d",
        composite_reward_config=composite,
    )
    cfg = client.tunings.tune.call_args.kwargs["config"]
    assert cfg.composite_reward_config is composite
    assert cfg.reward_config is None  # single and composite are mutually exclusive


def test_validate_reward_config_accepts_composite() -> None:
    client = MagicMock()
    composite = build_composite_reward_config([(build_reward_config(), 1.0)])
    record = {
        "contents": [{"role": "user", "parts": [{"text": "2+2?"}]}],
        "references": {"ground_truth_answer": "4"},
    }
    validate_reward_config(
        client,
        project="p",
        location="l",
        sample_answer="Answer: 4",
        example_record=record,
        composite_reward_config=composite,
    )
    kwargs = client.tunings.validate_reward.call_args.kwargs
    assert kwargs["composite_reward_config"] is composite
    assert kwargs["single_reward_config"] is None


def test_validate_reward_config_targets_parent() -> None:
    client = MagicMock()
    record = {
        "contents": [{"role": "user", "parts": [{"text": "2+2?"}]}],
        "references": {"ground_truth_answer": "4"},
    }
    validate_reward_config(
        client,
        project="p",
        location="l",
        sample_answer="Answer: 4",
        example_record=record,
    )
    kwargs = client.tunings.validate_reward.call_args.kwargs
    assert kwargs["parent"] == "projects/p/locations/l"
    assert kwargs["single_reward_config"].code_execution_reward_scorer is not None


def test_launch_rlft_job_threads_evaluate_interval() -> None:
    client = MagicMock()
    launch_rlft_job(client, train_uri="gs://b/t.jsonl", display_name="d", evaluate_interval=50)
    assert client.tunings.tune.call_args.kwargs["config"].evaluate_interval == 50


def test_launch_rlft_job_evaluate_interval_defaults_none() -> None:
    client = MagicMock()
    launch_rlft_job(client, train_uri="gs://b/t.jsonl", display_name="d")
    assert client.tunings.tune.call_args.kwargs["config"].evaluate_interval is None


def test_launch_rlft_job_threads_labels() -> None:
    client = MagicMock()
    launch_rlft_job(
        client, train_uri="gs://b/t.jsonl", display_name="d", labels={"project": "geap-tuning"}
    )
    assert client.tunings.tune.call_args.kwargs["config"].labels == {"project": "geap-tuning"}


def test_launch_rlft_job_labels_default_none() -> None:
    client = MagicMock()
    launch_rlft_job(client, train_uri="gs://b/t.jsonl", display_name="d")
    assert client.tunings.tune.call_args.kwargs["config"].labels is None
