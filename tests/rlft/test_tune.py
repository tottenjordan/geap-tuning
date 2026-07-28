"""Tests for the RLFT job launcher and reward preflight (mocked client)."""

from unittest.mock import MagicMock

import pytest

from geap_tuning.rlft.tune import (
    build_reward_config,
    launch_rlft_job,
    validate_reward_config,
)


def test_build_reward_config_ships_reward_source() -> None:
    cfg = build_reward_config(reward_name="math_correctness")
    assert cfg.reward_name == "math_correctness"
    snippet = cfg.code_execution_reward_scorer.python_code_snippet
    assert "def evaluate(" in snippet  # the real reward source is shipped verbatim


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
