"""Tests for the DPO job launcher (mocked client — no live tuning)."""

from unittest.mock import MagicMock

import pytest
from google.genai import types

from geap_tuning.preference.tune import launch_preference_job


def test_launch_preference_job_builds_config() -> None:
    client = MagicMock()

    launch_preference_job(
        client,
        train_uri="gs://b/train.jsonl",
        val_uri="gs://b/val.jsonl",
        display_name="d",
        base_model="gemini-2.5-flash",
        beta=0.1,
        adapter_size=8,
    )

    kwargs = client.tunings.tune.call_args.kwargs
    assert kwargs["base_model"] == "gemini-2.5-flash"
    assert kwargs["training_dataset"].gcs_uri == "gs://b/train.jsonl"
    cfg = kwargs["config"]
    assert cfg.method == "PREFERENCE_TUNING"
    assert cfg.beta == 0.1
    assert cfg.tuned_model_display_name == "d"
    assert cfg.adapter_size == "ADAPTER_SIZE_EIGHT"
    assert cfg.validation_dataset.gcs_uri == "gs://b/val.jsonl"


def test_launch_preference_job_without_validation() -> None:
    client = MagicMock()
    launch_preference_job(client, train_uri="gs://b/train.jsonl", display_name="d")
    cfg = client.tunings.tune.call_args.kwargs["config"]
    assert cfg.validation_dataset is None


def test_launch_preference_job_rejects_unknown_adapter_size() -> None:
    with pytest.raises(KeyError):
        launch_preference_job(
            MagicMock(), train_uri="gs://b/t.jsonl", display_name="d", adapter_size=7
        )


def test_launch_preference_job_exports_intermediate_checkpoints_by_default() -> None:
    client = MagicMock()
    launch_preference_job(client, train_uri="gs://b/t.jsonl", display_name="d")
    assert client.tunings.tune.call_args.kwargs["config"].export_last_checkpoint_only is False


def test_launch_preference_job_threads_evaluation_config() -> None:
    client = MagicMock()
    eval_cfg = types.EvaluationConfig()
    launch_preference_job(
        client, train_uri="gs://b/t.jsonl", display_name="d", evaluation_config=eval_cfg
    )
    assert client.tunings.tune.call_args.kwargs["config"].evaluation_config is eval_cfg


def test_launch_preference_job_continuous_from_pretuned() -> None:
    client = MagicMock()
    launch_preference_job(
        client,
        train_uri="gs://b/t.jsonl",
        display_name="d",
        base_model="projects/p/locations/l/models/m@1",
        pre_tuned_model_checkpoint_id="2",
    )
    kwargs = client.tunings.tune.call_args.kwargs
    assert kwargs["base_model"] == "projects/p/locations/l/models/m@1"
    assert kwargs["config"].pre_tuned_model_checkpoint_id == "2"
