"""Tests for the SFT job launcher (mocked client — no live tuning)."""

from unittest.mock import MagicMock

import pytest
from google.genai import types

from geap_tuning.sft.tune import ADAPTER_MAP, launch_sft_job


def test_adapter_map() -> None:
    assert ADAPTER_MAP[8] == "ADAPTER_SIZE_EIGHT"
    assert set(ADAPTER_MAP) == {1, 2, 4, 8, 16}


def test_launch_sft_job_builds_config() -> None:
    client = MagicMock()

    launch_sft_job(
        client,
        train_uri="gs://b/train.jsonl",
        val_uri="gs://b/val.jsonl",
        base_model="gemini-2.5-flash",
        display_name="d",
        epochs=2,
        adapter_size=8,
        learning_rate_multiplier=1.0,
    )

    kwargs = client.tunings.tune.call_args.kwargs
    assert kwargs["base_model"] == "gemini-2.5-flash"
    assert kwargs["training_dataset"].gcs_uri == "gs://b/train.jsonl"
    cfg = kwargs["config"]
    assert cfg.tuned_model_display_name == "d"
    assert cfg.epoch_count == 2
    assert cfg.adapter_size == "ADAPTER_SIZE_EIGHT"
    assert cfg.validation_dataset.gcs_uri == "gs://b/val.jsonl"


def test_launch_sft_job_without_validation() -> None:
    client = MagicMock()
    launch_sft_job(client, train_uri="gs://b/train.jsonl", display_name="d")
    cfg = client.tunings.tune.call_args.kwargs["config"]
    assert cfg.validation_dataset is None


def test_launch_sft_job_rejects_unknown_adapter_size() -> None:
    client = MagicMock()
    with pytest.raises(KeyError):
        launch_sft_job(client, train_uri="gs://b/t.jsonl", display_name="d", adapter_size=7)


def test_launch_sft_job_exports_intermediate_checkpoints_by_default() -> None:
    client = MagicMock()
    launch_sft_job(client, train_uri="gs://b/t.jsonl", display_name="d")
    assert client.tunings.tune.call_args.kwargs["config"].export_last_checkpoint_only is False


def test_launch_sft_job_can_export_last_checkpoint_only() -> None:
    client = MagicMock()
    launch_sft_job(
        client, train_uri="gs://b/t.jsonl", display_name="d", export_last_checkpoint_only=True
    )
    assert client.tunings.tune.call_args.kwargs["config"].export_last_checkpoint_only is True


def test_launch_sft_job_threads_evaluation_config() -> None:
    client = MagicMock()
    eval_cfg = types.EvaluationConfig()
    launch_sft_job(client, train_uri="gs://b/t.jsonl", display_name="d", evaluation_config=eval_cfg)
    assert client.tunings.tune.call_args.kwargs["config"].evaluation_config is eval_cfg


def test_launch_sft_job_continuous_from_pretuned() -> None:
    client = MagicMock()
    launch_sft_job(
        client,
        train_uri="gs://b/t.jsonl",
        display_name="d",
        base_model="projects/p/locations/l/models/m@1",
        pre_tuned_model_checkpoint_id="2",
    )
    kwargs = client.tunings.tune.call_args.kwargs
    assert kwargs["base_model"] == "projects/p/locations/l/models/m@1"
    assert kwargs["config"].pre_tuned_model_checkpoint_id == "2"


def test_launch_sft_job_threads_labels() -> None:
    client = MagicMock()
    launch_sft_job(
        client, train_uri="gs://b/t.jsonl", display_name="d", labels={"project": "geap-tuning"}
    )
    assert client.tunings.tune.call_args.kwargs["config"].labels == {"project": "geap-tuning"}


def test_launch_sft_job_labels_default_none() -> None:
    client = MagicMock()
    launch_sft_job(client, train_uri="gs://b/t.jsonl", display_name="d")
    assert client.tunings.tune.call_args.kwargs["config"].labels is None


def test_launch_sft_job_rejects_evaluate_interval() -> None:
    # evaluate_interval is RLFT-only in google-genai 2.14.0: the SDK serializes it
    # under reinforcementTuningSpec, so an SFT job carrying it 400s at the API.
    # The launcher deliberately omits the param, so passing it is a TypeError.
    client = MagicMock()
    with pytest.raises(TypeError):
        launch_sft_job(
            client,
            train_uri="gs://b/t.jsonl",
            display_name="d",
            evaluate_interval=50,  # ty: ignore[unknown-argument]
        )
