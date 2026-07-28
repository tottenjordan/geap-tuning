"""Tests for the SFT job launcher (mocked client — no live tuning)."""

from unittest.mock import MagicMock

import pytest

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
