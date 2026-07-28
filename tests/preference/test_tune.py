"""Tests for the DPO job launcher (mocked client — no live tuning)."""

from unittest.mock import MagicMock

import pytest

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
