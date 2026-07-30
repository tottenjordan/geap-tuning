"""Tests for tuning job monitoring and helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from geap_tuning.jobs import (
    cancel_tuning_job,
    cancel_tuning_job_by_display_name,
    checkpoint_endpoint,
    find_tuning_job_by_display_name,
    get_default_checkpoint_id,
    list_checkpoints,
    set_default_checkpoint,
    tuned_endpoint,
    tuned_model_name,
    wait_for_tuning_job,
)


def _job(endpoint: str | None = None, model: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(tuned_model=SimpleNamespace(endpoint=endpoint, model=model))


def _checkpoint(checkpoint_id: str, *, endpoint: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_id=checkpoint_id,
        epoch=1,
        step=10,
        endpoint=endpoint,
    )


def _job_with_checkpoints(
    checkpoints: list[SimpleNamespace] | None,
    *,
    model: str | None = "projects/p/locations/l/models/m@1",
) -> SimpleNamespace:
    return SimpleNamespace(
        tuned_model=SimpleNamespace(endpoint=None, model=model, checkpoints=checkpoints)
    )


def test_tuned_endpoint_prefers_endpoint() -> None:
    assert tuned_endpoint(_job(endpoint="ep", model="m")) == "ep"
    assert tuned_endpoint(_job(model="m")) == "m"


def test_tuned_endpoint_raises_when_missing() -> None:
    with pytest.raises(ValueError, match="no tuned endpoint"):
        tuned_endpoint(_job())


def test_find_by_display_name_returns_match() -> None:
    client = MagicMock()
    client.tunings.list.return_value = [
        SimpleNamespace(tuned_model_display_name="other", state="JOB_STATE_SUCCEEDED"),
        SimpleNamespace(tuned_model_display_name="mine", state="JOB_STATE_SUCCEEDED"),
    ]
    job = find_tuning_job_by_display_name(client, "mine")
    assert job is not None
    assert job.tuned_model_display_name == "mine"


def test_find_by_display_name_returns_none_when_absent() -> None:
    client = MagicMock()
    client.tunings.list.return_value = [
        SimpleNamespace(tuned_model_display_name="other", state="JOB_STATE_SUCCEEDED"),
    ]
    assert find_tuning_job_by_display_name(client, "mine") is None


def test_wait_polls_until_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    states = iter(
        [
            SimpleNamespace(state="JOB_STATE_RUNNING", name="n"),
            SimpleNamespace(state="JOB_STATE_SUCCEEDED", name="n"),
        ]
    )
    client.tunings.get.side_effect = lambda name: next(states)  # noqa: ARG005
    monkeypatch.setattr("geap_tuning.jobs.time.sleep", lambda _: None)

    job = wait_for_tuning_job(client, "n", poll_interval=0)
    assert job.state == "JOB_STATE_SUCCEEDED"


def test_wait_raises_on_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.tunings.get.return_value = SimpleNamespace(state="JOB_STATE_FAILED", name="n")
    monkeypatch.setattr("geap_tuning.jobs.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="JOB_STATE_FAILED"):
        wait_for_tuning_job(client, "n", poll_interval=0)


def test_list_checkpoints_returns_list() -> None:
    ckpts = [_checkpoint("1"), _checkpoint("2")]
    assert list_checkpoints(_job_with_checkpoints(ckpts)) == ckpts


def test_list_checkpoints_empty_when_none() -> None:
    assert list_checkpoints(_job_with_checkpoints(None)) == []


def test_list_checkpoints_empty_when_attr_absent() -> None:
    job = SimpleNamespace(tuned_model=SimpleNamespace(endpoint=None, model="m"))
    assert list_checkpoints(job) == []


def test_checkpoint_endpoint_returns_match() -> None:
    job = _job_with_checkpoints(
        [_checkpoint("1", endpoint="ep1"), _checkpoint("2", endpoint="ep2")]
    )
    assert checkpoint_endpoint(job, "2") == "ep2"


def test_checkpoint_endpoint_raises_when_not_found() -> None:
    job = _job_with_checkpoints([_checkpoint("1", endpoint="ep1")])
    with pytest.raises(ValueError, match="No checkpoint"):
        checkpoint_endpoint(job, "9")


def test_checkpoint_endpoint_raises_when_endpoint_empty() -> None:
    job = _job_with_checkpoints([_checkpoint("1", endpoint="")])
    with pytest.raises(ValueError, match="no endpoint"):
        checkpoint_endpoint(job, "1")


def test_tuned_model_name_returns_resource_name() -> None:
    job = _job_with_checkpoints(None, model="projects/p/locations/l/models/m@1")
    assert tuned_model_name(job) == "projects/p/locations/l/models/m@1"


def test_tuned_model_name_raises_when_empty() -> None:
    with pytest.raises(ValueError, match="no tuned model"):
        tuned_model_name(_job_with_checkpoints(None, model=None))


def test_get_default_checkpoint_id() -> None:
    client = MagicMock()
    client.models.get.return_value = SimpleNamespace(default_checkpoint_id="2")
    job = _job_with_checkpoints(None)
    assert get_default_checkpoint_id(client, job) == "2"
    assert client.models.get.call_args.kwargs["model"] == "projects/p/locations/l/models/m@1"


def test_set_default_checkpoint() -> None:
    client = MagicMock()
    job = _job_with_checkpoints(None)
    set_default_checkpoint(client, job, "1")
    kwargs = client.models.update.call_args.kwargs
    assert kwargs["model"] == "projects/p/locations/l/models/m@1"
    assert kwargs["config"].default_checkpoint_id == "1"


def test_cancel_tuning_job_calls_sdk() -> None:
    client = MagicMock()
    name = "projects/p/locations/l/tuningJobs/123"
    result = cancel_tuning_job(client, name)
    client.tunings.cancel.assert_called_once_with(name=name)
    assert result is client.tunings.cancel.return_value


def test_cancel_tuning_job_by_display_name_found() -> None:
    client = MagicMock()
    client.tunings.list.return_value = [
        SimpleNamespace(
            tuned_model_display_name="geap-rlft-math-v1",
            state="JOB_STATE_RUNNING",
            name="projects/p/locations/l/tuningJobs/999",
        ),
    ]
    name = cancel_tuning_job_by_display_name(client, "geap-rlft-math-v1")
    assert name == "projects/p/locations/l/tuningJobs/999"
    client.tunings.cancel.assert_called_once_with(name="projects/p/locations/l/tuningJobs/999")


def test_cancel_tuning_job_by_display_name_missing() -> None:
    client = MagicMock()
    client.tunings.list.return_value = [
        SimpleNamespace(tuned_model_display_name="other", state="JOB_STATE_RUNNING", name="n"),
    ]
    assert cancel_tuning_job_by_display_name(client, "geap-rlft-math-v1") is None
    client.tunings.cancel.assert_not_called()
