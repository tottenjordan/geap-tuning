"""Tests for tuning job monitoring and helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from geap_tuning.jobs import (
    find_tuning_job_by_display_name,
    tuned_endpoint,
    wait_for_tuning_job,
)


def _job(endpoint: str | None = None, model: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(tuned_model=SimpleNamespace(endpoint=endpoint, model=model))


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
