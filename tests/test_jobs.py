"""Tests for tuning job monitoring and helpers."""

import datetime
import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from geap_tuning.jobs import (
    cancel_tuning_job,
    cancel_tuning_job_by_display_name,
    checkpoint_endpoint,
    find_tuning_job_by_display_name,
    get_default_checkpoint_id,
    job_training_uri,
    list_checkpoints,
    set_default_checkpoint,
    tuned_endpoint,
    tuned_model_name,
    wait_for_tuning_job,
)
from geap_tuning.logs import configure_logging
from tests.conftest import make_checkpoint, make_job


def _job_with_checkpoints(
    checkpoints: list[SimpleNamespace] | None,
    *,
    model: str | None = "projects/p/locations/l/models/m@1",
) -> SimpleNamespace:
    return SimpleNamespace(
        tuned_model=SimpleNamespace(endpoint=None, model=model, checkpoints=checkpoints)
    )


def test_tuned_endpoint_prefers_endpoint() -> None:
    assert tuned_endpoint(make_job(endpoint="ep", model="m")) == "ep"
    assert tuned_endpoint(make_job(endpoint=None, model="m")) == "m"


def test_tuned_endpoint_raises_when_missing() -> None:
    with pytest.raises(ValueError, match="no tuned endpoint"):
        tuned_endpoint(make_job(endpoint=None))


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
    ckpts = [make_checkpoint("1"), make_checkpoint("2")]
    assert list_checkpoints(_job_with_checkpoints(ckpts)) == ckpts


def test_list_checkpoints_empty_when_none() -> None:
    assert list_checkpoints(_job_with_checkpoints(None)) == []


def test_list_checkpoints_empty_when_attr_absent() -> None:
    job = SimpleNamespace(tuned_model=SimpleNamespace(endpoint=None, model="m"))
    assert list_checkpoints(job) == []


def test_checkpoint_endpoint_returns_match() -> None:
    job = _job_with_checkpoints(
        [make_checkpoint("1", endpoint="ep1"), make_checkpoint("2", endpoint="ep2")]
    )
    assert checkpoint_endpoint(job, "2") == "ep2"


def test_checkpoint_endpoint_raises_when_not_found() -> None:
    job = _job_with_checkpoints([make_checkpoint("1", endpoint="ep1")])
    with pytest.raises(ValueError, match="No checkpoint"):
        checkpoint_endpoint(job, "9")


def test_checkpoint_endpoint_raises_when_endpoint_empty() -> None:
    job = _job_with_checkpoints([make_checkpoint("1", endpoint="")])
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


# --- wait_for_tuning_job: bounding, heartbeat, failure detail -------------------


def test_wait_times_out_instead_of_polling_forever(monkeypatch: pytest.MonkeyPatch) -> None:
    # A job wedged in a non-terminal state must not spin indefinitely.
    client = MagicMock()
    client.tunings.get.return_value = SimpleNamespace(state="JOB_STATE_RUNNING", name="n")
    monkeypatch.setattr("geap_tuning.jobs.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="still JOB_STATE_RUNNING"):
        wait_for_tuning_job(client, "n", poll_interval=0, timeout=0)


def test_wait_timeout_none_waits_indefinitely(monkeypatch: pytest.MonkeyPatch) -> None:
    # timeout=None preserves the original unbounded behaviour; it must still return
    # once the job reaches a terminal state.
    client = MagicMock()
    client.tunings.get.side_effect = [
        SimpleNamespace(state="JOB_STATE_RUNNING", name="n"),
        SimpleNamespace(state="JOB_STATE_SUCCEEDED", name="n"),
    ]
    monkeypatch.setattr("geap_tuning.jobs.time.sleep", lambda _: None)

    job = wait_for_tuning_job(client, "n", poll_interval=0, timeout=None)
    assert job.state == "JOB_STATE_SUCCEEDED"


def test_wait_logs_a_heartbeat(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = MagicMock()
    client.tunings.get.side_effect = [
        SimpleNamespace(state="JOB_STATE_RUNNING", name="n"),
        SimpleNamespace(state="JOB_STATE_SUCCEEDED", name="n"),
    ]
    monkeypatch.setattr("geap_tuning.jobs.time.sleep", lambda _: None)

    with caplog.at_level(logging.INFO, logger="geap_tuning"):
        wait_for_tuning_job(client, "projects/p/locations/l/tuningJobs/123", poll_interval=0)

    # The message is terse; the specifics ride on the record as structured fields.
    records = [r for r in caplog.records if hasattr(r, "state")]
    assert records
    assert {r.job for r in records} == {"123"}  # short job id, not the resource path
    states = [r.state for r in records]
    assert "JOB_STATE_RUNNING" in states
    assert "JOB_STATE_SUCCEEDED" in states  # a state change always emits
    assert all(hasattr(r, "elapsed_min") for r in records)


def test_wait_heartbeat_can_be_silenced(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = MagicMock()
    client.tunings.get.return_value = SimpleNamespace(state="JOB_STATE_SUCCEEDED", name="n")
    monkeypatch.setattr("geap_tuning.jobs.time.sleep", lambda _: None)

    with caplog.at_level(logging.INFO, logger="geap_tuning"):
        wait_for_tuning_job(client, "n", poll_interval=0, heartbeat=False)
    assert caplog.records == []


def test_wait_surfaces_the_sdk_error_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without this the caller gets only a state string and must open the console.
    client = MagicMock()
    client.tunings.get.return_value = SimpleNamespace(
        state="JOB_STATE_FAILED", name="n", error="quota exceeded for adapter size 16"
    )
    monkeypatch.setattr("geap_tuning.jobs.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="quota exceeded for adapter size 16"):
        wait_for_tuning_job(client, "n", poll_interval=0)


# --- reuse: recency + dataset awareness (W-6) -----------------------------------


def _listed(display_name: str, *, created: int, uri: str | None = None, name: str = "j") -> object:
    spec = SimpleNamespace(training_dataset_uri=uri) if uri else None
    return SimpleNamespace(
        name=name,
        tuned_model_display_name=display_name,
        state="JOB_STATE_SUCCEEDED",
        create_time=datetime.datetime(2026, 1, created, tzinfo=datetime.UTC),
        supervised_tuning_spec=spec,
        preference_optimization_spec=None,
        reinforcement_tuning_spec=None,
    )


def test_find_returns_the_most_recent_match() -> None:
    # A FAILED job triggers a relaunch under the same name, so duplicates happen.
    client = MagicMock()
    client.tunings.list.return_value = [
        _listed("d", created=1, name="old"),
        _listed("d", created=9, name="new"),
        _listed("d", created=5, name="mid"),
    ]
    assert find_tuning_job_by_display_name(client, "d").name == "new"


def test_find_tolerates_a_missing_create_time() -> None:
    client = MagicMock()
    undated = _listed("d", created=1, name="undated")
    undated.create_time = None
    client.tunings.list.return_value = [undated, _listed("d", created=2, name="dated")]
    assert find_tuning_job_by_display_name(client, "d").name == "dated"


def test_find_skips_a_job_trained_on_a_different_dataset() -> None:
    client = MagicMock()
    client.tunings.list.return_value = [_listed("d", created=1, uri="gs://b/old.jsonl")]
    assert find_tuning_job_by_display_name(client, "d", train_uri="gs://b/new.jsonl") is None


def test_find_reuses_a_job_trained_on_the_same_dataset() -> None:
    client = MagicMock()
    client.tunings.list.return_value = [_listed("d", created=1, uri="gs://b/t.jsonl", name="ok")]
    found = find_tuning_job_by_display_name(client, "d", train_uri="gs://b/t.jsonl")
    assert found.name == "ok"


def test_find_reuses_when_the_job_reports_no_uri() -> None:
    # Absent metadata must not silently block reuse of an otherwise-valid job.
    client = MagicMock()
    client.tunings.list.return_value = [_listed("d", created=1, name="nouri")]
    assert find_tuning_job_by_display_name(client, "d", train_uri="gs://b/t.jsonl").name == "nouri"


def test_job_training_uri_reads_each_method_spec() -> None:
    def job(**specs: object) -> SimpleNamespace:
        base = {
            "supervised_tuning_spec": None,
            "preference_optimization_spec": None,
            "reinforcement_tuning_spec": None,
        }
        return SimpleNamespace(**{**base, **specs})

    sft = job(supervised_tuning_spec=SimpleNamespace(training_dataset_uri="gs://b/sft"))
    dpo = job(preference_optimization_spec=SimpleNamespace(training_dataset_uri="gs://b/dpo"))
    rlft = job(reinforcement_tuning_spec=SimpleNamespace(training_dataset_uri="gs://b/rlft"))
    assert job_training_uri(sft) == "gs://b/sft"
    assert job_training_uri(dpo) == "gs://b/dpo"
    assert job_training_uri(rlft) == "gs://b/rlft"
    assert job_training_uri(job()) is None


def _labelled(fingerprint: str | None, *, name: str = "j") -> SimpleNamespace:
    job = _listed("d", created=1, name=name)
    job.labels = {"data_fingerprint": fingerprint} if fingerprint else {}
    return job


def test_find_skips_a_job_trained_on_different_bytes_at_the_same_uri() -> None:
    # The residual W-6 case: same staging path, edited dataset.
    client = MagicMock()
    client.tunings.list.return_value = [_labelled("aaaa1111")]
    assert find_tuning_job_by_display_name(client, "d", data_fingerprint="bbbb2222") is None


def test_find_reuses_a_job_trained_on_the_same_bytes() -> None:
    client = MagicMock()
    client.tunings.list.return_value = [_labelled("aaaa1111", name="same")]
    found = find_tuning_job_by_display_name(client, "d", data_fingerprint="aaaa1111")
    assert found.name == "same"


def test_find_reuses_an_unfingerprinted_job_but_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Jobs launched before fingerprints were recorded cannot be compared; reuse is
    # allowed, but the ambiguity must be visible rather than silent.
    client = MagicMock()
    client.tunings.list.return_value = [_labelled(None, name="legacy")]
    with caplog.at_level(logging.WARNING, logger="geap_tuning"):
        found = find_tuning_job_by_display_name(client, "d", data_fingerprint="aaaa1111")
    assert found.name == "legacy"
    assert "no dataset fingerprint" in caplog.text
    assert caplog.records[0].levelname == "WARNING"  # it is a caveat, not routine info


def test_find_ignores_fingerprints_when_none_requested(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MagicMock()
    client.tunings.list.return_value = [_labelled("aaaa1111", name="any")]
    with caplog.at_level(logging.WARNING, logger="geap_tuning"):
        assert find_tuning_job_by_display_name(client, "d").name == "any"
    assert caplog.records == []  # no note when the check is not requested


def test_heartbeat_reaches_a_redirected_stream_immediately() -> None:
    """The reason this used to need ``flush=True`` on a print.

    stdout is block-buffered when redirected, so a heartbeat meant to report
    progress during a 30-60 minute wait stayed invisible until exit. A
    ``StreamHandler`` flushes on every emit, so the record is readable straight
    away — asserted here against an in-memory stream.
    """
    buffer = io.StringIO()
    configure_logging(stream=buffer)
    client = MagicMock()
    client.tunings.get.return_value = SimpleNamespace(state="JOB_STATE_SUCCEEDED", name="n")
    with patch("geap_tuning.jobs.time.sleep", lambda _: None):
        wait_for_tuning_job(client, "projects/p/locations/l/tuningJobs/1", poll_interval=0)
    assert "JOB_STATE_SUCCEEDED" in buffer.getvalue()


def test_heartbeat_state_field_is_the_bare_enum_value() -> None:
    """``str()`` on the SDK's JobState yields 'JobState.JOB_STATE_RUNNING'.

    A structured field consumed by a log pipeline wants the bare value, so the
    heartbeat reads ``.value`` rather than stringifying the enum.
    """
    buffer = io.StringIO()
    configure_logging(fmt="json", stream=buffer)
    client = MagicMock()
    client.tunings.get.return_value = SimpleNamespace(
        state=types.JobState.JOB_STATE_SUCCEEDED, name="n"
    )
    with patch("geap_tuning.jobs.time.sleep", lambda _: None):
        wait_for_tuning_job(client, "jobs/1", poll_interval=0)
    assert json.loads(buffer.getvalue().strip())["state"] == "JOB_STATE_SUCCEEDED"
