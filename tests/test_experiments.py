"""Tests for Vertex AI Experiments / Managed TensorBoard tracking helpers.

The ``aiplatform`` module is monkeypatched with a ``MagicMock`` (mirroring
``tests/test_jobs.py`` patching ``geap_tuning.jobs.time.sleep``), so nothing here
touches live GCP; assertions inspect the recorded call args.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from geap_tuning import experiments


@pytest.fixture
def fake_aiplatform(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("geap_tuning.experiments.aiplatform", mock)
    return mock


def test_init_experiment_forwards_kwargs(fake_aiplatform: MagicMock) -> None:
    experiments.init_experiment("exp", project="p", location="us-central1", tensorboard="tb/1")
    kwargs = fake_aiplatform.init.call_args.kwargs
    assert kwargs["project"] == "p"
    assert kwargs["location"] == "us-central1"
    assert kwargs["experiment"] == "exp"
    assert kwargs["experiment_tensorboard"] == "tb/1"


def test_init_experiment_defaults_tensorboard_none(fake_aiplatform: MagicMock) -> None:
    experiments.init_experiment("exp", project="p", location="us-central1")
    assert fake_aiplatform.init.call_args.kwargs["experiment_tensorboard"] is None


def test_get_or_create_tensorboard_reuses_match(fake_aiplatform: MagicMock) -> None:
    fake_aiplatform.Tensorboard.list.return_value = [
        SimpleNamespace(display_name="other", resource_name="tb/other"),
        SimpleNamespace(display_name="mine", resource_name="tb/mine"),
    ]
    name = experiments.get_or_create_tensorboard("mine", project="p", location="l")
    assert name == "tb/mine"
    fake_aiplatform.Tensorboard.create.assert_not_called()


def test_get_or_create_tensorboard_creates_when_absent(fake_aiplatform: MagicMock) -> None:
    fake_aiplatform.Tensorboard.list.return_value = [
        SimpleNamespace(display_name="other", resource_name="tb/other"),
    ]
    fake_aiplatform.Tensorboard.create.return_value = SimpleNamespace(resource_name="tb/new")
    name = experiments.get_or_create_tensorboard("mine", project="p", location="l")
    assert name == "tb/new"
    assert fake_aiplatform.Tensorboard.create.call_args.kwargs["display_name"] == "mine"


def test_track_run_logs_params_and_yields_run(fake_aiplatform: MagicMock) -> None:
    run = SimpleNamespace(name="run-1")
    fake_aiplatform.start_run.return_value.__enter__.return_value = run

    with experiments.track_run("run-1", params={"epochs": 2}) as yielded:
        assert yielded is run

    assert fake_aiplatform.start_run.call_args.args[0] == "run-1"
    assert fake_aiplatform.log_params.call_args.args[0] == {"epochs": 2}


def test_track_run_skips_log_params_when_none(fake_aiplatform: MagicMock) -> None:
    with experiments.track_run("run-1"):
        pass
    fake_aiplatform.log_params.assert_not_called()


def test_log_summary_metrics_forwards(fake_aiplatform: MagicMock) -> None:
    experiments.log_summary_metrics({"accuracy": 0.9})
    assert fake_aiplatform.log_metrics.call_args.args[0] == {"accuracy": 0.9}


def test_log_timeseries_metrics_forwards_step(fake_aiplatform: MagicMock) -> None:
    experiments.log_timeseries_metrics({"accuracy": 0.9}, step=3)
    args = fake_aiplatform.log_time_series_metrics.call_args
    assert args.args[0] == {"accuracy": 0.9}
    assert args.kwargs["step"] == 3


def test_experiment_dataframe_returns_frame(fake_aiplatform: MagicMock) -> None:
    fake_aiplatform.Experiment.return_value.get_data_frame.return_value = "df"
    assert experiments.experiment_dataframe("exp") == "df"
    assert fake_aiplatform.Experiment.call_args.args[0] == "exp"
