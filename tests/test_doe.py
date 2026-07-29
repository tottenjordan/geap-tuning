"""Tests for the DOE (design-of-experiments) sweep framework.

The pure helpers (grid expansion, slugging, spec building, best-selection,
aggregation) are tested directly with plain data. The ``run_sweep`` driver is
tested with a ``MagicMock`` client and injected seams, and the Experiments
logging is monkeypatched (mirroring ``tests/test_experiments.py``) so nothing
here touches live GCP.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from geap_tuning import doe
from geap_tuning.doe import (
    RunResult,
    RunSpec,
    SweepConfig,
    aggregate_results,
    build_run_specs,
    collect_checkpoint_curve,
    expand_grid,
    run_spec_slug,
    run_sweep,
    select_best_run,
)

# --- expand_grid ---------------------------------------------------------------


def test_expand_grid_full_factorial_count() -> None:
    points = expand_grid({"epochs": [1, 2], "adapter_size": [4, 8, 16]})
    assert len(points) == 6  # 2 * 3
    assert {"epochs": 1, "adapter_size": 4} in points
    assert {"epochs": 2, "adapter_size": 16} in points


def test_expand_grid_empty_is_single_empty_point() -> None:
    assert expand_grid({}) == [{}]


def test_expand_grid_single_key() -> None:
    assert expand_grid({"epochs": [1, 2, 3]}) == [
        {"epochs": 1},
        {"epochs": 2},
        {"epochs": 3},
    ]


def test_expand_grid_is_deterministic_over_sorted_keys() -> None:
    a = expand_grid({"b": [1], "a": [2]})
    b = expand_grid({"a": [2], "b": [1]})
    assert a == b == [{"a": 2, "b": 1}]


# --- run_spec_slug -------------------------------------------------------------


def test_run_spec_slug_sorted_and_joined() -> None:
    assert run_spec_slug({"epochs": 2, "adapter_size": 8}) == "adapter_size8-epochs2"


def test_run_spec_slug_empty_point() -> None:
    assert run_spec_slug({}) == "default"


def test_run_spec_slug_floats_are_display_safe() -> None:
    slug = run_spec_slug({"learning_rate_multiplier": 1.0})
    assert slug == "learning_rate_multiplier1_0"
    assert "." not in slug


def test_run_spec_slug_is_deterministic() -> None:
    point = {"adapter_size": 4, "epochs": 1}
    assert run_spec_slug(point) == run_spec_slug(dict(reversed(point.items())))


# --- build_run_specs -----------------------------------------------------------


def test_build_run_specs_count_and_display_name() -> None:
    sweep = SweepConfig(name="cheap", grid={"epochs": [1, 2], "adapter_size": [4, 8]})
    specs = build_run_specs(sweep)
    assert len(specs) == 4
    for spec in specs:
        assert spec.display_name.startswith("geap-doe-cheap-")
        assert spec.base_model == "gemini-2.5-flash-lite"


def test_build_run_specs_merges_fixed_params() -> None:
    sweep = SweepConfig(
        name="s",
        grid={"epochs": [1]},
        fixed={"learning_rate_multiplier": 2.0},
    )
    (spec,) = build_run_specs(sweep)
    assert spec.params == {"epochs": 1, "learning_rate_multiplier": 2.0}


def test_build_run_specs_grid_overrides_fixed() -> None:
    sweep = SweepConfig(name="s", grid={"epochs": [3]}, fixed={"epochs": 1})
    (spec,) = build_run_specs(sweep)
    assert spec.params["epochs"] == 3


# --- select_best_run -----------------------------------------------------------


def test_select_best_run_picks_max_metric() -> None:
    results = {"a": {"accuracy": 0.7}, "b": {"accuracy": 0.9}}
    assert select_best_run(results) == "b"


def test_select_best_run_tie_breaks_by_name() -> None:
    results = {"b": {"accuracy": 0.9}, "a": {"accuracy": 0.9}}
    assert select_best_run(results) == "a"


def test_select_best_run_custom_metric() -> None:
    results = {"a": {"macro_f1": 0.8}, "b": {"macro_f1": 0.6}}
    assert select_best_run(results, metric="macro_f1") == "a"


def test_select_best_run_empty_raises() -> None:
    with pytest.raises(ValueError, match="No .* results"):
        select_best_run({})


# --- aggregate_results ---------------------------------------------------------


def _result(name: str, params: dict, metrics: dict) -> RunResult:
    spec = RunSpec(
        name=name,
        display_name=f"geap-doe-x-{name}",
        base_model="gemini-2.5-flash-lite",
        params=params,
    )
    return RunResult(spec=spec, job_name=f"jobs/{name}", endpoint=f"ep/{name}", metrics=metrics, reused=False)


def test_aggregate_results_flattens_params_and_metrics() -> None:
    results = [
        _result("r1", {"epochs": 1}, {"accuracy": 0.8, "macro_f1": 0.7, "report": {}}),
        _result("r2", {"epochs": 2}, {"accuracy": 0.9, "macro_f1": 0.85, "report": {}}),
    ]
    rows = aggregate_results(results)
    assert rows[0] == {
        "run": "r1",
        "base_model": "gemini-2.5-flash-lite",
        "epochs": 1,
        "accuracy": 0.8,
        "macro_f1": 0.7,
    }
    assert [r["run"] for r in rows] == ["r1", "r2"]  # order preserved


def test_aggregate_results_skips_absent_metrics() -> None:
    (row,) = aggregate_results([_result("r", {}, {"accuracy": 0.5})])
    assert "macro_f1" not in row
    assert row["accuracy"] == 0.5


# --- run_sweep (driver) --------------------------------------------------------


@pytest.fixture
def no_tracking(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace the Experiments logging seams so tests never touch aiplatform."""
    track = MagicMock()
    track.return_value.__enter__.return_value = SimpleNamespace(name="run")
    log = MagicMock()
    monkeypatch.setattr("geap_tuning.doe.track_run", track)
    monkeypatch.setattr("geap_tuning.doe.log_summary_metrics", log)
    return {"track": track, "log": log}


def _job(endpoint: str = "ep/x") -> SimpleNamespace:
    return SimpleNamespace(name="jobs/x", tuned_model=SimpleNamespace(endpoint=endpoint, model=None))


def test_run_sweep_reuses_existing_job(no_tracking: dict[str, MagicMock]) -> None:
    client = MagicMock()
    launch = MagicMock()
    results = run_sweep(
        client,
        SweepConfig(name="s", grid={"epochs": [1]}),
        train_uri="gs://b/train.jsonl",
        evaluate_fn=lambda _ep: {"accuracy": 0.9},
        launch_fn=launch,
        wait_fn=lambda _c, _n: _job(),
        find_fn=lambda _c, _dn, **_k: _job(),
    )
    launch.assert_not_called()
    assert results[0].reused is True
    assert results[0].metrics == {"accuracy": 0.9}


def test_run_sweep_launches_when_absent_and_passes_params(no_tracking: dict[str, MagicMock]) -> None:
    client = MagicMock()
    seen: dict[str, object] = {}

    def fake_launch(_client: object, spec: RunSpec, _train: str, _val: str | None) -> SimpleNamespace:
        seen["params"] = spec.params
        return _job()

    results = run_sweep(
        client,
        SweepConfig(name="s", grid={"epochs": [2], "adapter_size": [8]}),
        train_uri="gs://b/train.jsonl",
        evaluate_fn=lambda _ep: {"accuracy": 0.5},
        launch_fn=fake_launch,
        wait_fn=lambda _c, _n: _job(),
        find_fn=lambda _c, _dn, **_k: None,
    )
    assert seen["params"] == {"epochs": 2, "adapter_size": 8}
    assert results[0].reused is False


def test_run_sweep_default_launcher_calls_tune_with_params(no_tracking: dict[str, MagicMock]) -> None:
    client = MagicMock()
    run_sweep(
        client,
        SweepConfig(name="s", base_model="gemini-2.5-flash-lite", grid={"epochs": [2], "adapter_size": [8]}),
        train_uri="gs://b/train.jsonl",
        val_uri="gs://b/val.jsonl",
        evaluate_fn=lambda _ep: {"accuracy": 0.5},
        wait_fn=lambda _c, _n: _job(),
        find_fn=lambda _c, _dn, **_k: None,
    )
    cfg = client.tunings.tune.call_args.kwargs["config"]
    assert cfg.epoch_count == 2
    assert cfg.adapter_size == "ADAPTER_SIZE_EIGHT"


def test_run_sweep_logs_to_experiments_when_named(no_tracking: dict[str, MagicMock]) -> None:
    client = MagicMock()
    run_sweep(
        client,
        SweepConfig(name="s", grid={"epochs": [1]}),
        train_uri="gs://b/train.jsonl",
        evaluate_fn=lambda _ep: {"accuracy": 0.9, "macro_f1": 0.8, "report": {}},
        wait_fn=lambda _c, _n: _job(),
        find_fn=lambda _c, _dn, **_k: _job(),
        experiment="exp",
    )
    no_tracking["track"].assert_called_once()
    logged = no_tracking["log"].call_args.args[0]
    assert logged == {"accuracy": 0.9, "macro_f1": 0.8}  # non-numeric "report" dropped


def test_run_sweep_skips_experiments_when_none(no_tracking: dict[str, MagicMock]) -> None:
    client = MagicMock()
    run_sweep(
        client,
        SweepConfig(name="s", grid={"epochs": [1]}),
        train_uri="gs://b/train.jsonl",
        evaluate_fn=lambda _ep: {"accuracy": 0.9},
        wait_fn=lambda _c, _n: _job(),
        find_fn=lambda _c, _dn, **_k: _job(),
    )
    no_tracking["track"].assert_not_called()
    no_tracking["log"].assert_not_called()


# --- collect_checkpoint_curve --------------------------------------------------


def test_collect_checkpoint_curve_orders_by_epoch() -> None:
    checkpoints = [
        SimpleNamespace(checkpoint_id="c2", epoch=2, step=20, endpoint="ep/c2"),
        SimpleNamespace(checkpoint_id="c1", epoch=1, step=10, endpoint="ep/c1"),
    ]
    job = SimpleNamespace(tuned_model=SimpleNamespace(checkpoints=checkpoints))
    scores = {"ep/c1": {"accuracy": 0.6}, "ep/c2": {"accuracy": 0.9}}
    curve = collect_checkpoint_curve(job, lambda ep: scores[ep], metric="accuracy")
    assert curve == [(1, 0.6), (2, 0.9)]
