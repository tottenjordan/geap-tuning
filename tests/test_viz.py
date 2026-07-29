"""Tests for the multi-run visualization helpers.

matplotlib/pandas live in the optional ``viz`` group, so the module lazy-imports
them behind ``_import_matplotlib``/``_import_pandas``. Tests monkeypatch those
shims with a ``MagicMock`` so **no real rendering runs** and the suite passes
whether or not the ``viz`` group is installed; they also verify the shims raise
an actionable error when the dep is missing.
"""

from unittest.mock import MagicMock

import pytest

from geap_tuning import viz
from geap_tuning.viz import _import_matplotlib, _import_pandas

ROWS = [
    {"run": "a", "epochs": 1, "accuracy": 0.7, "macro_f1": 0.6},
    {"run": "b", "epochs": 2, "accuracy": 0.9, "macro_f1": 0.85},
]


@pytest.fixture
def fake_plt(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    plt = MagicMock()
    monkeypatch.setattr(viz, "_import_matplotlib", lambda: plt)
    return plt


@pytest.fixture
def fake_pandas(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    pd = MagicMock()
    monkeypatch.setattr(viz, "_import_pandas", lambda: pd)
    return pd


# --- lazy-import shims ---------------------------------------------------------


def test_import_matplotlib_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_name: str) -> object:
        raise ImportError

    monkeypatch.setattr(viz.importlib, "import_module", boom)
    with pytest.raises(RuntimeError, match="uv sync --group viz"):
        _import_matplotlib()


def test_import_pandas_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_name: str) -> object:
        raise ImportError

    monkeypatch.setattr(viz.importlib, "import_module", boom)
    with pytest.raises(RuntimeError, match="uv sync --group viz"):
        _import_pandas()


# --- dataframe bridges ---------------------------------------------------------


def test_rows_to_dataframe_uses_pandas(fake_pandas: MagicMock) -> None:
    out = viz.rows_to_dataframe(ROWS)
    assert fake_pandas.DataFrame.call_args.args[0] == ROWS
    assert out is fake_pandas.DataFrame.return_value


def test_dataframe_to_rows_via_to_dict() -> None:
    frame = MagicMock()
    frame.to_dict.return_value = ROWS
    assert viz.dataframe_to_rows(frame) == ROWS
    assert frame.to_dict.call_args.kwargs["orient"] == "records"


def test_normalize_experiment_rows_strips_prefixes() -> None:
    raw = [
        {"run_name": "r1", "metric.accuracy": 0.8, "param.epochs": 1, "extra": 9},
        {"run_name": "r2", "metric.accuracy": 0.9, "param.epochs": 2, "extra": 8},
    ]
    rows = viz.normalize_experiment_rows(raw)
    assert rows[0] == {"run": "r1", "accuracy": 0.8, "epochs": 1, "extra": 9}
    assert rows[1]["run"] == "r2"


# --- plots (rendering mocked) --------------------------------------------------


def test_plot_metric_bars_returns_figure(fake_plt: MagicMock) -> None:
    fig, ax = MagicMock(name="fig"), MagicMock(name="ax")
    fake_plt.subplots.return_value = (fig, ax)
    result = viz.plot_metric_bars(ROWS, metric="accuracy")
    assert result is fig
    labels, values = ax.bar.call_args.args
    assert list(labels) == ["a", "b"]
    assert list(values) == [0.7, 0.9]


def test_plot_grouped_metric_bars_plots_each_metric(fake_plt: MagicMock) -> None:
    fig, ax = MagicMock(name="fig"), MagicMock(name="ax")
    fake_plt.subplots.return_value = (fig, ax)
    result = viz.plot_grouped_metric_bars(ROWS, metrics=("accuracy", "macro_f1"))
    assert result is fig
    assert ax.bar.call_count == 2  # one bar group per metric


def test_plot_curves_one_line_per_run(fake_plt: MagicMock) -> None:
    fig, ax = MagicMock(name="fig"), MagicMock(name="ax")
    fake_plt.subplots.return_value = (fig, ax)
    series = {"a": [(1, 0.6), (2, 0.8)], "b": [(1, 0.5), (2, 0.9)]}
    result = viz.plot_curves(series, metric="accuracy")
    assert result is fig
    assert ax.plot.call_count == 2  # one line per run
