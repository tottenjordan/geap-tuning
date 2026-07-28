"""Smoke test confirming the package imports and the toolchain runs."""

import pytest

from geap_tuning import main


def test_main_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert "geap-tuning" in capsys.readouterr().out
