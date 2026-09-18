"""Tests for the ``geap-tuning`` console script.

It does no tuning; a bare invocation confirms the install and reports whether
``.env`` resolves, which is the most common first-run failure.
"""

from unittest.mock import MagicMock

import pytest

from geap_tuning import main


def test_main_reports_version_and_resolved_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = MagicMock(project="proj-x", location="us-central1", bucket="gs://b")
    monkeypatch.setattr("geap_tuning.config.load_config", lambda: cfg)

    main()

    out = capsys.readouterr().out
    assert "geap-tuning" in out
    assert "proj-x" in out
    assert "us-central1" in out
    assert "examples/run_sft.py" in out


def test_main_explains_an_unresolved_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom() -> None:
        msg = "No project set; expected one of ('PROJECT_ID', 'GOOGLE_CLOUD_PROJECT')"
        raise ValueError(msg)

    monkeypatch.setattr("geap_tuning.config.load_config", boom)

    main()  # must not raise: an unconfigured install should still report cleanly

    out = capsys.readouterr().out
    assert "NOT resolved" in out
    assert ".env.example" in out
