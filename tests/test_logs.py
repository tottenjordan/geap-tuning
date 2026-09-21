"""Tests for the structured-logging setup.

The library must stay silent until an application opts in, and both formatters
must carry the ``extra=`` fields, not just the rendered message.
"""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest  # noqa: TC002 - fixture type used at runtime by pytest

from geap_tuning.logs import (
    LOG_FORMAT_ENV,
    LOG_LEVEL_ENV,
    PACKAGE_LOGGER,
    configure_logging,
    get_logger,
)


def test_library_is_silent_until_configured() -> None:
    """The library-logging contract: importing us must not print anything."""
    handlers = logging.getLogger(PACKAGE_LOGGER).handlers
    assert handlers, "package logger should carry a NullHandler"
    assert all(isinstance(h, logging.NullHandler) for h in handlers)


def test_text_format_appends_structured_fields() -> None:
    buffer = io.StringIO()
    configure_logging(fmt="text", stream=buffer)
    get_logger("geap_tuning.x").info("tuning job state", extra={"job": "123", "state": "RUNNING"})
    line = buffer.getvalue().strip()
    assert "tuning job state" in line
    assert "job=123" in line
    assert "state=RUNNING" in line
    assert "INFO" in line


def test_json_format_emits_one_object_per_line() -> None:
    buffer = io.StringIO()
    configure_logging(fmt="json", stream=buffer)
    log = get_logger("geap_tuning.x")
    log.info("first", extra={"run": "a"})
    log.info("second", extra={"run": "b"})

    lines = buffer.getvalue().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["message"] == "first"
    assert first["run"] == "a"
    assert first["level"] == "INFO"
    assert first["logger"] == "geap_tuning.x"
    assert "time" in first


def test_json_format_survives_a_non_serializable_field() -> None:
    # Raising inside logging would swallow the record entirely, so we degrade.
    buffer = io.StringIO()
    configure_logging(fmt="json", stream=buffer)
    get_logger("geap_tuning.x").info("odd", extra={"obj": object()})
    assert "obj" in json.loads(buffer.getvalue().strip())


def test_json_format_includes_the_traceback() -> None:
    buffer = io.StringIO()
    configure_logging(fmt="json", stream=buffer)

    def boom() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    try:
        boom()
    except RuntimeError:
        get_logger("geap_tuning.x").exception("failed")
    assert "RuntimeError" in json.loads(buffer.getvalue().strip())["exception"]


def test_configure_is_idempotent() -> None:
    """A notebook re-running its setup cell must not double every line."""
    buffer = io.StringIO()
    for _ in range(3):
        configure_logging(stream=buffer)
    get_logger("geap_tuning.x").info("once")
    assert buffer.getvalue().count("once") == 1


def test_configure_does_not_disturb_foreign_handlers() -> None:
    """Re-configuring replaces only our own handler, not an app's."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    configure_logging(stream=io.StringIO())
    assert foreign in logger.handlers


def test_level_filters_records() -> None:
    buffer = io.StringIO()
    configure_logging(level="WARNING", stream=buffer)
    log = get_logger("geap_tuning.x")
    log.info("hidden")
    log.warning("shown")
    out = buffer.getvalue()
    assert "hidden" not in out
    assert "shown" in out


def test_env_vars_supply_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "ERROR")
    monkeypatch.setenv(LOG_FORMAT_ENV, "json")
    buffer = io.StringIO()
    configure_logging(stream=buffer)
    log = get_logger("geap_tuning.x")
    log.warning("filtered out by ERROR")
    log.error("kept", extra={"k": "v"})
    lines = [line for line in buffer.getvalue().strip().split("\n") if line]
    assert len(lines) == 1
    assert json.loads(lines[0])["k"] == "v"


def test_default_stream_is_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """A plain ``> run.log`` must capture the progress output.

    Logs on stderr meant the natural stdout-only redirect dropped the heartbeat —
    the very failure it exists to prevent. Applications get stdout by default;
    ``stream=sys.stderr`` restores the conventional split.
    """
    configure_logging()
    get_logger("geap_tuning.x").info("progress", extra={"run": "a"})
    captured = capsys.readouterr()
    assert "progress" in captured.out
    assert "progress" not in captured.err


def test_stream_can_be_overridden_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(stream=sys.stderr)
    get_logger("geap_tuning.x").info("progress")
    captured = capsys.readouterr()
    assert "progress" in captured.err
    assert "progress" not in captured.out
