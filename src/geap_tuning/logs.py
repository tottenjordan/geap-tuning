"""Structured logging for the ``geap_tuning`` library.

Tuning runs are long (30-60 minutes per job, hours per sweep) and usually watched
through a redirected log, so the operational messages the library emits — job
state, reuse decisions, sweep progress — need timestamps, levels and machine-
readable fields rather than bare ``print``.

Two rules keep this from being intrusive:

- **The library never configures logging.** Importing :mod:`geap_tuning` attaches
  a :class:`~logging.NullHandler` to the package logger and nothing else, so a
  consumer that imports us sees no output unless they ask for it. This is the
  standard library-logging contract.
- **Applications opt in** with :func:`configure_logging`. Every example calls it
  once at the top of ``main()``; a notebook or a downstream app can call it too,
  or install its own handlers and ignore ours entirely.

Records carry structured fields via ``extra=``, which both formatters render:

    logger.info("tuning job state", extra={"job": "123", "state": "RUNNING"})

    text: 2026-09-21 19:04:11 INFO  geap_tuning.jobs: tuning job state job=123 state=RUNNING
    json: {"time": "...", "level": "INFO", "logger": "geap_tuning.jobs",
           "message": "tuning job state", "job": "123", "state": "RUNNING"}

Set ``GEAP_LOG_LEVEL`` (default ``INFO``) and ``GEAP_LOG_FORMAT`` (``text``, the
default, or ``json``) to control it without touching code.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

PACKAGE_LOGGER = "geap_tuning"
LOG_LEVEL_ENV = "GEAP_LOG_LEVEL"
LOG_FORMAT_ENV = "GEAP_LOG_FORMAT"

_TEXT_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Attributes every LogRecord carries. Anything else was passed via ``extra=`` and
# is therefore one of our structured fields.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

# Attach once at import: the library emits nothing until an application opts in.
logging.getLogger(PACKAGE_LOGGER).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the logger a library module should use.

    Pass ``__name__`` so records are namespaced under ``geap_tuning.*`` and a
    consumer can silence or re-route the whole package with one call.
    """
    return logging.getLogger(name)


def _extras(record: logging.LogRecord) -> Iterator[tuple[str, Any]]:
    """Yield the structured fields a caller attached via ``extra=``."""
    for key, value in record.__dict__.items():
        if key not in _RESERVED and not key.startswith("_"):
            yield key, value


class _ManagedHandler(logging.StreamHandler):
    """Marker subclass so :func:`configure_logging` can replace only its own handler."""


class StructuredFormatter(logging.Formatter):
    """Human-readable formatter that appends ``key=value`` for structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as text with its ``extra=`` fields appended."""
        base = super().format(record)
        fields = " ".join(f"{key}={value}" for key, value in _extras(record))
        return f"{base} {fields}" if fields else base


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with ``extra=`` fields promoted to top level."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a single-line JSON object."""
        payload: dict[str, Any] = {
            "time": self.formatTime(record, _TIME_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(dict(_extras(record)))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str so a stray non-serializable value degrades instead of raising
        # inside logging, which would swallow the message entirely.
        return json.dumps(payload, default=str)


def configure_logging(
    *,
    level: int | str | None = None,
    fmt: str | None = None,
    stream: Any = None,  # noqa: ANN401 - any text stream, defaults to stderr
) -> logging.Logger:
    """Install one handler on the package logger and return it. Idempotent.

    Call this once from an application (every example does, at the top of
    ``main()``). Re-calling replaces our handler rather than stacking a second
    one, so a notebook re-running its setup cell does not double every line.

    ``level`` and ``fmt`` default to ``GEAP_LOG_LEVEL`` / ``GEAP_LOG_FORMAT``,
    themselves defaulting to ``INFO`` and ``text``. Pass ``fmt="json"`` for
    one JSON object per line.

    Logs go to **stderr** by default so a script's own ``print`` output on stdout
    stays separately redirectable.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    resolved_level = level if level is not None else os.environ.get(LOG_LEVEL_ENV, "INFO")
    resolved_fmt = (fmt if fmt is not None else os.environ.get(LOG_FORMAT_ENV, "text")).lower()

    for existing in [h for h in logger.handlers if isinstance(h, _ManagedHandler)]:
        logger.removeHandler(existing)

    handler = _ManagedHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if resolved_fmt == "json"
        else StructuredFormatter(_TEXT_FORMAT, _TIME_FORMAT)
    )
    logger.addHandler(handler)
    logger.setLevel(resolved_level)
    # Our handler already emits; propagating would duplicate under an app that
    # configures the root logger too.
    logger.propagate = False
    return logger
