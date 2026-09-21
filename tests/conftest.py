"""Shared test factories.

Fake SDK objects that the tuning helpers read attributes off. These live here
because the same shapes were being redefined per-module with divergent
signatures — ``_job`` existed in both ``test_jobs.py`` and ``test_doe.py`` with
different defaults, and ``test_doe.py`` hand-built the checkpoint namespace
``test_jobs.py`` already had a helper for.

The defaults describe a *healthy* finished job (it has an endpoint), which is
what most callers want. Tests that exercise the missing-field paths pass the
``None`` explicitly, which reads better there anyway.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from geap_tuning.logs import PACKAGE_LOGGER

if TYPE_CHECKING:
    from collections.abc import Iterator


def make_job(
    *,
    name: str = "jobs/x",
    endpoint: str | None = "ep/x",
    model: str | None = None,
) -> SimpleNamespace:
    """Return a fake tuning job exposing ``name`` and ``tuned_model``."""
    return SimpleNamespace(name=name, tuned_model=SimpleNamespace(endpoint=endpoint, model=model))


def make_checkpoint(
    checkpoint_id: str,
    *,
    epoch: int = 1,
    step: int = 10,
    endpoint: str = "",
) -> SimpleNamespace:
    """Return a fake exported checkpoint.

    ``epoch``/``step`` are parameters because ``collect_checkpoint_curve`` sorts
    on epoch, so its test needs distinct values.
    """
    return SimpleNamespace(
        checkpoint_id=checkpoint_id,
        epoch=epoch,
        step=step,
        endpoint=endpoint,
    )


@pytest.fixture(autouse=True)
def _reset_package_logger() -> Iterator[None]:
    """Undo any ``configure_logging`` a test performs.

    ``configure_logging`` sets ``propagate = False`` on the package logger, which
    would stop ``caplog`` seeing records in every test that ran afterwards.
    Restoring the logger keeps tests order-independent.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate
