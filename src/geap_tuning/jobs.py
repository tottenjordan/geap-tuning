"""Tuning-job monitoring, idempotency, and endpoint resolution.

Shared by every tuning service: poll a job to completion, reuse an existing job
with the same display name (cost control), and pull the tuned model's endpoint
resource name for inference. The Gen AI SDK job object exposes ``state``,
``name``, ``tuned_model_display_name``, and
``tuned_model.{endpoint,model,checkpoints}`` — the last enables the checkpoint
and continuous-tuning helpers below (see
``docs/notes/checkpoints-and-continuous-tuning.md``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from google.genai import types

if TYPE_CHECKING:
    from collections.abc import Sequence

# Job lifecycle states (Gen AI SDK / GEAP tuningJobs).
STATE_SUCCEEDED = "JOB_STATE_SUCCEEDED"
STATE_FAILED = "JOB_STATE_FAILED"
STATE_CANCELLED = "JOB_STATE_CANCELLED"
_TERMINAL_STATES = frozenset({STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELLED})
_REUSABLE_STATES = ("JOB_STATE_SUCCEEDED", "JOB_STATE_RUNNING", "JOB_STATE_PENDING")


def tuned_endpoint(job: Any) -> str:  # noqa: ANN401 - SDK job type is dynamic
    """Return the tuned model's endpoint, falling back to its model resource name."""
    endpoint = job.tuned_model.endpoint or job.tuned_model.model
    if not endpoint:
        msg = "Job has no tuned endpoint yet (not finished successfully?)"
        raise ValueError(msg)
    return endpoint


def find_tuning_job_by_display_name(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    display_name: str,
    *,
    states: Sequence[str] = _REUSABLE_STATES,
) -> Any | None:  # noqa: ANN401 - returns the SDK job object or None
    """Return the first existing job matching ``display_name`` in a reusable state.

    Lets an example reuse a prior run instead of launching (and paying for) a
    duplicate tuning job. Returns ``None`` when there is no match.
    """
    for job in client.tunings.list():
        if job.tuned_model_display_name == display_name and job.state in states:
            return job
    return None


def list_checkpoints(job: Any) -> list[Any]:  # noqa: ANN401 - SDK job type is dynamic
    """Return the job's intermediate checkpoints (empty when none were exported).

    Populated only when the job ran with ``export_last_checkpoint_only=False``
    (the default). Each checkpoint exposes ``checkpoint_id``, ``epoch``,
    ``step``, and its own ``endpoint`` for per-checkpoint inference.
    """
    return list(getattr(job.tuned_model, "checkpoints", None) or [])


def checkpoint_endpoint(job: Any, checkpoint_id: str) -> str:  # noqa: ANN401 - SDK job type
    """Return the inference endpoint for a specific checkpoint of ``job``."""
    for checkpoint in list_checkpoints(job):
        if checkpoint.checkpoint_id == checkpoint_id:
            if not checkpoint.endpoint:
                msg = f"Checkpoint {checkpoint_id} has no endpoint yet"
                raise ValueError(msg)
            return checkpoint.endpoint
    msg = f"No checkpoint {checkpoint_id!r} on this job"
    raise ValueError(msg)


def tuned_model_name(job: Any) -> str:  # noqa: ANN401 - SDK job type is dynamic
    """Return the tuned model's resource name (``projects/.../models/id@ver``).

    This is the value passed as ``base_model`` to continue-tune from this model
    (the SDK auto-detects the ``projects/`` prefix as a pre-tuned model).
    """
    model = job.tuned_model.model
    if not model:
        msg = "Job has no tuned model name yet (not finished successfully?)"
        raise ValueError(msg)
    return model


def get_default_checkpoint_id(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    job: Any,  # noqa: ANN401 - SDK job type is dynamic
) -> str | None:
    """Return the tuned model's current default checkpoint id (serves inference)."""
    model = client.models.get(model=tuned_model_name(job))
    return model.default_checkpoint_id


def set_default_checkpoint(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    job: Any,  # noqa: ANN401 - SDK job type is dynamic
    checkpoint_id: str,
) -> Any:  # noqa: ANN401 - returns the SDK model object
    """Point the tuned model's default endpoint at ``checkpoint_id``."""
    return client.models.update(
        model=tuned_model_name(job),
        config=types.UpdateModelConfig(default_checkpoint_id=checkpoint_id),
    )


def wait_for_tuning_job(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    job_name: str,
    *,
    poll_interval: float = 60.0,
) -> Any:  # noqa: ANN401 - returns the SDK job object
    """Poll ``job_name`` until it reaches a terminal state; raise if it failed."""
    while True:
        job = client.tunings.get(name=job_name)
        if job.state in _TERMINAL_STATES:
            if job.state != STATE_SUCCEEDED:
                msg = f"Tuning job {job_name} ended in {job.state}"
                raise RuntimeError(msg)
            return job
        time.sleep(poll_interval)
