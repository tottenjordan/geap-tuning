"""Tuning-job monitoring, idempotency, and endpoint resolution.

Shared by every tuning service: poll a job to completion, reuse an existing job
with the same display name (cost control), and pull the tuned model's endpoint
resource name for inference. The Gen AI SDK job object exposes ``state``,
``name``, ``tuned_model_display_name``, and ``tuned_model.{endpoint,model}``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

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
