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

import datetime
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

# Tuning takes 30-60 minutes; 4h is generous headroom while still bounding a job
# that is wedged in a non-terminal state.
_DEFAULT_TIMEOUT = 4 * 60 * 60

# Resource label carrying the fingerprint of the dataset a job trained on, so
# reuse can compare content and not just the (always identical) staging URI.
DATA_FINGERPRINT_LABEL = "data_fingerprint"

# Where each tuning method records the dataset it trained from.
_TUNING_SPECS = (
    "supervised_tuning_spec",
    "preference_optimization_spec",
    "reinforcement_tuning_spec",
)
# Sort floor for jobs whose create_time the SDK did not populate, so they lose
# the recency tiebreak rather than crashing the comparison.
_EPOCH = datetime.datetime.fromtimestamp(0, tz=datetime.UTC)
# Print a heartbeat on every state change, and otherwise once every N polls, so a
# 60-minute wait yields a readable trail rather than 60 identical lines.
_HEARTBEAT_EVERY = 5


def _print_heartbeat(job_name: str, state: str, elapsed: float) -> None:
    """Print one timestamped ``state``/``elapsed`` line for a long-running job."""
    stamp = time.strftime("%H:%M:%S")
    short = job_name.rsplit("/", 1)[-1]
    print(f"[{stamp}] tuning job {short}: {state} ({elapsed / 60:.1f} min elapsed)")


def tuned_endpoint(job: Any) -> str:  # noqa: ANN401 - SDK job type is dynamic
    """Return the tuned model's endpoint, falling back to its model resource name."""
    endpoint = job.tuned_model.endpoint or job.tuned_model.model
    if not endpoint:
        msg = "Job has no tuned endpoint yet (not finished successfully?)"
        raise ValueError(msg)
    return endpoint


def job_training_uri(job: Any) -> str | None:  # noqa: ANN401 - SDK job type is dynamic
    """Return the training dataset URI a job was launched with, across all methods.

    SFT, DPO and RLFT each record it under their own spec
    (``supervised_tuning_spec`` / ``preference_optimization_spec`` /
    ``reinforcement_tuning_spec``); returns ``None`` if none is populated.
    """
    for spec_name in _TUNING_SPECS:
        spec = getattr(job, spec_name, None)
        uri = getattr(spec, "training_dataset_uri", None) if spec else None
        if uri:
            return uri
    return None


def with_data_fingerprint(
    labels: dict[str, str] | None,
    fingerprint: str | None,
) -> dict[str, str] | None:
    """Return ``labels`` plus the dataset fingerprint, so reuse can compare content.

    Pair with ``data_fingerprint=`` on :func:`find_tuning_job_by_display_name`: the
    launcher records the fingerprint here, the lookup checks it later. Returns
    ``labels`` untouched when the fingerprint is unknown (e.g. Cloud Storage was
    unreachable), keeping this a safe drop-in.
    """
    if not fingerprint:
        return labels
    return {**(labels or {}), DATA_FINGERPRINT_LABEL: fingerprint}


def _fingerprint_allows_reuse(job: Any, expected: str) -> bool:  # noqa: ANN401 - SDK job
    """Return whether ``job`` may be reused for a dataset fingerprinted ``expected``."""
    recorded = (getattr(job, "labels", None) or {}).get(DATA_FINGERPRINT_LABEL)
    if not recorded:
        # Launched before fingerprints were recorded: unknowable, so allow reuse but
        # say so — silence here is exactly the failure mode this check exists for.
        print(
            f"note: reusing {getattr(job, 'name', '?')} which records no dataset "
            f"fingerprint; if you have edited the data since, change the display name"
        )
        return True
    return recorded == expected


def find_tuning_job_by_display_name(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    display_name: str,
    *,
    states: Sequence[str] = _REUSABLE_STATES,
    train_uri: str | None = None,
    data_fingerprint: str | None = None,
) -> Any | None:  # noqa: ANN401 - returns the SDK job object or None
    """Return the **most recent** reusable job matching ``display_name``, or ``None``.

    Lets an example reuse a prior run instead of launching (and paying for) a
    duplicate tuning job.

    Two correctness details beyond a plain name match:

    - **Most recent wins.** Duplicates under one display name are reachable in
      practice, because a FAILED job is not reusable and so triggers a relaunch
      under the same name. Candidates are ordered by ``create_time`` (descending)
      instead of returning whichever the API listed first.
    - **``train_uri`` gates reuse on the dataset path.** When given, a candidate
      trained from a different URI is not reused.
    - **``data_fingerprint`` gates reuse on the dataset *content*.** Every driver
      stages to a fixed path, so comparing URIs alone cannot tell an edited dataset
      from the original. Pass the content fingerprint (see
      :func:`geap_tuning.gcs.object_fingerprint`) and a job trained on different
      bytes is not reused, even at the same URI. It travels on the job as the
      :data:`DATA_FINGERPRINT_LABEL` resource label.

    A candidate carrying **no** fingerprint label — one launched before this was
    recorded — is still reused, because its data cannot be compared. A note is
    printed so that ambiguity is visible rather than silent.
    """
    matches = [
        job
        for job in client.tunings.list()
        if job.tuned_model_display_name == display_name and job.state in states
    ]
    if train_uri is not None:
        matches = [job for job in matches if (job_training_uri(job) or train_uri) == train_uri]
    if data_fingerprint is not None:
        matches = [job for job in matches if _fingerprint_allows_reuse(job, data_fingerprint)]
    if not matches:
        return None
    return max(matches, key=lambda job: getattr(job, "create_time", None) or _EPOCH)


def cancel_tuning_job(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    name: str,
) -> Any:  # noqa: ANN401 - returns the SDK cancel response
    """Request cancellation of a running/pending tuning job by resource name.

    Only a non-terminal job can be cancelled; the SDK exposes no ``delete`` for
    tuning jobs (see ``docs/notes/endpoints-and-cost.md``).
    """
    return client.tunings.cancel(name=name)


def cancel_tuning_job_by_display_name(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    display_name: str,
) -> str | None:
    """Find a live job by display name and cancel it; return its name or ``None``.

    Uses :func:`find_tuning_job_by_display_name`, whose default states include
    ``JOB_STATE_RUNNING`` / ``JOB_STATE_PENDING``, so an in-flight job is found.
    Returns ``None`` when no cancellable job matches.
    """
    job = find_tuning_job_by_display_name(client, display_name)
    if job is None:
        return None
    cancel_tuning_job(client, job.name)
    return job.name


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
    timeout: float | None = _DEFAULT_TIMEOUT,
    heartbeat: bool = True,
) -> Any:  # noqa: ANN401 - returns the SDK job object
    """Poll ``job_name`` until it reaches a terminal state; raise if it failed.

    Tuning runs for 30-60 minutes, so two things matter beyond the polling itself.
    ``timeout`` (seconds, default 4h) bounds the wait: without it a job wedged in a
    non-terminal state — or in a state string the SDK adds that is not in
    ``_TERMINAL_STATES`` — would poll forever. Pass ``None`` to wait indefinitely.

    ``heartbeat`` prints a timestamped state line on the first poll, on every state
    change, and every ``_HEARTBEAT_EVERY`` polls thereafter, so a long wait leaves a
    readable trail instead of total silence. Transient HTTP failures are handled a
    layer down by the client's retry policy (see ``config._http_options``).
    """
    started = time.monotonic()
    last_state: str | None = None
    polls = 0
    while True:
        job = client.tunings.get(name=job_name)
        elapsed = time.monotonic() - started
        if heartbeat and (job.state != last_state or polls % _HEARTBEAT_EVERY == 0):
            _print_heartbeat(job_name, job.state, elapsed)
        last_state = job.state
        polls += 1

        if job.state in _TERMINAL_STATES:
            if job.state != STATE_SUCCEEDED:
                # Carry the SDK's error detail; without it the caller gets only a
                # state string and has to open the console to learn why it failed.
                detail = getattr(job, "error", None)
                msg = f"Tuning job {job_name} ended in {job.state}"
                if detail:
                    msg = f"{msg}: {detail}"
                raise RuntimeError(msg)
            return job

        if timeout is not None and elapsed >= timeout:
            msg = (
                f"Tuning job {job_name} still {job.state} after {elapsed / 60:.1f} min "
                f"(timeout {timeout / 60:.1f} min); it may still be running — "
                f"check the console or re-run to re-attach by display name"
            )
            raise TimeoutError(msg)

        time.sleep(poll_interval)
