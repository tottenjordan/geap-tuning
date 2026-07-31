"""Launch a preference-tuning (DPO) job on GEAP via the Gen AI SDK.

DPO builds directly on the SFT surface: the *same* ``client.tunings.tune(...)``
call, with ``method="PREFERENCE_TUNING"`` and a ``beta`` coefficient added to the
config. Training data pairs a preferred and a dispreferred completion per prompt
(``schemas.preference_example``); the model learns to widen the likelihood gap
between them.

Best practice (per the GEAP docs) is to SFT on the preferred responses first and
then *continuous-tune* from that checkpoint with DPO. This launcher runs DPO
directly on the base model to keep the example self-contained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.genai import types

from geap_tuning.sft.tune import ADAPTER_MAP

if TYPE_CHECKING:
    from google import genai


def launch_preference_job(  # noqa: PLR0913 - explicit tuning hyperparameters, all keyword-only
    client: genai.Client,
    *,
    train_uri: str,
    display_name: str,
    val_uri: str | None = None,
    base_model: str = "gemini-2.5-flash",
    epochs: int = 2,
    adapter_size: int = 8,
    learning_rate_multiplier: float = 1.0,
    beta: float = 0.1,
    export_last_checkpoint_only: bool = False,
    evaluation_config: types.EvaluationConfig | None = None,
    pre_tuned_model_checkpoint_id: str | None = None,
    labels: dict[str, str] | None = None,
) -> Any:  # noqa: ANN401 - SDK returns a dynamically-typed TuningJob
    """Submit a DPO job and return the created tuning job.

    ``train_uri``/``val_uri`` are ``gs://`` URIs to preference-format JSONL
    (``completions`` with binary ``score``). ``beta`` (recommended 0.01-0.5)
    controls how closely the tuned model stays to the base; lower means more
    aggressive updates toward the preferred response. ``adapter_size`` must be a
    key of :data:`geap_tuning.sft.tune.ADAPTER_MAP` (raises ``KeyError``
    otherwise). This does not wait — poll with
    :func:`geap_tuning.jobs.wait_for_tuning_job`.

    ``export_last_checkpoint_only``, ``evaluation_config`` and
    ``pre_tuned_model_checkpoint_id`` behave as in :func:`launch_sft_job` — the
    documented DPO best practice is to SFT first, then continuous-tune from that
    SFT model by passing its resource name as ``base_model`` (see
    ``docs/notes/checkpoints-and-continuous-tuning.md``). Like SFT, DPO has **no**
    ``evaluate_interval`` — that field is RLFT-only in google-genai 2.14.0 (it
    serializes under the reinforcement spec, so passing it here 400s).

    ``labels`` behaves as in :func:`launch_sft_job`: resource labels on the job
    that propagate to the generated Model/Endpoint; ``None`` omits the field.
    See ``docs/notes/resource-labels.md``.
    """
    config = types.CreateTuningJobConfig(
        method="PREFERENCE_TUNING",
        tuned_model_display_name=display_name,
        epoch_count=epochs,
        adapter_size=ADAPTER_MAP[adapter_size],
        learning_rate_multiplier=learning_rate_multiplier,
        beta=beta,
        validation_dataset=types.TuningDataset(gcs_uri=val_uri) if val_uri else None,
        export_last_checkpoint_only=export_last_checkpoint_only,
        evaluation_config=evaluation_config,
        pre_tuned_model_checkpoint_id=pre_tuned_model_checkpoint_id,
        labels=labels,
    )
    return client.tunings.tune(
        base_model=base_model,
        training_dataset=types.TuningDataset(gcs_uri=train_uri),
        config=config,
    )
