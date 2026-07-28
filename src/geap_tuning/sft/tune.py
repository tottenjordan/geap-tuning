"""Launch a supervised fine-tuning (SFT) job on GEAP via the Gen AI SDK.

SFT trains the base model to reproduce the ``model`` turn of each ``contents``
record. The job is submitted with ``client.tunings.tune(...)`` — the same entry
point every ``tunings``-based GEAP service uses. Preference tuning (DPO) later
reuses this exact call shape, adding ``method="PREFERENCE_TUNING"`` and a
``beta`` hyperparameter to the config; RLFT adds ``method="REINFORCEMENT_TUNING"``
and a ``reward_config`` on the same call (see ``rlft/tune.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.genai import types

if TYPE_CHECKING:
    from google import genai

# Adapter size (LoRA rank) → the SDK enum string. Larger adapters have more
# trainable parameters: more capacity but higher cost and overfitting risk.
ADAPTER_MAP = {
    1: "ADAPTER_SIZE_ONE",
    2: "ADAPTER_SIZE_TWO",
    4: "ADAPTER_SIZE_FOUR",
    8: "ADAPTER_SIZE_EIGHT",
    16: "ADAPTER_SIZE_SIXTEEN",
}


def launch_sft_job(  # noqa: PLR0913 - explicit tuning hyperparameters, all keyword-only
    client: genai.Client,
    *,
    train_uri: str,
    display_name: str,
    val_uri: str | None = None,
    base_model: str = "gemini-2.5-flash",
    epochs: int = 2,
    adapter_size: int = 8,
    learning_rate_multiplier: float = 1.0,
    export_last_checkpoint_only: bool = False,
    evaluation_config: types.EvaluationConfig | None = None,
    pre_tuned_model_checkpoint_id: str | None = None,
) -> Any:  # noqa: ANN401 - SDK returns a dynamically-typed TuningJob
    """Submit an SFT job and return the created tuning job.

    ``train_uri``/``val_uri`` are ``gs://`` URIs to ``contents``-format JSONL.
    ``adapter_size`` must be a key of :data:`ADAPTER_MAP` (raises ``KeyError``
    otherwise). This does not wait for completion — poll with
    :func:`geap_tuning.jobs.wait_for_tuning_job`.

    Cross-cutting features (see
    ``docs/notes/checkpoints-and-continuous-tuning.md``):
    ``export_last_checkpoint_only=False`` (default) keeps intermediate
    checkpoints so you can compare/roll back; ``evaluation_config`` attaches
    auto-eval that runs after each checkpoint (``us-central1`` only). To
    *continuous-tune* from an existing tuned model, pass its resource name
    (``projects/.../models/id@ver``) as ``base_model`` — the SDK auto-detects it
    as a pre-tuned model — and optionally pin a source checkpoint with
    ``pre_tuned_model_checkpoint_id``.
    """
    config = types.CreateTuningJobConfig(
        tuned_model_display_name=display_name,
        epoch_count=epochs,
        adapter_size=ADAPTER_MAP[adapter_size],
        learning_rate_multiplier=learning_rate_multiplier,
        validation_dataset=types.TuningDataset(gcs_uri=val_uri) if val_uri else None,
        export_last_checkpoint_only=export_last_checkpoint_only,
        evaluation_config=evaluation_config,
        pre_tuned_model_checkpoint_id=pre_tuned_model_checkpoint_id,
    )
    return client.tunings.tune(
        base_model=base_model,
        training_dataset=types.TuningDataset(gcs_uri=train_uri),
        config=config,
    )
