"""Launch a reinforcement-tuning (RLFT) job on GEAP via the Gen AI SDK.

RLFT builds on the same ``client.tunings.tune(...)`` call as SFT/DPO, with
``method="REINFORCEMENT_TUNING"`` and a ``reward_config`` on the config. There is
no gold completion — each generation is scored by a reward function over the
record's ``references``. This launcher ships the tested :mod:`geap_tuning.rlft.reward`
module verbatim as a code-execution scorer (its ``evaluate(example, response)``
runs in the GEAP sandbox). Best practice (documented): SFT first, then
continuous-tune with RLFT; this launcher tunes the base model directly for a
self-contained demo.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from google.genai import types

from geap_tuning.rlft import reward
from geap_tuning.sft.tune import ADAPTER_MAP

if TYPE_CHECKING:
    from google import genai


def build_reward_config(
    reward_name: str = "math_correctness",
) -> types.SingleReinforcementTuningRewardConfig:
    """Ship :mod:`geap_tuning.rlft.reward` verbatim as a code-execution reward scorer.

    ``inspect.getsource`` pulls the exact, unit-tested module source so the reward
    that runs in the sandbox is the same one the tests exercise — one function,
    two uses (training reward + offline eval).
    """
    return types.SingleReinforcementTuningRewardConfig(
        reward_name=reward_name,
        code_execution_reward_scorer=types.ReinforcementTuningCodeExecutionRewardScorer(
            python_code_snippet=inspect.getsource(reward),
        ),
    )


def launch_rlft_job(  # noqa: PLR0913 - explicit tuning hyperparameters, all keyword-only
    client: genai.Client,
    *,
    train_uri: str,
    display_name: str,
    val_uri: str | None = None,
    base_model: str = "gemini-2.5-flash",
    epochs: int = 5,
    adapter_size: int = 16,
    learning_rate_multiplier: float = 1.0,
    samples_per_prompt: int = 8,
    reward_config: types.SingleReinforcementTuningRewardConfig | None = None,
) -> Any:  # noqa: ANN401 - SDK returns a dynamically-typed TuningJob
    """Submit an RLFT job and return the created tuning job.

    ``train_uri``/``val_uri`` are ``gs://`` URIs to RLFT-format JSONL (``contents``
    + ``references``, no completion). ``samples_per_prompt`` is how many candidate
    generations are drawn per prompt for reward comparison. ``adapter_size`` must be
    a key of :data:`geap_tuning.sft.tune.ADAPTER_MAP` (raises ``KeyError``
    otherwise). NOTE: the RLFT docs call out ``gemini-3.5-flash`` as the supported
    base model — verify availability in your region before running live. Does not
    wait — poll with :func:`geap_tuning.jobs.wait_for_tuning_job`.
    """
    config = types.CreateTuningJobConfig(
        method="REINFORCEMENT_TUNING",
        tuned_model_display_name=display_name,
        epoch_count=epochs,
        adapter_size=ADAPTER_MAP[adapter_size],
        learning_rate_multiplier=learning_rate_multiplier,
        samples_per_prompt=samples_per_prompt,
        reward_config=reward_config or build_reward_config(),
        validation_dataset=types.TuningDataset(gcs_uri=val_uri) if val_uri else None,
    )
    return client.tunings.tune(
        base_model=base_model,
        training_dataset=types.TuningDataset(gcs_uri=train_uri),
        config=config,
    )


def validate_reward_config(  # noqa: PLR0913 - explicit preflight inputs, all keyword-only
    client: genai.Client,
    *,
    project: str,
    location: str,
    sample_answer: str,
    example_record: dict[str, Any],
    reward_config: types.SingleReinforcementTuningRewardConfig | None = None,
) -> Any:  # noqa: ANN401 - SDK returns a dynamically-typed ValidateRewardResponse
    """Preflight the reward on one example via ``tunings.validate_reward``.

    Returns the SDK ``ValidateRewardResponse`` (carrying ``overall_reward`` /
    ``error``); a non-null error or ``NaN`` means the reward is broken. Optional
    but recommended before launching — RLFT auto-stops if >80% of reward calls
    fail. ``example_record`` is an RLFT record (``contents`` + ``references``).
    """
    return client.tunings.validate_reward(
        parent=f"projects/{project}/locations/{location}",
        sample_response=types.Content(role="model", parts=[types.Part(text=sample_answer)]),
        example=types.ReinforcementTuningExample(
            contents=example_record["contents"],
            references=example_record["references"],
        ),
        single_reward_config=reward_config or build_reward_config(),
    )
