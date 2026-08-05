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
    from types import ModuleType

    from google import genai


def build_reward_config(
    reward_name: str = "math_correctness",
    *,
    module: ModuleType = reward,
) -> types.SingleReinforcementTuningRewardConfig:
    """Ship a reward ``module`` verbatim as a code-execution reward scorer.

    ``inspect.getsource`` pulls the exact, unit-tested module source so the reward
    that runs in the sandbox is the same one the tests exercise — one module, two
    uses (training reward + offline eval). Defaults to
    :mod:`geap_tuning.rlft.reward`; pass ``module`` to ship a different scorer
    (e.g. :mod:`geap_tuning.rlft.constraint_reward`). The module must be
    self-contained (stdlib only) and expose ``evaluate(example, response)``.
    """
    return types.SingleReinforcementTuningRewardConfig(
        reward_name=reward_name,
        code_execution_reward_scorer=types.ReinforcementTuningCodeExecutionRewardScorer(
            python_code_snippet=inspect.getsource(module),
        ),
    )


def build_string_match_reward_config(
    *,
    reward_name: str = "answer_format",
    expression: str = r"Answer:\s*-?\d+",
    match_operation: types.MatchOperation = types.MatchOperation.REGEX_CONTAINS,
    correct_answer_reward: float = 1.0,
    wrong_answer_reward: float = -1.0,
) -> types.SingleReinforcementTuningRewardConfig:
    """Build a **declarative** string-match reward scorer (no shipped Python).

    Rewards a generation whose text matches ``expression`` under ``match_operation``
    (``REGEX_CONTAINS`` / ``PARTIAL_MATCH`` / ``EXACT_MATCH``) with
    ``correct_answer_reward``, else ``wrong_answer_reward``. This is a cheap
    **format/keyword** reward — the default checks the response ends with an
    ``Answer: <number>`` line — and needs no per-example ground truth or sandbox
    execution. For per-example matching against a ``references`` key, use the
    scorer's ``json_match_expression``
    (:class:`~google.genai.types.ReinforcementTuningStringMatchRewardScorerJsonMatchExpression`,
    fields ``key_name`` + ``value_string_match_expression``) instead.
    """
    return types.SingleReinforcementTuningRewardConfig(
        reward_name=reward_name,
        string_match_reward_scorer=types.ReinforcementTuningStringMatchRewardScorer(
            correct_answer_reward=correct_answer_reward,
            wrong_answer_reward=wrong_answer_reward,
            string_match_expression=(
                types.ReinforcementTuningStringMatchRewardScorerStringMatchExpression(
                    match_operation=match_operation,
                    expression=expression,
                )
            ),
        ),
    )


_AUTORATER_PROMPT = (
    "You are grading a math tutor's answer for explanation quality.\n"
    "Reward clear, correct step-by-step reasoning that ends with a final answer.\n"
    "Question and response follow. Respond with exactly 'SCORE: 1' for a good "
    "explanation or 'SCORE: 0' for a poor one."
)


def build_autorater_reward_config(
    *,
    reward_name: str = "explanation_quality",
    autorater_prompt: str = _AUTORATER_PROMPT,
    autorater_model: str | None = None,
    sampling_count: int = 4,
) -> types.SingleReinforcementTuningRewardConfig:
    """Build an **LLM-judge** (autorater) reward scorer for subjective quality.

    A judge model runs ``autorater_prompt`` over each generation; its reply is
    parsed with a ``REGEX_EXTRACT`` (``SCORE:\\s*([01])``) and converted to a
    reward via an exact-match on ``"1"`` (reward 1.0) vs. otherwise (0.0).
    ``autorater_model`` is **required by the live API** (there is no service
    default — an empty ``autorater_config`` fails with "Missing autorater model")
    and must be a **fully-qualified publisher resource path**, e.g.
    ``projects/<p>/locations/<l>/publishers/google/models/gemini-2.5-flash``;
    bare names or ``publishers/...`` fragments fail with an opaque "Internal
    error occurred for computing reward". ``sampling_count`` (1-32) sets how many
    judge samples are averaged.

    NOTE: RLFT is Pre-GA — re-verify the parse/scorer enum names
    (:class:`~google.genai.types.ResponseParseType`,
    :class:`~google.genai.types.ReinforcementTuningAutoraterScorerExactMatchScorer`)
    against the installed SDK before a live run.
    """
    return types.SingleReinforcementTuningRewardConfig(
        reward_name=reward_name,
        autorater_scorer=types.ReinforcementTuningAutoraterScorer(
            autorater_config=types.AutoraterConfig(
                sampling_count=sampling_count,
                autorater_model=autorater_model,
            ),
            autorater_prompt=autorater_prompt,
            autorater_response_parse_config=types.ReinforcementTuningParseResponseConfig(
                parse_type=types.ResponseParseType.REGEX_EXTRACT,
                regex_extract_expression=r"SCORE:\s*([01])",
            ),
            exact_match_scorer=types.ReinforcementTuningAutoraterScorerExactMatchScorer(
                correct_answer_reward=1.0,
                wrong_answer_reward=0.0,
                expression="1",
            ),
        ),
    )


def build_cloud_run_reward_config(
    *,
    reward_name: str,
    cloud_run_uri: str,
) -> types.SingleReinforcementTuningRewardConfig:
    """Build a Cloud Run reward scorer — **documented builder only**.

    Delegates scoring to a separately deployed Cloud Run service at
    ``cloud_run_uri`` that implements the GEAP reward contract. This repo does not
    deploy such a service, so the example scripts do not exercise this scorer; it
    is provided to show the fourth reward-scorer shape. Use it when your reward
    needs custom dependencies or logic that will not run in the code-execution
    sandbox.
    """
    return types.SingleReinforcementTuningRewardConfig(
        reward_name=reward_name,
        cloud_run_reward_scorer=types.ReinforcementTuningCloudRunRewardScorer(
            cloud_run_uri=cloud_run_uri,
        ),
    )


def build_composite_reward_config(
    weighted: list[tuple[types.SingleReinforcementTuningRewardConfig, float]],
) -> types.CompositeReinforcementTuningRewardConfig:
    """Combine several single reward configs into one weighted composite reward.

    ``weighted`` is a list of ``(single_reward_config, weight)`` pairs; the final
    reward is the weighted sum of each scorer. Typical use: pair verifiable
    correctness with subjective quality, e.g. code-execution (0.8) + autorater
    (0.2). Pass the result to :func:`launch_rlft_job` /
    :func:`validate_reward_config` as ``composite_reward_config``.
    """
    return types.CompositeReinforcementTuningRewardConfig(
        weighted_reward_configs=[
            types.CompositeReinforcementTuningRewardConfigWeightedRewardConfig(
                reward_config=cfg,
                weight=weight,
            )
            for cfg, weight in weighted
        ],
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
    composite_reward_config: types.CompositeReinforcementTuningRewardConfig | None = None,
    export_last_checkpoint_only: bool = False,
    evaluation_config: types.EvaluationConfig | None = None,
    evaluate_interval: int | None = None,
    pre_tuned_model_checkpoint_id: str | None = None,
    checkpoint_interval: int | None = None,
    labels: dict[str, str] | None = None,
) -> Any:  # noqa: ANN401 - SDK returns a dynamically-typed TuningJob
    """Submit an RLFT job and return the created tuning job.

    ``train_uri``/``val_uri`` are ``gs://`` URIs to RLFT-format JSONL (``contents``
    + ``references``, no completion). ``samples_per_prompt`` is how many candidate
    generations are drawn per prompt for reward comparison. ``adapter_size`` must be
    a key of :data:`geap_tuning.sft.tune.ADAPTER_MAP` (raises ``KeyError``
    otherwise). NOTE: the RLFT docs call out ``gemini-3.5-flash`` as the supported
    base model — verify availability in your region before running live. Does not
    wait — poll with :func:`geap_tuning.jobs.wait_for_tuning_job`.

    ``export_last_checkpoint_only``, ``evaluation_config`` and
    ``pre_tuned_model_checkpoint_id`` behave as in :func:`launch_sft_job`; the
    documented best practice is SFT first, then continuous-tune with RLFT by
    passing the SFT model's resource name as ``base_model``.
    ``evaluate_interval`` sets the step cadence for ``evaluation_config`` eval
    runs and ``checkpoint_interval`` sets how many steps elapse between exported
    checkpoints — **both are reinforcement-tuning only** in google-genai 2.14.0
    (the SDK serializes them under ``reinforcementTuningSpec``, so the SFT/DPO
    launchers deliberately omit ``evaluate_interval``). See
    ``docs/notes/checkpoints-and-continuous-tuning.md``.

    ``reward_config`` and ``composite_reward_config`` are mutually exclusive: pass
    a :func:`build_composite_reward_config` result as ``composite_reward_config``
    to score with several weighted scorers, and the single-reward default is
    skipped. When neither is given, a default code-execution reward is used.

    ``labels`` behaves as in :func:`launch_sft_job`: resource labels on the job
    that propagate to the generated Model/Endpoint; ``None`` omits the field.
    See ``docs/notes/resource-labels.md``.
    """
    single_reward = None if composite_reward_config else (reward_config or build_reward_config())
    config = types.CreateTuningJobConfig(
        method="REINFORCEMENT_TUNING",
        tuned_model_display_name=display_name,
        epoch_count=epochs,
        adapter_size=ADAPTER_MAP[adapter_size],
        learning_rate_multiplier=learning_rate_multiplier,
        samples_per_prompt=samples_per_prompt,
        reward_config=single_reward,
        composite_reward_config=composite_reward_config,
        validation_dataset=types.TuningDataset(gcs_uri=val_uri) if val_uri else None,
        export_last_checkpoint_only=export_last_checkpoint_only,
        evaluation_config=evaluation_config,
        evaluate_interval=evaluate_interval,
        pre_tuned_model_checkpoint_id=pre_tuned_model_checkpoint_id,
        checkpoint_interval=checkpoint_interval,
        labels=labels,
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
    composite_reward_config: types.CompositeReinforcementTuningRewardConfig | None = None,
) -> Any:  # noqa: ANN401 - SDK returns a dynamically-typed ValidateRewardResponse
    """Preflight the reward on one example via ``tunings.validate_reward``.

    Returns the SDK ``ValidateRewardResponse`` (carrying ``overall_reward`` /
    ``error``); a non-null error or ``NaN`` means the reward is broken. Optional
    but recommended before launching — RLFT auto-stops if >80% of reward calls
    fail. ``example_record`` is an RLFT record (``contents`` + ``references``).

    ``reward_config`` and ``composite_reward_config`` are mutually exclusive and
    mirror :func:`launch_rlft_job`; pass a composite to preflight a weighted
    reward. When neither is given, the default code-execution reward is scored.
    """
    single_reward = None if composite_reward_config else (reward_config or build_reward_config())
    return client.tunings.validate_reward(
        parent=f"projects/{project}/locations/{location}",
        sample_response=types.Content(role="model", parts=[types.Part(text=sample_answer)]),
        example=types.ReinforcementTuningExample(
            contents=example_record["contents"],
            references=example_record["references"],
        ),
        single_reward_config=single_reward,
        composite_reward_config=composite_reward_config,
    )
