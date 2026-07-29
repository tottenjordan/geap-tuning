"""Build the auto-eval config that GEAP runs during a tuning job.

Any tuning job (SFT/DPO/RLFT) can attach an :class:`~google.genai.types.EvaluationConfig`
via ``CreateTuningJobConfig.evaluation_config``; GEAP then runs the requested
metrics after each checkpoint and writes results to Cloud Storage. This is
distinct from the per-service *offline* ``evaluate.py`` modules (which score a
finished endpoint locally) — hence the noun name ``autoeval``.

See ``docs/notes/checkpoints-and-continuous-tuning.md``. The auto-eval service is
Preview and (per the GEAP docs) available in ``us-central1`` only.
"""

from __future__ import annotations

from google.genai import types

from geap_tuning.gcs import build_gcs_uri

# A single pointwise metric as a sensible default. NOTE: metric names and
# prompt templates must be verified against the live GEAP eval catalog before a
# real run — the catalog evolves and this is only a starting point.
_DEFAULT_METRIC = types.Metric(
    name="FLUENCY",
    prompt_template="Evaluate the fluency of this response: {prediction}",
)


def llm_judge_metric(
    name: str,
    prompt_template: str,
    *,
    judge_model_system_instruction: str | None = None,
) -> types.Metric:
    """Build a pointwise **LLM-judge** metric (a judge model scores each response).

    ``prompt_template`` is a judge instruction that may reference ``{prediction}``
    (and other dataset fields); ``judge_model_system_instruction`` optionally sets
    the judge's system prompt. NOTE: the SDK **lowercases** ``name``.
    """
    return types.Metric(
        name=name,
        prompt_template=prompt_template,
        judge_model_system_instruction=judge_model_system_instruction,
    )


def computation_metric(metric_type: types.ComputationBasedMetricType) -> types.UnifiedMetric:
    """Build a deterministic **computation** metric (no judge model).

    ``metric_type`` is one of :class:`~google.genai.types.ComputationBasedMetricType`
    (``EXACT_MATCH`` / ``BLEU`` / ``ROUGE``); the metric compares each prediction
    against the record's reference text. ``UnifiedMetric`` has no separate name —
    the spec type identifies the metric in the output.
    """
    return types.UnifiedMetric(
        computation_based_metric_spec=types.ComputationBasedMetricSpec(type=metric_type),
    )


def predefined_metric(metric_spec_name: str) -> types.UnifiedMetric:
    """Build a **predefined catalog** metric by name.

    ``metric_spec_name`` selects a managed metric from the live GEAP catalog
    (e.g. ``"text_quality_v1"``, ``"instruction_following_v1"``). Catalog names
    evolve — **verify the name exists before a real run.**
    """
    return types.UnifiedMetric(
        predefined_metric_spec=types.PredefinedMetricSpec(metric_spec_name=metric_spec_name),
    )


def build_autorater_config(
    *,
    sampling_count: int = 4,
    flip_enabled: bool | None = None,
    autorater_model: str | None = None,
) -> types.AutoraterConfig:
    """Configure the judge model shared by all LLM-judge metrics in a run.

    ``sampling_count`` (1-32) is how many judge samples are averaged per metric;
    ``flip_enabled`` swaps candidate order to counter position bias (pairwise
    only); ``autorater_model`` overrides the default judge with a publisher model
    or tuned endpoint. Pass the result as ``build_evaluation_config(..., autorater_config=)``.
    """
    return types.AutoraterConfig(
        sampling_count=sampling_count,
        flip_enabled=flip_enabled,
        autorater_model=autorater_model,
    )


def build_evaluation_config(
    bucket: str,
    *,
    prefix: str = "tuning_eval",
    metrics: list[types.Metric | types.UnifiedMetric] | None = None,
    autorater_config: types.AutoraterConfig | None = None,
    inference_generation_config: types.GenerationConfig | None = None,
) -> types.EvaluationConfig:
    """Build the GEAP auto-eval config that runs after each checkpoint.

    ``bucket`` may be a bare name or a ``gs://`` URI; results are written under
    ``bucket/prefix``. ``metrics`` defaults to one pointwise metric — override it
    with a mix of :func:`llm_judge_metric`, :func:`computation_metric`, and
    :func:`predefined_metric`. ``autorater_config`` (see
    :func:`build_autorater_config`) tunes the judge model shared by LLM-judge
    metrics; ``inference_generation_config`` (a
    :class:`~google.genai.types.GenerationConfig`) controls how the tuned model
    generates the responses being evaluated (e.g. ``temperature=0.0`` for
    deterministic scoring). Both are omitted from the config when ``None``.
    Auto-eval is Preview and available in ``us-central1`` only.
    """
    return types.EvaluationConfig(
        metrics=metrics if metrics is not None else [_DEFAULT_METRIC],
        output_config=types.OutputConfig(
            gcs_destination=types.GcsDestination(
                output_uri_prefix=build_gcs_uri(bucket, prefix),
            ),
        ),
        autorater_config=autorater_config,
        inference_generation_config=inference_generation_config,
    )
