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


def build_evaluation_config(
    bucket: str,
    *,
    prefix: str = "tuning_eval",
    metrics: list[types.Metric] | None = None,
) -> types.EvaluationConfig:
    """Build the GEAP auto-eval config that runs after each checkpoint.

    ``bucket`` may be a bare name or a ``gs://`` URI; results are written under
    ``bucket/prefix``. ``metrics`` defaults to one pointwise metric — override it
    with metrics from the live eval catalog. Auto-eval is Preview and available
    in ``us-central1`` only.
    """
    return types.EvaluationConfig(
        metrics=metrics if metrics is not None else [_DEFAULT_METRIC],
        output_config=types.OutputConfig(
            gcs_destination=types.GcsDestination(
                output_uri_prefix=build_gcs_uri(bucket, prefix),
            ),
        ),
    )
