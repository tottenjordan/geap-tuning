"""Evaluate a tuned multimodal SFT model on the oral-disease splits.

The scoring is identical to text SFT — multi-class classification metrics — so
this reuses :func:`geap_tuning.sft.evaluate.score_classification`. The only
image-specific parts are (1) pulling the gold label and image URI out of a
``fileData`` record, (2) mapping a staged GCS URI back to its local copy so the
driver can send raw image bytes to the endpoint, and (3) canonicalizing the
model's free-text answer to one of the five display labels.

The model call is injected as ``predict_fn`` so the logic stays testable offline:
the driver passes a closure that loads local image bytes and calls the endpoint,
while tests pass a trivial stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from geap_tuning.sft.evaluate import score_classification
from geap_tuning.sft_vision.data import DATA_SEGMENT, LABEL_MAP

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from geap_tuning.schemas import Record

UNKNOWN = "unknown"


def gold_label(record: Record) -> str:
    """Return the ground-truth display label (the model turn's text)."""
    return record["contents"][1]["parts"][0]["text"]


def image_gcs_uri_of(record: Record) -> str:
    """Return the ``fileData`` GCS URI from a multimodal record's user turn."""
    return record["contents"][0]["parts"][0]["fileData"]["fileUri"]


def resolve_local_path(gcs_uri: str, local_root: str | Path) -> Path:
    """Map a staged image ``gcs_uri`` back to its local copy under ``local_root``.

    Splits on the ``/data/`` segment the staging layout inserts (see
    :func:`geap_tuning.sft_vision.data.image_gcs_uri`) and rejoins the
    ``{split}/{class}/{filename}`` tail under ``local_root``.
    """
    marker = f"/{DATA_SEGMENT}/"
    _, _, tail = gcs_uri.partition(marker)
    if not tail:
        msg = f"URI is not a staged dataset image (no {marker!r} segment): {gcs_uri}"
        raise ValueError(msg)
    return Path(local_root) / tail


def parse_prediction(text: str) -> str:
    """Canonicalize a free-text model answer to one of the display labels.

    Matches case-insensitively on either the raw class key (``"gingivitis"``) or
    the display label (``"Dental Calculus"``). Returns :data:`UNKNOWN` when the
    answer mentions no known class.
    """
    lower = text.lower()
    for class_name, label in LABEL_MAP.items():
        if class_name in lower or label.lower() in lower:
            return label
    return UNKNOWN


def run_image_eval(
    records: Sequence[Record],
    predict_fn: Callable[[Record], str],
) -> dict[str, Any]:
    """Predict each record with ``predict_fn`` and score against the gold labels.

    ``predict_fn`` receives the whole record and returns the model's **raw** text;
    this canonicalizes it via :func:`parse_prediction` before scoring.
    """
    y_true = [gold_label(record) for record in records]
    y_pred = [parse_prediction(predict_fn(record)) for record in records]
    return score_classification(y_true, y_pred)


def select_best_experiment(results: Mapping[str, dict[str, Any]]) -> str:
    """Return the experiment name with the highest accuracy (tie-break by name).

    ``results`` maps an experiment name to its metrics dict (as returned by
    :func:`run_image_eval`). Raises ``ValueError`` when empty.
    """
    if not results:
        msg = "No experiment results to select from"
        raise ValueError(msg)
    return max(sorted(results), key=lambda name: results[name]["accuracy"])
