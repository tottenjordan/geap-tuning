"""Evaluate a tuned SFT model on the held-out support-intent split.

Scoring is standard multi-class classification metrics from scikit-learn. The
model call is injected as ``predict_fn`` so the logic is testable offline: the
example driver passes a closure over :func:`geap_tuning.inference.generate`
bound to the tuned endpoint, while tests pass a trivial stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.metrics import accuracy_score, classification_report, f1_score

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from geap_tuning.schemas import Record


def score_classification(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, Any]:
    """Return accuracy, macro-averaged F1, and a per-label classification report.

    ``macro_f1`` weights every intent equally regardless of support, so it is a
    fairer headline metric than accuracy on this roughly-balanced dataset.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def _user_text(record: Record) -> str:
    return record["contents"][0]["parts"][0]["text"]


def _gold_label(record: Record) -> str:
    return record["contents"][1]["parts"][0]["text"]


def run_eval(
    records: Sequence[Record],
    predict_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Predict each record's label with ``predict_fn`` and score against gold."""
    y_true = [_gold_label(r) for r in records]
    y_pred = [predict_fn(_user_text(r)).strip() for r in records]
    return score_classification(y_true, y_pred)
