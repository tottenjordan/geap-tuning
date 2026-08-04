"""Evaluate a tuned SFT model on the structured JSON-extraction split.

Unlike :mod:`geap_tuning.sft.evaluate` (classification), this is a *generative*
scorer: the model emits a JSON string, which we parse robustly (stripping code
fences / surrounding prose) and compare field-by-field against the gold object.

The headline metric is ``accuracy`` — the micro field-level exact-match rate
(fraction of all ``field`` values across all records that match gold) — so it
plugs into any ``"accuracy"``-keyed machinery. Comparison is type-insensitive
(``3`` matches ``"3"``) so the metric measures *content*, while ``json_validity``
separately captures whether the model emitted parseable JSON at all. The model
call is injected as ``predict_fn`` so the logic is testable offline.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from geap_tuning.schemas import Record

_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Strip code fences, take the first ``{...}`` span, ``json.loads`` it.

    Returns ``None`` when nothing parseable is found or the parsed value is not
    a JSON object (e.g. an array).
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _norm(value: Any) -> str | None:  # noqa: ANN401 - a JSON scalar of unknown type
    """Type-insensitive normalization so ``3`` and ``"3"`` compare equal."""
    return None if value is None else str(value).strip().lower()


def score_extraction(
    golds: Sequence[dict[str, Any]],
    preds: Sequence[dict[str, Any] | None],
) -> dict[str, Any]:
    """Score parsed predictions against gold objects.

    Returns the micro field exact-match ``accuracy`` (headline), ``exact_match``
    (fraction of records matching gold on every field), ``json_validity``
    (fraction of predictions that parsed), a ``per_field`` accuracy breakdown,
    plus ``n``/``field_hits``/``field_total``. A ``None`` prediction misses every
    field and does not count as valid JSON.
    """
    field_hits = 0
    field_total = 0
    exact = 0
    valid = 0
    per_field_hits: dict[str, int] = {}
    per_field_total: dict[str, int] = {}
    for gold, pred in zip(golds, preds, strict=True):
        if pred is not None:
            valid += 1
        record_all_match = True
        for field, gold_value in gold.items():
            field_total += 1
            per_field_total[field] = per_field_total.get(field, 0) + 1
            pred_value = None if pred is None else pred.get(field)
            match = _norm(pred_value) == _norm(gold_value) and pred_value is not None
            if match:
                field_hits += 1
                per_field_hits[field] = per_field_hits.get(field, 0) + 1
            else:
                record_all_match = False
        if record_all_match and pred is not None:
            exact += 1
    n = len(golds)
    per_field = {
        field: per_field_hits.get(field, 0) / total for field, total in per_field_total.items()
    }
    return {
        "accuracy": field_hits / field_total if field_total else 0.0,
        "exact_match": exact / n if n else 0.0,
        "json_validity": valid / n if n else 0.0,
        "per_field": per_field,
        "n": n,
        "field_hits": field_hits,
        "field_total": field_total,
    }


def _user_text(record: Record) -> str:
    return record["contents"][0]["parts"][0]["text"]


def _gold_text(record: Record) -> str:
    return record["contents"][1]["parts"][0]["text"]


def run_eval(
    records: Sequence[Record],
    predict_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Predict each record's JSON with ``predict_fn`` and score against gold."""
    golds = [json.loads(_gold_text(r)) for r in records]
    preds = [parse_json_object(predict_fn(_user_text(r))) for r in records]
    return score_extraction(golds, preds)
