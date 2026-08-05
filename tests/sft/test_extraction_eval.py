"""Tests for the structured JSON-extraction eval scorer."""

import json

from geap_tuning.sft.extraction import EXTRACTION_EXAMPLES, build_records
from geap_tuning.sft.extraction_eval import (
    parse_json_object,
    run_eval,
    score_extraction,
)


def test_parse_plain_json() -> None:
    assert parse_json_object('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_parse_fenced_json() -> None:
    text = '```json\n{"a": 1}\n```'
    assert parse_json_object(text) == {"a": 1}


def test_parse_json_in_prose() -> None:
    text = 'Sure! Here is the object: {"a": 1, "b": 2} — hope that helps.'
    assert parse_json_object(text) == {"a": 1, "b": 2}


def test_parse_non_json_returns_none() -> None:
    assert parse_json_object("no json here") is None


def test_parse_json_array_returns_none() -> None:
    assert parse_json_object("[1, 2, 3]") is None


def test_score_one_wrong_of_five() -> None:
    gold = {"order_id": "A1", "item": "mouse", "quantity": 3, "city": "Austin", "priority": "high"}
    pred = {"order_id": "A1", "item": "mouse", "quantity": 3, "city": "Austin", "priority": "low"}
    result = score_extraction([gold], [pred])
    assert result["accuracy"] == 0.8
    assert result["exact_match"] == 0.0
    assert result["json_validity"] == 1.0


def test_score_identical_is_perfect() -> None:
    gold = {"order_id": "A1", "item": "mouse", "quantity": 3, "city": "Austin", "priority": "high"}
    result = score_extraction([gold], [dict(gold)])
    assert result["accuracy"] == 1.0
    assert result["exact_match"] == 1.0


def test_none_pred_drops_validity_and_misses_all() -> None:
    gold = {"order_id": "A1", "item": "mouse", "quantity": 3, "city": "Austin", "priority": "high"}
    result = score_extraction([gold], [None])
    assert result["json_validity"] == 0.0
    assert result["accuracy"] == 0.0
    assert result["exact_match"] == 0.0


def test_int_vs_str_quantity_matches() -> None:
    gold = {"order_id": "A1", "item": "mouse", "quantity": 3, "city": "Austin", "priority": "high"}
    pred = {
        "order_id": "A1",
        "item": "mouse",
        "quantity": "3",
        "city": "Austin",
        "priority": "high",
    }
    result = score_extraction([gold], [pred])
    assert result["accuracy"] == 1.0
    assert result["exact_match"] == 1.0


def test_per_field_breakdown() -> None:
    gold = {"order_id": "A1", "item": "mouse", "quantity": 3, "city": "Austin", "priority": "high"}
    pred = {"order_id": "A1", "item": "mouse", "quantity": 3, "city": "Austin", "priority": "low"}
    result = score_extraction([gold], [pred])
    assert result["per_field"]["priority"] == 0.0
    assert result["per_field"]["item"] == 1.0


def test_run_eval_with_stub() -> None:
    records = build_records(EXTRACTION_EXAMPLES[:3])

    def predict_fn(_user_text: str) -> str:
        # Echo the first gold answer for all — a deliberately imperfect stub.
        return records[0]["contents"][1]["parts"][0]["text"]

    result = run_eval(records, predict_fn)
    assert result["n"] == 3
    assert 0.0 <= result["accuracy"] <= 1.0
    # First record's gold echoed back → its fields all match.
    first_gold = json.loads(records[0]["contents"][1]["parts"][0]["text"])
    assert set(first_gold)  # sanity
