"""Tests for the structured JSON-extraction SFT dataset module."""

import json
from pathlib import Path

from geap_tuning.sft.extraction import (
    _CITY_MAP,
    _PRIORITY_MAP,
    EXTRACTION_EXAMPLES,
    SCHEMA_FIELDS,
    SYSTEM_INSTRUCTION,
    ExtractionExample,
    _build_bank,
    build_extraction_dataset,
    build_records,
    split_dataset,
)


def test_build_bank_is_deterministic() -> None:
    first = _build_bank()
    second = _build_bank()
    assert first == second
    assert len(first) == 200


def test_build_bank_lines_unique() -> None:
    lines = [ex.line for ex in _build_bank()]
    assert len(lines) == len(set(lines))


def test_examples_are_extraction_examples() -> None:
    assert EXTRACTION_EXAMPLES
    assert all(isinstance(ex, ExtractionExample) for ex in EXTRACTION_EXAMPLES)


def test_every_target_json_matches_schema() -> None:
    for ex in EXTRACTION_EXAMPLES:
        obj = json.loads(ex.target_json)
        assert set(obj) == set(SCHEMA_FIELDS)
        assert isinstance(obj["quantity"], int)


def test_gold_applies_the_house_normalization_convention() -> None:
    # The point of the redesign: the gold object is NOT the raw text — each field is
    # normalized by a convention the base cannot guess, so it has real headroom.
    priority_codes = set(_PRIORITY_MAP.values())
    canonical_cities = set(_CITY_MAP.values())
    canon_to_abbr = {canon: abbr for abbr, canon in _CITY_MAP.items()}
    for ex in EXTRACTION_EXAMPLES:
        obj = json.loads(ex.target_json)
        # priority normalized to a P-code, never the raw urgency word.
        assert obj["priority"] in priority_codes
        assert obj["priority"] not in _PRIORITY_MAP
        # city expanded to its canonical name; the raw line shows the abbreviation.
        assert obj["city"] in canonical_cities
        assert canon_to_abbr[obj["city"]] in ex.line
        # order_id has the ord- prefix stripped and is upper-cased; raw line keeps it.
        assert "ord-" in ex.line
        assert not obj["order_id"].startswith("ord-")
        assert obj["order_id"] == obj["order_id"].upper()


def test_target_json_is_sorted_keys() -> None:
    for ex in EXTRACTION_EXAMPLES[:20]:
        obj = json.loads(ex.target_json)
        assert ex.target_json == json.dumps(obj, sort_keys=True)


def test_build_records_shape() -> None:
    records = build_records(EXTRACTION_EXAMPLES[:5])
    assert len(records) == 5
    for record, ex in zip(records, EXTRACTION_EXAMPLES[:5], strict=True):
        contents = record["contents"]
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == ex.line
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == ex.target_json
        assert record["systemInstruction"]["parts"][0]["text"] == SYSTEM_INSTRUCTION


def test_split_dataset_is_deterministic_partition() -> None:
    train, val, test = split_dataset(EXTRACTION_EXAMPLES)
    again = split_dataset(EXTRACTION_EXAMPLES)
    assert (train, val, test) == again
    # Disjoint partition covering every example.
    all_lines = [ex.line for ex in train + val + test]
    assert len(all_lines) == len(EXTRACTION_EXAMPLES)
    assert set(all_lines) == {ex.line for ex in EXTRACTION_EXAMPLES}
    assert train
    assert val
    assert test


def test_build_extraction_dataset_writes_splits(tmp_path: Path) -> None:
    paths = build_extraction_dataset(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    for path in paths.values():
        lines = [json.loads(line) for line in open(path, encoding="utf-8")]  # noqa: SIM115, PTH123
        assert lines
        for record in lines:
            assert "contents" in record
