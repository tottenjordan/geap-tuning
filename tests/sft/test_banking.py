"""Tests for the banking77 intent-classification dataset loader.

Dataset-shaping tests use the offline ``csv_dir`` path (fake CSVs written to
``tmp_path``). The download path is covered too, with ``urlopen`` monkeypatched, so
no test touches the network.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Self

import pytest

from geap_tuning.sft import banking
from geap_tuning.sft.banking import (
    banking_labels,
    build_banking_dataset,
    build_banking_records,
    build_system_instruction,
    load_pairs_from_csv,
    parse_banking_prediction,
    sample_balanced,
)

# A tiny fake banking77 split: 3 labels, uneven counts (one label short of 3).
_TRAIN_ROWS = [
    ("I am still waiting on my card?", "card_arrival"),
    ("Where is my new card?", "card_arrival"),
    ("Has my card shipped yet?", "card_arrival"),
    ("What is the current exchange rate?", "exchange_rate"),
    ("How is the exchange rate calculated?", "exchange_rate"),
    ("Do you support USD to EUR rates?", "exchange_rate"),
    ("My PIN is blocked, help.", "pin_blocked"),  # only 1 -> tests capping
]
_TEST_ROWS = [
    ("Still no card after two weeks.", "card_arrival"),
    ("Rate seems wrong today.", "exchange_rate"),
    ("PIN got blocked again.", "pin_blocked"),
]


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["text,category", *(f'"{text}",{label}' for text, label in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_csv_dir(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_csv(tmp_path / "train.csv", _TRAIN_ROWS)
    _write_csv(tmp_path / "test.csv", _TEST_ROWS)
    return tmp_path


def test_load_pairs_round_trips(tmp_path: Path) -> None:
    _write_csv(tmp_path / "train.csv", _TRAIN_ROWS)
    pairs = load_pairs_from_csv(tmp_path / "train.csv")
    assert pairs == _TRAIN_ROWS


def test_banking_labels_sorted_unique() -> None:
    assert banking_labels(_TRAIN_ROWS) == ("card_arrival", "exchange_rate", "pin_blocked")


def test_sample_balanced_caps_and_is_deterministic() -> None:
    first = sample_balanced(_TRAIN_ROWS, 2, seed=7)
    second = sample_balanced(_TRAIN_ROWS, 2, seed=7)
    assert first == second  # deterministic
    counts = Counter(label for _, label in first)
    assert counts["card_arrival"] == 2
    assert counts["exchange_rate"] == 2
    assert counts["pin_blocked"] == 1  # only one available -> capped


def test_build_system_instruction_lists_every_label() -> None:
    labels = banking_labels(_TRAIN_ROWS)
    instruction = build_system_instruction(labels)
    for label in labels:
        assert label in instruction


def test_build_banking_records_shape() -> None:
    system = build_system_instruction(banking_labels(_TRAIN_ROWS))
    rec = build_banking_records([("Where is my card?", "card_arrival")], system_instruction=system)[
        0
    ]
    assert rec["contents"][0]["parts"][0]["text"] == "Where is my card?"
    assert rec["contents"][1]["parts"][0]["text"] == "card_arrival"
    assert rec["systemInstruction"]["parts"][0]["text"] == system


def test_parse_prediction_variants() -> None:
    labels = ("card_arrival", "exchange_rate", "pin_blocked")
    assert parse_banking_prediction("card_arrival", labels) == "card_arrival"
    assert parse_banking_prediction("Card Arrival.", labels) == "card_arrival"
    assert parse_banking_prediction("the intent is card_arrival", labels) == "card_arrival"
    assert parse_banking_prediction("total gibberish", labels) == "total_gibberish"


def test_build_banking_dataset_disjoint_and_written(tmp_path: Path) -> None:
    csv_dir = _fake_csv_dir(tmp_path / "raw")
    out = tmp_path / "out"
    paths = build_banking_dataset(
        out,
        csv_dir=csv_dir,
        per_class={"train": 2, "val": 1, "test": 1},
    )
    assert set(paths) == {"train", "val", "test"}

    def read(name: str) -> list[dict]:
        return [json.loads(line) for line in Path(paths[name]).read_text().splitlines()]

    train, val, test = read("train"), read("val"), read("test")

    def user_texts(records: list[dict]) -> set[str]:
        return {r["contents"][0]["parts"][0]["text"] for r in records}

    # train/val disjoint per the carve-out.
    assert user_texts(train) & user_texts(val) == set()
    # every record carries the shared system instruction.
    system = build_system_instruction(banking_labels(_TRAIN_ROWS))
    for records in (train, val, test):
        assert all(r["systemInstruction"]["parts"][0]["text"] == system for r in records)
    # test split drawn from the held-out test CSV.
    assert user_texts(test) <= {text for text, _ in _TEST_ROWS}


# --- download caching: a partial fetch must never be cached as valid -------------


def test_download_is_atomic_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupted download must leave no CSV behind, not a truncated one."""

    class _Partial:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            msg = "connection reset mid-download"
            raise OSError(msg)

    monkeypatch.setattr(banking.urllib.request, "urlopen", lambda *_a, **_k: _Partial())

    with pytest.raises(OSError, match="connection reset"):
        banking.download_banking77(tmp_path)

    assert not (tmp_path / "train.csv").exists()  # no truncated cache
    assert not list(tmp_path.glob("*.part"))  # and no leftover temp file


def test_download_writes_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"text,category\nhi,card_arrival\n"
    calls: list[str] = []

    class _Ok:
        def __init__(self, url: str) -> None:
            calls.append(url)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return payload

    monkeypatch.setattr(banking.urllib.request, "urlopen", lambda url, **_k: _Ok(url))

    paths = banking.download_banking77(tmp_path)
    assert paths["train"].read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))

    # Second call is served from cache: no further network access.
    banking.download_banking77(tmp_path)
    assert len(calls) == 2  # train + test, from the first call only
