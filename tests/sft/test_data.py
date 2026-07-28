"""Tests for the SFT support-intent dataset builder."""

from collections import Counter
from pathlib import Path

from geap_tuning.sft.data import (
    LABELS,
    PROMPT,
    SUPPORT_TICKETS,
    build_records,
    build_sft_dataset,
    split_dataset,
)


def test_all_tickets_use_known_labels() -> None:
    assert {label for _, label in SUPPORT_TICKETS} <= set(LABELS)


def test_dataset_is_balanced_across_labels() -> None:
    counts = Counter(label for _, label in SUPPORT_TICKETS)
    assert len(SUPPORT_TICKETS) == 65
    assert set(counts) == set(LABELS)
    assert len(set(counts.values())) == 1  # every intent equally represented


def test_build_records_uses_prompt_and_label() -> None:
    rec = build_records([("My card was charged twice", "billing")])[0]
    assert PROMPT in rec["contents"][0]["parts"][0]["text"]
    assert "My card was charged twice" in rec["contents"][0]["parts"][0]["text"]
    assert rec["contents"][1]["parts"][0]["text"] == "billing"


def test_split_is_reproducible_and_disjoint() -> None:
    first = split_dataset(SUPPORT_TICKETS, seed=42)
    second = split_dataset(SUPPORT_TICKETS, seed=42)
    assert first == second  # deterministic

    train, val, test = first

    def texts(rows: list[tuple[str, str]]) -> set[str]:
        return {text for text, _ in rows}

    assert texts(train) & texts(val) == set()
    assert texts(train) & texts(test) == set()
    assert texts(val) & texts(test) == set()
    assert len(train) + len(val) + len(test) == len(SUPPORT_TICKETS)


def test_build_sft_dataset_writes_three_files(tmp_path: Path) -> None:
    paths = build_sft_dataset(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    for path in paths.values():
        assert Path(path).exists()
        assert Path(path).read_text(encoding="utf-8").strip()
