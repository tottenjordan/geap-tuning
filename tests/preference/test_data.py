"""Tests for the DPO support-reply style dataset builder."""

from pathlib import Path

from geap_tuning.preference.data import (
    SUPPORT_REPLIES,
    build_preference_dataset,
    build_preference_records,
    split_dataset,
)


def test_dataset_has_expected_size_and_unique_prompts() -> None:
    assert len(SUPPORT_REPLIES) == 37
    prompts = [message for message, _, _ in SUPPORT_REPLIES]
    assert len(set(prompts)) == len(prompts)  # no duplicate prompts


def test_build_records_orders_preferred_first() -> None:
    rec = build_preference_records([("Where is my order?", "Let me check now!", "Idk.")])[0]
    assert rec["contents"][0]["parts"][0]["text"] == "Where is my order?"
    assert [c["score"] for c in rec["completions"]] == [1, 0]
    assert rec["completions"][0]["completion"]["parts"][0]["text"] == "Let me check now!"
    assert rec["completions"][1]["completion"]["parts"][0]["text"] == "Idk."


def test_split_is_reproducible_and_disjoint() -> None:
    first = split_dataset(SUPPORT_REPLIES, seed=42)
    assert first == split_dataset(SUPPORT_REPLIES, seed=42)
    train, val, test = first
    prompts = lambda split: {p for p, _, _ in split}  # noqa: E731
    assert prompts(train) & prompts(val) == set()
    assert prompts(train) & prompts(test) == set()
    assert prompts(val) & prompts(test) == set()


def test_build_preference_dataset_writes_three_splits(tmp_path: Path) -> None:
    paths = build_preference_dataset(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    for path in paths.values():
        assert Path(path).exists()
