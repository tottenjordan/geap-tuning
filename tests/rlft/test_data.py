"""Tests for the verifiable-math RLFT dataset builder."""

from pathlib import Path

from geap_tuning.rlft.data import (
    MATH_PROBLEMS,
    build_rlft_dataset,
    build_rlft_records,
    split_dataset,
)


def test_build_records_carries_references_and_no_completions() -> None:
    rec = build_rlft_records([("What is 2+2? End with 'Answer: <n>'.", "4")])[0]
    assert rec["references"] == {"ground_truth_answer": "4"}
    assert len(rec["contents"]) == 1
    assert rec["contents"][0]["role"] == "user"
    assert "completions" not in rec


def test_split_is_reproducible_and_disjoint() -> None:
    first = split_dataset(MATH_PROBLEMS, seed=42)
    assert first == split_dataset(MATH_PROBLEMS, seed=42)
    train, val, test = first
    questions = lambda split: {q for q, _ in split}  # noqa: E731
    assert questions(train) & questions(val) == set()
    assert questions(train) & questions(test) == set()
    assert questions(val) & questions(test) == set()


def test_build_rlft_dataset_writes_three_splits(tmp_path: Path) -> None:
    paths = build_rlft_dataset(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    for path in paths.values():
        assert Path(path).exists()
