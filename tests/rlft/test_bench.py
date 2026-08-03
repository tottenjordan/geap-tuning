"""Tests for the harder, difficulty-tiered ranking bench (`rlft.bench`)."""

import json
from collections import Counter
from pathlib import Path

from geap_tuning.rlft.bench import (
    DIFFICULTIES,
    HARD_MATH_PROBLEMS,
    NEUTRAL_SYSTEM_INSTRUCTION,
    build_bench_dataset,
    build_bench_records,
    split_stratified,
)
from geap_tuning.rlft.reward import normalize_number


def test_bank_is_balanced_unique_and_numeric() -> None:
    # Every tier is equally represented and questions never repeat.
    by_tier = Counter(problem.difficulty for problem in HARD_MATH_PROBLEMS)
    assert set(by_tier) == set(DIFFICULTIES)
    assert len(set(by_tier.values())) == 1  # same count per tier
    questions = [problem.question for problem in HARD_MATH_PROBLEMS]
    assert len(questions) == len(set(questions))
    # Every computed answer is a parseable number in canonical form.
    for problem in HARD_MATH_PROBLEMS:
        assert float(problem.answer) == float(normalize_number(problem.answer))


def test_neutral_instruction_omits_marker_contract() -> None:
    # Format headroom hinges on the instruction NOT naming the "Answer:" marker.
    assert "Answer:" not in NEUTRAL_SYSTEM_INSTRUCTION


def test_split_stratified_balances_tiers_and_is_deterministic() -> None:
    train, val, test = split_stratified(HARD_MATH_PROBLEMS)
    # No overlap and full coverage.
    assert len(train) + len(val) + len(test) == len(HARD_MATH_PROBLEMS)
    all_questions = {p.question for p in (*train, *val, *test)}
    assert len(all_questions) == len(HARD_MATH_PROBLEMS)
    # Each split keeps every tier represented (balanced mix).
    for split in (train, val, test):
        tiers = Counter(p.difficulty for p in split)
        assert set(tiers) == set(DIFFICULTIES)
        assert len(set(tiers.values())) == 1
    # Deterministic for a fixed seed; a different seed reorders.
    assert split_stratified(HARD_MATH_PROBLEMS) == (train, val, test)
    other = split_stratified(HARD_MATH_PROBLEMS, seed=7)
    assert other != (train, val, test)


def test_build_bench_records_carries_difficulty_and_instruction() -> None:
    _, _, test = split_stratified(HARD_MATH_PROBLEMS)
    records = build_bench_records(test[:3])
    for record, problem in zip(records, test[:3], strict=True):
        assert record["references"]["ground_truth_answer"] == problem.answer
        assert record["references"]["difficulty"] == problem.difficulty
        assert record["systemInstruction"]["parts"][0]["text"] == NEUTRAL_SYSTEM_INSTRUCTION
        # RLFT records carry only the user turn (no gold completion).
        assert len(record["contents"]) == 1
        assert record["contents"][0]["role"] == "user"


def test_build_bench_records_accepts_custom_instruction() -> None:
    _, _, test = split_stratified(HARD_MATH_PROBLEMS)
    records = build_bench_records(test[:1], system_instruction="Custom framing.")
    assert records[0]["systemInstruction"]["parts"][0]["text"] == "Custom framing."


def test_build_bench_dataset_writes_all_splits(tmp_path: Path) -> None:
    paths = build_bench_dataset(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    for name, path in paths.items():
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        assert lines, f"{name} split is empty"
        record = json.loads(lines[0])
        assert record["references"]["ground_truth_answer"]
        assert record["references"]["difficulty"] in DIFFICULTIES
        assert record["systemInstruction"]["parts"][0]["text"] == NEUTRAL_SYSTEM_INSTRUCTION
