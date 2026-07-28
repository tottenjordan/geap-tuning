"""Tests for the self-contained RLFT math reward (also shipped to the sandbox)."""

from geap_tuning.rlft.reward import evaluate, extract_answer


def test_extract_answer_from_answer_line() -> None:
    assert extract_answer("Reasoning...\nAnswer: 391") == "391"


def test_extract_answer_handles_commas_and_trailing_period() -> None:
    assert extract_answer("Answer: 1,024.") == "1024"


def test_extract_answer_takes_last_marker() -> None:
    assert extract_answer("Answer: 5\nWait, correction.\nAnswer: 6") == "6"


def test_extract_answer_missing_returns_none() -> None:
    assert extract_answer("I am not sure.") is None


def test_evaluate_correct_returns_positive() -> None:
    example = {"references": {"ground_truth_answer": "391"}}
    response = {"parts": [{"text": "17 * 23 = 391\nAnswer: 391"}]}
    assert evaluate(example, response) == 1.0


def test_evaluate_wrong_returns_negative() -> None:
    example = {"references": {"ground_truth_answer": "391"}}
    response = {"parts": [{"text": "Answer: 40"}]}
    assert evaluate(example, response) == -1.0


def test_evaluate_unparseable_returns_negative() -> None:
    example = {"references": {"ground_truth_answer": "391"}}
    response = {"parts": [{"text": "no answer here"}]}
    assert evaluate(example, response) == -1.0


def test_evaluate_accepts_content_wrapper() -> None:
    example = {"references": {"ground_truth_answer": "4"}}
    response = {"content": {"parts": [{"text": "Answer: 4"}]}}
    assert evaluate(example, response) == 1.0
