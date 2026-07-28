"""Tests for RLFT offline accuracy evaluation (reuses the training reward)."""

from geap_tuning.rlft.evaluate import run_rlft_eval, score_accuracy


def test_score_accuracy() -> None:
    metrics = score_accuracy([1.0, -1.0, 1.0, 1.0])
    assert metrics["accuracy"] == 0.75
    assert metrics["correct"] == 3
    assert metrics["n"] == 4
    empty = score_accuracy([])
    assert empty["accuracy"] == 0.0
    assert empty["correct"] == 0


def test_run_rlft_eval_uses_injected_generate_fn() -> None:
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "2+2?"}]}],
            "references": {"ground_truth_answer": "4"},
        },
        {
            "contents": [{"role": "user", "parts": [{"text": "3+3?"}]}],
            "references": {"ground_truth_answer": "6"},
        },
    ]
    # Generator answers the first correctly, the second wrongly.
    replies = iter(["Answer: 4", "Answer: 99"])
    metrics = run_rlft_eval(records, generate_fn=lambda _user: next(replies))
    assert metrics["accuracy"] == 0.5
    assert metrics["correct"] == 1
    assert metrics["n"] == 2
