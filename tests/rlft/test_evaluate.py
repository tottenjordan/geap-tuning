"""Tests for RLFT offline accuracy evaluation (reuses the training reward)."""

from geap_tuning.rlft.evaluate import content_correct, run_rlft_eval, score_accuracy


def test_score_accuracy() -> None:
    metrics = score_accuracy([1.0, -1.0, 1.0, 1.0])
    assert metrics["accuracy"] == 0.75
    assert metrics["correct"] == 3
    assert metrics["n"] == 4
    empty = score_accuracy([])
    assert empty["accuracy"] == 0.0
    assert empty["correct"] == 0


def test_content_correct_marker_agnostic() -> None:
    # Correct number in prose, no "Answer:" marker → still a content match.
    assert content_correct("The sum is **55**.", "55") is True
    # Normalization: commas, currency prose, and trailing ".0" all reconcile.
    assert content_correct("It costs $6.50 total.", "6.50") is True
    assert content_correct("1,000 people", "1000") is True
    assert content_correct("That is 6 items", "6.0") is True
    # Wrong number, and empty ground truth, are never matches.
    assert content_correct("The answer is 54.", "55") is False
    assert content_correct("55", "") is False


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
    # Both replies use the marker; content accuracy tracks the marker score here.
    assert metrics["content_accuracy"] == 0.5
    assert metrics["content_correct"] == 1


def test_run_rlft_eval_content_accuracy_diverges_from_marker() -> None:
    # Correct answers stated in prose (no "Answer:" marker): the reward-based
    # accuracy scores 0, but content_accuracy credits the correct math.
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "sum 1..10?"}]}],
            "references": {"ground_truth_answer": "55"},
        },
        {
            "contents": [{"role": "user", "parts": [{"text": "2^8?"}]}],
            "references": {"ground_truth_answer": "256"},
        },
    ]
    replies = iter(["The sum is **55**.", "2 to the 8th is 256."])
    metrics = run_rlft_eval(records, generate_fn=lambda _user: next(replies))
    assert metrics["accuracy"] == 0.0
    assert metrics["content_accuracy"] == 1.0
    assert metrics["content_correct"] == 2
