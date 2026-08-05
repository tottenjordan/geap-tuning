"""Tests for RLFT offline accuracy evaluation (reuses the training reward)."""

from geap_tuning.rlft.evaluate import (
    bootstrap_ci,
    content_correct,
    run_rlft_eval,
    run_rlft_multimetric_eval,
    score_accuracy,
)


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
    metrics = run_rlft_eval(records, generate_fn=lambda _user, _sys=None: next(replies))
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
    metrics = run_rlft_eval(records, generate_fn=lambda _user, _sys=None: next(replies))
    assert metrics["accuracy"] == 0.0
    assert metrics["content_accuracy"] == 1.0
    assert metrics["content_correct"] == 2


def test_run_rlft_eval_forwards_system_instruction() -> None:
    # The record's systemInstruction must reach generate_fn so eval replays the
    # training framing; a record without one forwards None.
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "2+2?"}]}],
            "systemInstruction": {"parts": [{"text": "End with 'Answer: <number>'."}]},
            "references": {"ground_truth_answer": "4"},
        },
        {
            "contents": [{"role": "user", "parts": [{"text": "3+3?"}]}],
            "references": {"ground_truth_answer": "6"},
        },
    ]
    seen: list[str | None] = []

    def generate_fn(_user: str, system_instruction: str | None) -> str:
        seen.append(system_instruction)
        return "Answer: 4"

    run_rlft_eval(records, generate_fn=generate_fn)
    assert seen == ["End with 'Answer: <number>'.", None]


def test_run_rlft_eval_format_rate_counts_markers_regardless_of_correctness() -> None:
    # format_rate credits a parseable "Answer:" marker even when the number is wrong,
    # and withholds credit when the correct number is stated only in prose.
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
    replies = iter(["Answer: 99", "The total is 6."])
    metrics = run_rlft_eval(records, generate_fn=lambda _u, _s=None: next(replies))
    assert metrics["format_rate"] == 0.5  # only the (wrong) marker reply counts
    assert metrics["format_count"] == 1
    assert metrics["accuracy"] == 0.0  # neither is both correct AND marked
    assert metrics["content_accuracy"] == 0.5  # the prose "6" is content-correct


def test_run_rlft_multimetric_eval_reports_all_axes_and_tiers() -> None:
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "2+2?"}]}],
            "references": {"ground_truth_answer": "4", "difficulty": "easy"},
        },
        {
            "contents": [{"role": "user", "parts": [{"text": "sum 1..10?"}]}],
            "references": {"ground_truth_answer": "55", "difficulty": "hard"},
        },
    ]
    # First: correct + marker. Second: correct in prose, no marker.
    replies = iter(["Answer: 4", "The sum is **55**."])
    metrics = run_rlft_multimetric_eval(
        records,
        generate_fn=lambda _u, _s=None: next(replies),
        judge_fn=lambda _q, _r, _t: 0.75,
    )
    assert metrics["n"] == 2
    assert metrics["correctness"] == 1.0  # both right (marker-agnostic)
    assert metrics["format_rate"] == 0.5  # only the first has a marker
    assert metrics["marker_accuracy"] == 0.5  # correct AND marked
    assert metrics["explanation_quality"] == 0.75
    assert metrics["by_difficulty"]["easy"] == {"correctness": 1.0, "n": 1}
    assert metrics["by_difficulty"]["hard"] == {"correctness": 1.0, "n": 1}


def test_run_rlft_multimetric_eval_omits_quality_without_judge() -> None:
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "2+2?"}]}],
            "references": {"ground_truth_answer": "4", "difficulty": "easy"},
        },
    ]
    metrics = run_rlft_multimetric_eval(records, generate_fn=lambda _u, _s=None: "Answer: 4")
    assert "explanation_quality" not in metrics


def test_run_rlft_multimetric_eval_forwards_system_instruction_and_judge_inputs() -> None:
    records = [
        {
            "contents": [{"role": "user", "parts": [{"text": "2+2?"}]}],
            "systemInstruction": {"parts": [{"text": "Think step by step."}]},
            "references": {"ground_truth_answer": "4", "difficulty": "easy"},
        },
    ]
    seen_sys: list[str | None] = []
    seen_judge: list[tuple[str, str, str]] = []

    def generate_fn(_user: str, system_instruction: str | None) -> str:
        seen_sys.append(system_instruction)
        return "Answer: 4"

    def judge_fn(question: str, reply: str, truth: str) -> float:
        seen_judge.append((question, reply, truth))
        return 1.0

    run_rlft_multimetric_eval(records, generate_fn=generate_fn, judge_fn=judge_fn)
    assert seen_sys == ["Think step by step."]
    assert seen_judge == [("2+2?", "Answer: 4", "4")]


def test_bootstrap_ci_is_seed_deterministic_and_bounded() -> None:
    # Bounds lie within [0, 1], bracket the point estimate, and are reproducible.
    low, high = bootstrap_ci(7, 10, seed=123)
    assert 0.0 <= low <= 0.7 <= high <= 1.0
    assert bootstrap_ci(7, 10, seed=123) == (low, high)
    # Degenerate cases: all-hit collapses to 1.0, empty set is (0, 0).
    assert bootstrap_ci(5, 5, seed=1) == (1.0, 1.0)
    assert bootstrap_ci(0, 0, seed=1) == (0.0, 0.0)
