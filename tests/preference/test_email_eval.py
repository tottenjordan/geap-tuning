"""Tests for the concise-email eval scorer (win-rate + compression)."""

from geap_tuning.preference.email import EMAIL_DRAFTS, build_preference_records
from geap_tuning.preference.email_eval import run_email_eval, score_compression


def test_compression_mean_below_one_when_shorter() -> None:
    drafts = ["one two three four five six", "a b c d"]
    rewrites = ["one two three", "a b"]
    result = score_compression(drafts, rewrites)
    assert result["mean_compression"] < 1.0
    assert result["shorter_rate"] == 1.0
    assert result["n"] == 2


def test_compression_shorter_rate_partial() -> None:
    drafts = ["one two three four", "a b"]
    rewrites = ["one two", "a b c d"]  # first shorter, second longer
    result = score_compression(drafts, rewrites)
    assert result["shorter_rate"] == 0.5


def test_compression_empty_inputs() -> None:
    result = score_compression([], [])
    assert result["mean_compression"] == 0.0
    assert result["shorter_rate"] == 0.0
    assert result["n"] == 0


def test_compression_skips_zero_word_draft() -> None:
    drafts = ["", "one two three four"]
    rewrites = ["hello", "one two"]
    result = score_compression(drafts, rewrites)
    # The empty draft is skipped from the ratio (no ZeroDivision), n still counts it.
    assert result["mean_compression"] == 0.5
    assert result["n"] == 2


def test_run_email_eval_merges_metrics() -> None:
    records = build_preference_records(EMAIL_DRAFTS[:4])
    calls: list[str] = []

    def generate_fn(draft: str) -> str:
        calls.append(draft)
        return "short reply"  # 2 words — shorter than every verbose draft

    def judge_fn(_user: str, _cand: str, _ref: str) -> str:
        return "A"  # tuned candidate always wins

    result = run_email_eval(records, generate_fn, judge_fn)
    assert result["win_rate"] == 1.0
    assert "mean_compression" in result
    assert result["n"] == 4
    # generate_fn called exactly once per record.
    assert len(calls) == 4
