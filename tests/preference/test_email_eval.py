"""Tests for the concise-email eval scorer (win-rate + compression)."""

from geap_tuning.preference.email import EMAIL_DRAFTS, build_preference_records
from geap_tuning.preference.email_eval import (
    run_email_eval,
    run_head_to_head_eval,
    run_pilot_eval,
    score_compression,
)


def _shorter_wins(_draft: str, cand_a: str, cand_b: str) -> str:
    """A position-agnostic judge: prefer whichever candidate slot has fewer words."""
    return "A" if len(cand_a.split()) <= len(cand_b.split()) else "B"


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


def test_head_to_head_tuned_wins_when_shorter() -> None:
    # Head-to-head is the honest before→after measure: tuned rewrite (candidate)
    # vs base rewrite (reference), blind with randomized A/B position. A judge that
    # always prefers the shorter slot must credit the shorter tuned rewrite in EVERY
    # item regardless of which slot the flip placed it in.
    records = build_preference_records(EMAIL_DRAFTS[:5])
    base_calls: list[str] = []
    tuned_calls: list[str] = []

    def base_gen(draft: str) -> str:
        base_calls.append(draft)
        return "this is a considerably longer base rewrite with many extra words indeed"

    def tuned_gen(draft: str) -> str:
        tuned_calls.append(draft)
        return "short"

    result = run_head_to_head_eval(records, base_gen, tuned_gen, _shorter_wins)
    assert result["win_rate"] == 1.0
    assert result["wins"] == result["hits"] == 5
    assert result["n"] == 5
    assert len(base_calls) == 5
    assert len(tuned_calls) == 5
    assert result["tuned_mean_compression"] < result["base_mean_compression"]


def test_head_to_head_tuned_loses_when_longer() -> None:
    # The mirror: a longer tuned rewrite must lose in every item — proving the
    # position flip is decoded correctly for both A and B placements.
    records = build_preference_records(EMAIL_DRAFTS[:4])

    def base_gen(_draft: str) -> str:
        return "tiny"

    def tuned_gen(_draft: str) -> str:
        return "this tuned rewrite is definitely quite a bit longer than the base one"

    result = run_head_to_head_eval(records, base_gen, tuned_gen, _shorter_wins)
    assert result["win_rate"] == 0.0
    assert result["wins"] == 0


def test_pilot_eval_base_loses_to_gold_when_longer() -> None:
    # The gate: base rewrite (candidate) vs the GOLD preferred reference. A base
    # that writes longer than the concise gold loses → real headroom for tuning.
    records = build_preference_records(EMAIL_DRAFTS[:5])
    calls: list[str] = []

    def gen(draft: str) -> str:
        calls.append(draft)
        return "this base rewrite is quite a lot longer than the gold reference is by a wide margin"

    result = run_pilot_eval(records, gen, _shorter_wins)
    assert result["win_rate"] == 0.0
    assert result["n"] == 5
    assert len(calls) == 5


def test_pilot_eval_base_wins_when_shorter() -> None:
    # A base that already writes shorter than the gold saturates the gate.
    records = build_preference_records(EMAIL_DRAFTS[:5])

    def gen(_draft: str) -> str:
        return "ok"

    result = run_pilot_eval(records, gen, _shorter_wins)
    assert result["win_rate"] == 1.0
