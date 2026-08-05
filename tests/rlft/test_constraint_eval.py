"""Tests for the constrained-generation multimetric eval scorer."""

from geap_tuning.rlft.constrained import CONSTRAINT_SPECS, build_records
from geap_tuning.rlft.constraint_eval import run_eval


def _spec_records(n: int):  # noqa: ANN202 - test helper
    return build_records(CONSTRAINT_SPECS[:n])


def test_accuracy_is_mean_graded_reward() -> None:
    records = _spec_records(2)

    # Reply that satisfies everything → reward 1.0; empty reply → partial/low.
    def generate_fn(user_text: str, system_instruction: str | None) -> str:  # noqa: ARG001
        return "placeholder"

    result = run_eval(records, generate_fn)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n"] == 2


def test_full_satisfaction_counts_only_perfect() -> None:
    # One record; craft a reply that satisfies every constraint of spec 0. The
    # counts are EXACT (min == max), so the reply must hit them precisely.
    spec = CONSTRAINT_SPECS[0]
    records = build_records([spec])
    refs = spec.references
    keywords = refs["required_keywords"].split(",")
    n_words = int(refs["min_words"])  # == max_words (exact target)
    n_sentences = int(refs["min_sentences"])  # == max_sentences (exact target)
    # Build exactly n_words words: all keywords, padded with a filler token that is
    # neither a keyword nor a forbidden word.
    words = list(keywords)
    while len(words) < n_words:
        words.append("word")
    words = words[:n_words]
    # Distribute into exactly n_sentences sentences (each non-empty), remainder up front.
    per = n_words // n_sentences
    counts = [per] * n_sentences
    for i in range(n_words - per * n_sentences):
        counts[i] += 1
    sentences = []
    idx = 0
    for count in counts:
        sentences.append(" ".join(words[idx : idx + count]))
        idx += count
    reply = ". ".join(sentences) + "."

    def perfect_fn(user_text: str, system_instruction: str | None) -> str:  # noqa: ARG001
        return reply

    result = run_eval(records, perfect_fn)
    assert result["full_satisfaction_rate"] == 1.0
    assert result["accuracy"] == 1.0


def test_by_constraint_type_aggregates() -> None:
    records = _spec_records(3)

    def generate_fn(user_text: str, system_instruction: str | None) -> str:  # noqa: ARG001
        return "a b c d e f g h i j"

    result = run_eval(records, generate_fn)
    assert "keywords" in result["by_constraint_type"]
    for stats in result["by_constraint_type"].values():
        assert 0.0 <= stats["rate"] <= 1.0
        assert stats["total"] >= 0


def test_system_instruction_is_replayed() -> None:
    records = _spec_records(1)
    seen: list[str | None] = []

    def generate_fn(user_text: str, system_instruction: str | None) -> str:  # noqa: ARG001
        seen.append(system_instruction)
        return "reply"

    run_eval(records, generate_fn)
    assert seen[0] is not None
    assert "precise writer" in seen[0]
