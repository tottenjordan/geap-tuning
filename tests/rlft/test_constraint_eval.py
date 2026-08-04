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
    # One record; craft a reply that satisfies every constraint of spec 0.
    spec = CONSTRAINT_SPECS[0]
    records = build_records([spec])
    refs = spec.references
    keywords = refs["required_keywords"].split(",")
    min_words = int(refs["min_words"])
    # Build a reply: all keywords + enough filler words + right sentence count.
    body_words = list(keywords)
    while len(body_words) < min_words + 2:
        body_words.append("word")
    n_sentences = int(refs["min_sentences"])
    # Distribute into the minimum number of sentences.
    chunk = max(1, len(body_words) // n_sentences)
    sentences = []
    for i in range(n_sentences):
        piece = body_words[i * chunk : (i + 1) * chunk] or ["word"]
        sentences.append(" ".join(piece))
    # Attach leftover words to the last sentence.
    leftover = body_words[n_sentences * chunk :]
    if leftover:
        sentences[-1] += " " + " ".join(leftover)
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
