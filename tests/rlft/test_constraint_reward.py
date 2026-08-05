"""Tests for the graded constraint-satisfaction reward module."""

import ast
from pathlib import Path

from geap_tuning.rlft import constraint_reward
from geap_tuning.rlft.constraint_reward import component_breakdown, evaluate


def _response(text: str) -> dict[str, object]:
    return {"parts": [{"text": text}]}


def _refs(**kwargs: str) -> dict[str, str]:
    return dict(kwargs)


def test_all_constraints_met_scores_one() -> None:
    refs = _refs(
        required_keywords="launch,roadmap",
        forbidden_words="very",
        min_words="3",
        max_words="20",
        min_sentences="1",
        max_sentences="3",
    )
    text = "We launch the roadmap today. It ships soon."
    assert evaluate({"references": refs}, _response(text)) == 1.0


def test_miss_exactly_one_component_is_fractional() -> None:
    # Four keyword components; miss one → 3/4. Nothing else constrained.
    refs = _refs(required_keywords="alpha,beta,gamma,delta")
    text = "alpha beta gamma only"
    reward = evaluate({"references": refs}, _response(text))
    assert reward == 0.75
    assert 0.0 < reward < 1.0


def test_forbidden_word_respects_word_boundary() -> None:
    # "every" must NOT trip the forbidden "very" check.
    refs = _refs(forbidden_words="very")
    assert evaluate({"references": refs}, _response("every day counts")) == 1.0
    assert evaluate({"references": refs}, _response("this is very good")) == 0.0


def test_multi_word_keyword_phrase() -> None:
    refs = _refs(required_keywords="product launch")
    assert evaluate({"references": refs}, _response("our product launch is ready")) == 1.0
    assert evaluate({"references": refs}, _response("our product is ready")) == 0.0


def test_open_ended_word_band_max_only() -> None:
    refs = _refs(max_words="3")
    assert evaluate({"references": refs}, _response("one two three")) == 1.0
    assert evaluate({"references": refs}, _response("one two three four")) == 0.0


def test_open_ended_word_band_min_only() -> None:
    refs = _refs(min_words="3")
    assert evaluate({"references": refs}, _response("one two")) == 0.0
    assert evaluate({"references": refs}, _response("one two three")) == 1.0


def test_sentence_counting() -> None:
    refs = _refs(min_sentences="3", max_sentences="3")
    assert evaluate({"references": refs}, _response("A. B! C?")) == 1.0
    assert evaluate({"references": refs}, _response("A. B!")) == 0.0


def test_content_wrapper_is_unwrapped() -> None:
    refs = _refs(required_keywords="hello")
    response = {"content": {"parts": [{"text": "well hello there"}]}}
    assert evaluate({"references": refs}, response) == 1.0


def test_partial_references_counts_present_only() -> None:
    # Only forbidden present: one component, satisfied → 1.0.
    refs = _refs(forbidden_words="stuff")
    assert evaluate({"references": refs}, _response("clean prose here")) == 1.0


def test_empty_references_scores_one() -> None:
    assert evaluate({"references": {}}, _response("anything")) == 1.0


def test_breakdown_sum_matches_evaluate() -> None:
    refs = _refs(
        required_keywords="alpha,beta",
        forbidden_words="very,really",
        min_words="2",
        max_words="50",
    )
    text = "alpha only here"  # has alpha, missing beta; no forbidden; word band ok
    breakdown = component_breakdown(refs, _response(text))
    sat = sum(h for h, _ in breakdown.values())
    tot = sum(t for _, t in breakdown.values())
    assert evaluate({"references": refs}, _response(text)) == sat / tot


def test_module_imports_only_stdlib() -> None:
    # Sandbox guard: shipped verbatim, so it must import only re/typing/__future__.
    source = Path(constraint_reward.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"re", "typing", "__future__"}
