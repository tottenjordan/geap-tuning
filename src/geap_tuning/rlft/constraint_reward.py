"""Graded constraint-satisfaction reward for the RLFT constrained-generation demo.

This module is BOTH imported (offline eval + unit tests) AND shipped verbatim to
the GEAP code-execution reward sandbox as ``python_code_snippet`` (see
:func:`geap_tuning.rlft.tune.build_reward_config`). The sandbox runs this source,
then calls ``evaluate(example, response) -> float`` with camelCase ProtoJSON
dicts (``example`` carries ``references``; ``response`` is a ``Content`` with
``parts``). It must therefore stay self-contained — stdlib only, no
``geap_tuning`` imports.

The reward is the **fraction** of independently-checked constraint components
satisfied, in ``[0, 1]``. Each required keyword and each forbidden word is its
own component, and the word-count / sentence-count bands are one component each.
This graded design is deliberate: it yields a non-degenerate reward distribution
(variance even for a mediocre rollout) *and* real headroom (satisfying every
component at once is hard), which is what a useful RL signal needs.
"""

from __future__ import annotations

import re
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_RE = re.compile(r"[.!?]+")


def _response_text(response: dict[str, Any]) -> str:
    content = response.get("content", response)
    return " ".join(p.get("text", "") for p in content.get("parts", []))


def _split_csv(value: str) -> list[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def _parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _appears(text_lower: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", text_lower) is not None


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _sentence_count(text: str) -> int:
    return len([s for s in _SENT_RE.split(text) if s.strip()])


def component_breakdown(
    references: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, tuple[int, int]]:
    """Return ``{constraint_type: (satisfied, total)}``.

    Each keyword and each forbidden word is its own component (fine-grained
    partial credit); each count-band is a single component. A constraint type
    absent from ``references`` contributes ``(0, 0)`` and does not affect the
    score.
    """
    text = _response_text(response)
    lower = text.lower()
    out: dict[str, list[int]] = {
        "keywords": [0, 0],
        "forbidden": [0, 0],
        "word_count": [0, 0],
        "sentence_count": [0, 0],
    }
    for kw in _split_csv(references.get("required_keywords", "")):
        out["keywords"][1] += 1
        out["keywords"][0] += int(_appears(lower, kw))
    for word in _split_csv(references.get("forbidden_words", "")):
        out["forbidden"][1] += 1
        out["forbidden"][0] += int(not _appears(lower, word))
    lo = _parse_int(references.get("min_words", ""))
    hi = _parse_int(references.get("max_words", ""))
    if lo is not None or hi is not None:
        out["word_count"][1] += 1
        n = _word_count(text)
        ok = (lo is None or n >= lo) and (hi is None or n <= hi)
        out["word_count"][0] += int(ok)
    slo = _parse_int(references.get("min_sentences", ""))
    shi = _parse_int(references.get("max_sentences", ""))
    if slo is not None or shi is not None:
        out["sentence_count"][1] += 1
        s = _sentence_count(text)
        ok = (slo is None or s >= slo) and (shi is None or s <= shi)
        out["sentence_count"][0] += int(ok)
    return {c: (h, t) for c, (h, t) in out.items()}


def evaluate(example: dict[str, Any], response: dict[str, Any]) -> float:
    """Graded reward: fraction of constraint components satisfied, in ``[0, 1]``.

    Returns ``1.0`` when there are no constraints (nothing to violate).
    """
    refs = example.get("references", {}) or {}
    sat = tot = 0
    for h, t in component_breakdown(refs, response).values():
        sat += h
        tot += t
    return sat / tot if tot else 1.0
