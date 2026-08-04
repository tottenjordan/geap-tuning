"""Evaluate a DPO-tuned concise-email model: win-rate + a compression proxy.

The headline metric is the autorater ``win_rate`` (reused from
:mod:`geap_tuning.preference.evaluate`): the tuned rewrite versus the
dispreferred reference in a blind A/B judgment. Because a judge can be noisy on
style, an **objective** compression proxy corroborates it — ``mean_compression``
(rewrite/draft word ratio; below 1.0 means the model shortened the draft) and
``shorter_rate``. Both the generator and judge are injected so the logic is
unit-testable offline; a single generation pass feeds both metrics.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from geap_tuning.preference.evaluate import score_winrate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from geap_tuning.schemas import Record

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def score_compression(
    drafts: Sequence[str],
    rewrites: Sequence[str],
) -> dict[str, Any]:
    """Objective concision metrics for rewrites relative to their drafts.

    ``mean_compression`` averages the rewrite/draft word ratio (below 1.0 =
    shorter), skipping zero-word drafts to avoid division by zero.
    ``shorter_rate`` is the fraction of pairs where the rewrite is strictly
    shorter. ``n`` counts all pairs (including any skipped from the ratio).
    """
    ratios: list[float] = []
    shorter = 0
    total_rewrite_words = 0
    for draft, rewrite in zip(drafts, rewrites, strict=True):
        d_words = _words(draft)
        r_words = _words(rewrite)
        total_rewrite_words += r_words
        if r_words < d_words:
            shorter += 1
        if d_words > 0:
            ratios.append(r_words / d_words)
    n = len(drafts)
    return {
        "mean_compression": sum(ratios) / len(ratios) if ratios else 0.0,
        "shorter_rate": shorter / n if n else 0.0,
        "mean_rewrite_words": total_rewrite_words / n if n else 0.0,
        "n": n,
    }


def _draft_text(record: Record) -> str:
    return record["contents"][0]["parts"][0]["text"]


def _dispreferred_text(record: Record) -> str:
    for completion in record["completions"]:
        if completion["score"] == 0:
            return completion["completion"]["parts"][0]["text"]
    msg = "no dispreferred (score 0) completion"
    raise ValueError(msg)


def run_email_eval(
    records: Sequence[Record],
    generate_fn: Callable[[str], str],
    judge_fn: Callable[[str, str, str], str],
) -> dict[str, Any]:
    """Generate a rewrite per draft, A/B-judge vs the dispreferred reference.

    ``judge_fn(draft, candidate_a, candidate_b)`` returns ``"A"``/``"B"``;
    candidate A is the tuned rewrite and B the dispreferred reference, so an
    ``"A"`` verdict is a win. Returns ``score_winrate`` merged with
    ``score_compression`` (headline ``win_rate``), from one generation pass.
    """
    drafts = [_draft_text(r) for r in records]
    rewrites = [generate_fn(d) for d in drafts]
    dispreferred = [_dispreferred_text(r) for r in records]
    judgments = [
        judge_fn(draft, rewrite, ref)
        for draft, rewrite, ref in zip(drafts, rewrites, dispreferred, strict=True)
    ]
    return {**score_winrate(judgments), **score_compression(drafts, rewrites)}
