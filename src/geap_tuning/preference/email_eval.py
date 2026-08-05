"""Evaluate a DPO-tuned concise-email model: win-rate + a compression proxy.

The honest **before → after** metric is a *head-to-head* autorater win-rate
(:func:`run_head_to_head_eval`): the tuned rewrite versus the **base** rewrite of
the same draft, judged blind with a randomized A/B position to cancel a judge's
slot bias. ``win_rate`` above 0.5 means tuning helped. This replaces the earlier
``run_email_eval`` design, which judged the model against the dataset's *fixed
dispreferred reference* — a strawman the base already beat ~100% of the time, so
it was pre-saturated and could never surface a lift (kept below only for the unit
test that documents it).

Before spending on a job, :func:`run_pilot_eval` gates on headroom the RLFT way:
it judges the base rewrite against the **gold preferred** reference. If the base
already matches/beats the concise gold, there is nothing to teach and the driver
refuses to tune.

An **objective** compression proxy corroborates the judge — ``mean_compression``
(rewrite/draft word ratio; below 1.0 means the model shortened the draft) and
``shorter_rate``. Both the generator(s) and judge are injected so the logic is
unit-testable offline; a single generation pass per model feeds every metric.
"""

from __future__ import annotations

import random
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


def _completion_text(record: Record, score: int) -> str:
    for completion in record["completions"]:
        if completion["score"] == score:
            return completion["completion"]["parts"][0]["text"]
    msg = f"no completion with score {score}"
    raise ValueError(msg)


def _dispreferred_text(record: Record) -> str:
    return _completion_text(record, 0)


def _preferred_text(record: Record) -> str:
    return _completion_text(record, 1)


def _candidate_wins(
    draft: str,
    candidate: str,
    reference: str,
    judge_fn: Callable[[str, str, str], str],
    *,
    flip: bool,
) -> bool:
    """Blind A/B judge whether ``candidate`` beats ``reference``.

    ``flip`` places ``candidate`` in slot B (reference in A) so a judge that
    favors one slot cannot bias the aggregate; the winning letter is decoded back
    to "did the candidate win". A verdict for neither slot counts as not-a-win.
    """
    if flip:
        verdict = judge_fn(draft, reference, candidate).strip().upper()
        return verdict.startswith("B")
    verdict = judge_fn(draft, candidate, reference).strip().upper()
    return verdict.startswith("A")


def _flips(n: int, *, seed: int) -> list[bool]:
    """Deterministic per-item A/B position flips for blind judging."""
    rng = random.Random(seed)  # noqa: S311 - deterministic position balancing, not cryptographic
    return [rng.random() < 0.5 for _ in range(n)]  # noqa: PLR2004 - fair coin


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


def run_pilot_eval(
    records: Sequence[Record],
    generate_fn: Callable[[str], str],
    judge_fn: Callable[[str, str, str], str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Headroom gate: judge the base rewrite against the **gold preferred** ref.

    ``win_rate`` is the fraction of drafts where the base's own rewrite beats the
    concise human gold in a blind, position-randomized A/B. A high win-rate means
    the base is already as good as the gold (no headroom → don't tune); a low one
    means real room for tuning to close. Also returns the base ``mean_compression``
    and ``shorter_rate`` as an objective corroboration, and ``n``.
    """
    drafts = [_draft_text(r) for r in records]
    rewrites = [generate_fn(d) for d in drafts]
    golds = [_preferred_text(r) for r in records]
    flips = _flips(len(records), seed=seed)
    wins = sum(
        _candidate_wins(draft, rewrite, gold, judge_fn, flip=flip)
        for draft, rewrite, gold, flip in zip(drafts, rewrites, golds, flips, strict=True)
    )
    n = len(records)
    compression = score_compression(drafts, rewrites)
    return {
        "win_rate": wins / n if n else 0.0,
        "wins": wins,
        "n": n,
        "mean_compression": compression["mean_compression"],
        "shorter_rate": compression["shorter_rate"],
    }


def run_head_to_head_eval(
    records: Sequence[Record],
    base_generate_fn: Callable[[str], str],
    tuned_generate_fn: Callable[[str], str],
    judge_fn: Callable[[str, str, str], str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Before → after: judge the tuned rewrite against the **base** rewrite.

    The candidate is the tuned model, the reference is the base model, judged blind
    with a randomized A/B position (``seed``) so the aggregate is free of the
    judge's slot bias. ``win_rate`` (== ``hits`` / ``n``) above 0.5 means tuning
    helped; ``hits`` feeds ``bootstrap_ci``. ``base_mean_compression`` and
    ``tuned_mean_compression`` give the objective concision delta.
    """
    drafts = [_draft_text(r) for r in records]
    base_rewrites = [base_generate_fn(d) for d in drafts]
    tuned_rewrites = [tuned_generate_fn(d) for d in drafts]
    flips = _flips(len(records), seed=seed)
    wins = sum(
        _candidate_wins(draft, tuned, base, judge_fn, flip=flip)
        for draft, tuned, base, flip in zip(
            drafts, tuned_rewrites, base_rewrites, flips, strict=True
        )
    )
    n = len(records)
    base_compression = score_compression(drafts, base_rewrites)
    tuned_compression = score_compression(drafts, tuned_rewrites)
    return {
        "win_rate": wins / n if n else 0.0,
        "wins": wins,
        "hits": wins,
        "n": n,
        "base_mean_compression": base_compression["mean_compression"],
        "tuned_mean_compression": tuned_compression["mean_compression"],
    }
