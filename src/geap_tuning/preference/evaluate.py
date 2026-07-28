"""Evaluate a DPO-tuned model with an autorater pairwise win-rate.

DPO has no single gold answer, so accuracy doesn't apply. Instead we measure
whether the tuned model's reply beats the dispreferred reference in a blind A/B
judgment. Both the generator (``generate_fn``) and the judge (``judge_fn``) are
injected so the logic is unit-testable without live calls — the example driver
binds ``generate_fn`` to :func:`geap_tuning.inference.generate` on the tuned
endpoint and ``judge_fn`` to a base-model autorater.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from geap_tuning.schemas import Record


def score_winrate(
    judgments: Sequence[str],
    *,
    win_label: str = "A",
    lose_label: str = "B",
) -> dict[str, Any]:
    """Summarize A/B judgments for the tuned reply (candidate ``win_label``).

    Returns ``win_rate`` plus raw ``wins``/``losses``/``ties``/``n`` counts (a
    judgment that starts with neither label counts as a tie). The three counts
    always sum to ``n``.
    """
    n = len(judgments)
    verdicts = [j.strip().upper() for j in judgments]
    wins = sum(1 for v in verdicts if v.startswith(win_label.upper()))
    losses = sum(1 for v in verdicts if v.startswith(lose_label.upper()))
    ties = n - wins - losses
    return {
        "win_rate": wins / n if n else 0.0,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n": n,
    }


def _user_text(record: Record) -> str:
    return record["contents"][0]["parts"][0]["text"]


def _completion_text(record: Record, score: int) -> str:
    for completion in record["completions"]:
        if completion["score"] == score:
            return completion["completion"]["parts"][0]["text"]
    msg = f"no completion with score {score}"
    raise ValueError(msg)


def run_preference_eval(
    records: Sequence[Record],
    generate_fn: Callable[[str], str],
    judge_fn: Callable[[str, str, str], str],
) -> dict[str, Any]:
    """Generate a reply per prompt and A/B-judge it against the dispreferred ref.

    ``judge_fn(user_text, candidate_a, candidate_b)`` returns ``"A"``/``"B"``;
    candidate A is the tuned reply and candidate B is the dispreferred reference,
    so an ``"A"`` verdict counts as a win for the tuned model.
    """
    judgments = [
        judge_fn(_user_text(record), generate_fn(_user_text(record)), _completion_text(record, 0))
        for record in records
    ]
    return score_winrate(judgments)
