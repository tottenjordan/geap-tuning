"""Offline accuracy eval for an RLFT-tuned model.

RLFT has no gold completion, so we reuse the training reward: generate a reply per
held-out prompt, score it with :func:`geap_tuning.rlft.reward.evaluate` against the
record's ``references``, and report the fraction with a positive reward.
``generate_fn`` (the tuned-endpoint call) is injected so the logic is unit-testable
without live calls — the example driver binds it to
:func:`geap_tuning.inference.generate` on the tuned endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geap_tuning.rlft.reward import evaluate as reward_evaluate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from geap_tuning.schemas import Record


def score_accuracy(rewards: Sequence[float]) -> dict[str, Any]:
    """Fraction of generations that earned a positive reward (plus raw counts)."""
    n = len(rewards)
    correct = sum(1 for reward in rewards if reward > 0)
    return {"accuracy": correct / n if n else 0.0, "correct": correct, "n": n}


def run_rlft_eval(
    records: Sequence[Record],
    generate_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Generate a reply per prompt and reward-score it against ``references``.

    The generated text is wrapped as a ``{"parts": [...]}`` Content dict so the
    reward sees the same shape it receives in the GEAP sandbox.
    """
    rewards = [
        reward_evaluate(
            {"references": record["references"]},
            {"parts": [{"text": generate_fn(record["contents"][0]["parts"][0]["text"])}]},
        )
        for record in records
    ]
    return score_accuracy(rewards)
