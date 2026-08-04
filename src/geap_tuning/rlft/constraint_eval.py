"""Offline multimetric eval for the constrained-generation RLFT model.

RLFT has no gold completion, so we reuse the training reward: generate one reply
per held-out prompt (replaying the record's ``systemInstruction`` for train/eval
parity, exactly like :func:`geap_tuning.rlft.evaluate.run_rlft_eval`), score it
with the graded :func:`geap_tuning.rlft.constraint_reward.evaluate`, and report
several axes from that single pass:

- ``accuracy`` — the mean graded reward (fraction of constraint components
  satisfied). The headline, so it plugs into ``"accuracy"``-keyed machinery.
- ``full_satisfaction_rate`` — fraction of replies that satisfied **every**
  component (reward ``>= 1.0``); pair it with
  :func:`geap_tuning.rlft.evaluate.bootstrap_ci` for a confidence interval.
- ``by_constraint_type`` — per-type ``{rate, satisfied, total}`` from
  :func:`geap_tuning.rlft.constraint_reward.component_breakdown`, so you can see
  *which* constraint kind the tuning improved.

``generate_fn`` is injected (called as ``generate_fn(user_text, system_instruction)``)
so the logic is unit-testable offline; the driver binds it to a closure over
:func:`geap_tuning.inference.generate` on the tuned endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geap_tuning.rlft.constraint_reward import component_breakdown
from geap_tuning.rlft.constraint_reward import evaluate as reward_evaluate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from geap_tuning.schemas import Record

_FULL_REWARD = 1.0


def _record_system_instruction(record: Record) -> str | None:
    parts = record.get("systemInstruction", {}).get("parts")
    if not parts:
        return None
    return parts[0].get("text")


def run_eval(
    records: Sequence[Record],
    generate_fn: Callable[[str, str | None], str],
) -> dict[str, Any]:
    """Generate one reply per prompt and score graded constraint satisfaction.

    Returns ``accuracy`` (mean graded reward), ``full_satisfaction_rate`` (plus
    ``full_satisfaction_hits``), ``by_constraint_type``, and raw ``n`` — all from a
    single generation pass. The reply is wrapped as a ``{"parts": [...]}`` Content
    dict so the reward sees the same shape it gets in the GEAP sandbox.
    """
    n = len(records)
    reward_total = 0.0
    full_hits = 0
    type_satisfied: dict[str, int] = {}
    type_total: dict[str, int] = {}
    for record in records:
        reply = generate_fn(
            record["contents"][0]["parts"][0]["text"],
            _record_system_instruction(record),
        )
        response = {"parts": [{"text": reply}]}
        references = record["references"]
        reward = reward_evaluate({"references": references}, response)
        reward_total += reward
        if reward >= _FULL_REWARD:
            full_hits += 1
        for kind, (satisfied, total) in component_breakdown(references, response).items():
            type_satisfied[kind] = type_satisfied.get(kind, 0) + satisfied
            type_total[kind] = type_total.get(kind, 0) + total
    by_constraint_type = {
        kind: {
            "rate": type_satisfied[kind] / total if total else 0.0,
            "satisfied": type_satisfied[kind],
            "total": total,
        }
        for kind, total in type_total.items()
    }
    return {
        "accuracy": reward_total / n if n else 0.0,
        "full_satisfaction_rate": full_hits / n if n else 0.0,
        "full_satisfaction_hits": full_hits,
        "by_constraint_type": by_constraint_type,
        "n": n,
    }
