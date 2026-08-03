"""Offline accuracy eval for an RLFT-tuned model.

RLFT has no gold completion, so we reuse the training reward: generate a reply per
held-out prompt, score it with :func:`geap_tuning.rlft.reward.evaluate` against the
record's ``references``, and report the fraction with a positive reward.
``generate_fn`` (the tuned-endpoint call) is injected so the logic is unit-testable
without live calls — the example driver binds it to
:func:`geap_tuning.inference.generate` on the tuned endpoint.

**The eval replays each record's own ``systemInstruction``.** RLFT records here are
trained under a system instruction (e.g. "end with the final answer on its own line
as 'Answer: <number>'") that carries the output contract the reward scores. Dropping
it at eval time makes the model answer in free prose and never emit the marker —
scoring 0 on the reward-based ``accuracy`` even when it solves every problem. So
``run_rlft_eval`` extracts each record's ``systemInstruction`` and passes it to
``generate_fn`` alongside the user text, keeping inference faithful to training.

Two metrics come out of one generation pass (the model call is the expensive part):

- ``accuracy`` — the **reward-based** score: correct *and* in the ``Answer: <n>``
  marker format the reward parser requires. This is the contract the reward trains.
- ``content_accuracy`` — a **marker-agnostic** score: does the ground-truth number
  appear anywhere in the reply? A format-only reward (e.g. string-match) can leave a
  model that solves the problem correctly but states the number in prose without the
  marker; ``accuracy`` scores that 0 while ``content_accuracy`` still credits the
  math. Reporting both separates "got the answer" from "honored the output contract".
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from geap_tuning.rlft.reward import evaluate as reward_evaluate
from geap_tuning.rlft.reward import normalize_number

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from geap_tuning.schemas import Record

# Any numeric token (optional sign, thousands commas, optional decimals). Used only
# by the marker-agnostic content check, not the reward (which parses "Answer: <n>").
_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def content_correct(text: str, ground_truth: str) -> bool:
    """Return ``True`` if ``ground_truth`` appears as a number anywhere in ``text``.

    Marker-agnostic complement to the reward-based check: scans every numeric token
    in the reply and compares its normalized form (see
    :func:`geap_tuning.rlft.reward.normalize_number`) to the ground truth, so a
    correct answer stated in prose (``"... is **55**."``) counts even without the
    ``Answer: <n>`` marker. Empty ``ground_truth`` is never a match.
    """
    if not ground_truth:
        return False
    target = normalize_number(ground_truth)
    return any(normalize_number(token) == target for token in _NUMBER_RE.findall(text))


def score_accuracy(rewards: Sequence[float]) -> dict[str, Any]:
    """Fraction of generations that earned a positive reward (plus raw counts)."""
    n = len(rewards)
    correct = sum(1 for reward in rewards if reward > 0)
    return {"accuracy": correct / n if n else 0.0, "correct": correct, "n": n}


def _record_system_instruction(record: Record) -> str | None:
    """Return the record's flat system-instruction text, or ``None`` if it has none.

    RLFT records built by :func:`geap_tuning.rlft.data.build_rlft_records` carry a
    ``systemInstruction`` (see :func:`geap_tuning.schemas.rlft_example`) shaped
    ``{"parts": [{"text": ...}]}``. Replaying it at eval time is what keeps
    inference faithful to training (see the module docstring).
    """
    parts = record.get("systemInstruction", {}).get("parts")
    if not parts:
        return None
    return parts[0].get("text")


def run_rlft_eval(
    records: Sequence[Record],
    generate_fn: Callable[[str, str | None], str],
) -> dict[str, Any]:
    """Generate one reply per prompt and score it two ways against ``references``.

    Returns the reward-based ``accuracy`` (plus ``correct``/``n`` from
    :func:`score_accuracy`) **and** the marker-agnostic ``content_accuracy`` (plus
    ``content_correct``), both from the same generation pass. The generated text is
    wrapped as a ``{"parts": [...]}`` Content dict so the reward sees the same shape
    it receives in the GEAP sandbox.

    ``generate_fn`` is called as ``generate_fn(user_text, system_instruction)`` — the
    per-record ``systemInstruction`` (or ``None``) is threaded through so the tuned
    model is prompted under the same framing it was trained on. The example driver
    binds it to a closure over :func:`geap_tuning.inference.generate` that forwards
    ``system_instruction=`` to the endpoint.
    """
    rewards: list[float] = []
    content_hits = 0
    for record in records:
        reply = generate_fn(
            record["contents"][0]["parts"][0]["text"],
            _record_system_instruction(record),
        )
        rewards.append(
            reward_evaluate(
                {"references": record["references"]},
                {"parts": [{"text": reply}]},
            )
        )
        content_hits += content_correct(reply, record["references"].get("ground_truth_answer", ""))
    metrics = score_accuracy(rewards)
    n = metrics["n"]
    metrics["content_accuracy"] = content_hits / n if n else 0.0
    metrics["content_correct"] = content_hits
    return metrics
