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

For the reward-*ranking* DOE (``examples/run_doe_rlft_reward_ranking.py``) a single
headline is not enough — each reward shape optimizes a different objective, so
:func:`run_rlft_multimetric_eval` scores several independent axes (``correctness``,
``format_rate``, ``marker_accuracy``, optional ``explanation_quality``, and a
per-difficulty breakdown) from one generation pass, and :func:`bootstrap_ci` gives
each a confidence interval so "best shape" is reported with significance, not as a
coin-flip tie-break.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

from geap_tuning.rlft.reward import evaluate as reward_evaluate
from geap_tuning.rlft.reward import extract_answer, normalize_number

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
    format_hits = 0
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
        format_hits += extract_answer(reply) is not None
    metrics = score_accuracy(rewards)
    n = metrics["n"]
    metrics["content_accuracy"] = content_hits / n if n else 0.0
    metrics["content_correct"] = content_hits
    metrics["format_rate"] = format_hits / n if n else 0.0
    metrics["format_count"] = format_hits
    return metrics


def bootstrap_ci(
    hits: int,
    n: int,
    *,
    seed: int = 0,
    resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a bootstrap ``(low, high)`` CI for a proportion ``hits / n``.

    Reconstructs the per-item 0/1 outcome vector (``hits`` ones, ``n - hits``
    zeros), resamples it with replacement ``resamples`` times, and returns the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of the resampled means — the
    standard nonparametric CI for a binary metric. stdlib only and seeded, so it
    is deterministic. An empty test set yields ``(0.0, 0.0)``.
    """
    if n <= 0:
        return (0.0, 0.0)
    population = [1] * hits + [0] * (n - hits)
    rng = random.Random(seed)  # noqa: S311 - deterministic resampling, not cryptographic
    means = sorted(sum(rng.choice(population) for _ in range(n)) / n for _ in range(resamples))
    low_index = max(0, int((alpha / 2) * resamples))
    high_index = min(resamples - 1, int((1 - alpha / 2) * resamples))
    return (means[low_index], means[high_index])


def run_rlft_multimetric_eval(
    records: Sequence[Record],
    generate_fn: Callable[[str, str | None], str],
    *,
    judge_fn: Callable[[str, str, str], float] | None = None,
) -> dict[str, Any]:
    """Score a tuned model on several *independent* axes from one generation pass.

    A reward-shape sweep only ranks when each shape's objective has its own axis.
    Generating once per record (replaying its ``systemInstruction`` for train/eval
    parity), this returns:

    - ``correctness`` — marker-agnostic: the right number appears anywhere in the
      reply (:func:`content_correct`). The primary rank axis.
    - ``format_rate`` — fraction whose reply carries a parseable ``Answer: <n>``
      marker (:func:`geap_tuning.rlft.reward.extract_answer`), *regardless* of
      correctness. The axis the format-only ``string-match`` reward targets.
    - ``marker_accuracy`` — the reward-based score (correct **and** marker present),
      i.e. the old ``run_rlft_eval`` ``accuracy``.
    - ``explanation_quality`` — mean of ``judge_fn(question, reply, ground_truth)``
      over 0..1 when a ``judge_fn`` is given (else omitted). The judge **must** be a
      different model/prompt than any training autorater reward, or it grades with
      the trainer.
    - ``by_difficulty`` — per-tier ``{correctness, n}`` (tier read from each
      record's ``references['difficulty']`` when present), so a good reward's lift
      can be seen where there is headroom.

    Also returns raw ``n`` and the ``*_hits`` counts (handy for :func:`bootstrap_ci`).
    """
    n = len(records)
    content_hits = 0
    format_hits = 0
    marker_hits = 0
    quality_total = 0.0
    by_difficulty: dict[str, dict[str, Any]] = {}
    for record in records:
        question = record["contents"][0]["parts"][0]["text"]
        reply = generate_fn(question, _record_system_instruction(record))
        truth = record["references"].get("ground_truth_answer", "")
        is_correct = content_correct(reply, truth)
        content_hits += is_correct
        format_hits += extract_answer(reply) is not None
        marker_hits += (
            reward_evaluate({"references": record["references"]}, {"parts": [{"text": reply}]}) > 0
        )
        if judge_fn is not None:
            quality_total += judge_fn(question, reply, truth)
        tier = record["references"].get("difficulty", "unknown")
        bucket = by_difficulty.setdefault(tier, {"correct": 0, "n": 0})
        bucket["correct"] += is_correct
        bucket["n"] += 1

    metrics: dict[str, Any] = {
        "n": n,
        "correctness": content_hits / n if n else 0.0,
        "content_hits": content_hits,
        "format_rate": format_hits / n if n else 0.0,
        "format_hits": format_hits,
        "marker_accuracy": marker_hits / n if n else 0.0,
        "marker_hits": marker_hits,
        "by_difficulty": {
            tier: {
                "correctness": bucket["correct"] / bucket["n"] if bucket["n"] else 0.0,
                "n": bucket["n"],
            }
            for tier, bucket in by_difficulty.items()
        },
    }
    if judge_fn is not None:
        metrics["explanation_quality"] = quality_total / n if n else 0.0
    return metrics
