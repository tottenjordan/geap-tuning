"""Harder, difficulty-tiered verifiable-math bank for the reward-*ranking* DOE.

This is the sibling of :mod:`geap_tuning.rlft.data`, built to fix the flat null
result that dataset produced (every reward shape *and* the untuned baseline scored
1.000 — see ``docs/doe/rlft-reward-shapes/README.md``). Three deliberate changes
give a reward-shape sweep something to measure:

1. **Headroom** — problems are multi-step and split across ``easy``/``medium``/
   ``hard`` tiers, so a *weaker* base model does **not** saturate the task.
2. **A larger held-out set** — a ~150-problem bank with a stratified split yields
   a test set of ~30, enough for a bootstrap confidence interval to mean something.
3. **Format headroom** — :data:`NEUTRAL_SYSTEM_INSTRUCTION` deliberately **omits**
   the ``Answer: <number>`` contract that :data:`geap_tuning.rlft.data.SYSTEM_INSTRUCTION`
   bakes in. With the marker no longer handed to the model for free, the
   format-only ``string-match`` reward finally has something real to teach, so it
   can diverge from the correctness rewards on the ``format_rate`` axis.

Every answer is **computed in Python** at bank-construction time (not hand-typed),
so the ground truth the reward compares against is correct by construction. The
bank is generated deterministically from a fixed seed, so :data:`HARD_MATH_PROBLEMS`
is stable across runs and processes.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from geap_tuning.schemas import rlft_example, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Callable

    from geap_tuning.schemas import Record

NEUTRAL_SYSTEM_INSTRUCTION = (
    "You are a careful math tutor. Think step by step and show your reasoning, "
    "then state the final answer."
)
"""A framing with **no** output-format contract.

Contrast with :data:`geap_tuning.rlft.data.SYSTEM_INSTRUCTION`, which instructs the
model to end with ``Answer: <number>``. Dropping that marker is load-bearing for
the ranking DOE: it creates *format* headroom so the ``string-match`` reward (which
trains the marker) can be measured against rewards that train correctness instead.
"""

DIFFICULTIES = ("easy", "medium", "hard")


class MathProblem(NamedTuple):
    """A single tiered word problem: the prompt, its numeric answer, and its tier."""

    question: str
    answer: str
    difficulty: str


def _fmt(value: float) -> str:
    """Format a computed answer as a canonical numeric string.

    Integers render without a decimal point; non-integers round to two decimals
    with trailing zeros stripped. This matches what
    :func:`geap_tuning.rlft.reward.normalize_number` reconciles at scoring time.
    """
    rounded = round(value, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


# --- Per-tier problem templates -------------------------------------------------
# Each template draws its parameters from a seeded RNG and RETURNS the exact answer
# alongside the prompt, so ground truth is correct by construction (never typed).
# Numbers are chosen so results are integers or clean two-decimal values.


def _easy_boxes(rng: random.Random) -> tuple[str, float]:
    shelves = rng.randint(4, 12)
    per_shelf = rng.randint(6, 15)
    removed = rng.randint(1, shelves * per_shelf // 2)
    question = (
        f"A warehouse has {shelves} shelves holding {per_shelf} boxes each. "
        f"If {removed} boxes are removed, how many boxes are left?"
    )
    return question, shelves * per_shelf - removed


def _easy_percent_plus(rng: random.Random) -> tuple[str, float]:
    percent = rng.choice([10, 20, 25, 40, 50, 60, 75])
    base = rng.choice([80, 120, 160, 200, 240, 320])
    extra = rng.randint(5, 40)
    question = f"What is {percent}% of {base}, plus {extra}?"
    return question, base * percent / 100 + extra


def _easy_wage(rng: random.Random) -> tuple[str, float]:
    rate = rng.randint(12, 30)
    hours = rng.randint(5, 20)
    fee = rng.randint(5, 40)
    question = (
        f"A worker earns ${rate} per hour and works {hours} hours, then pays a "
        f"${fee} fee. What are the net earnings in dollars?"
    )
    return question, rate * hours - fee


def _medium_tank(rng: random.Random) -> tuple[str, float]:
    capacity = rng.choice([200, 300, 400, 500, 600, 800])
    percent = rng.choice([10, 20, 25, 40, 50, 60, 75])
    added = rng.randint(10, 60)
    question = (
        f"A tank holds {capacity} liters and is {percent}% full. After {added} "
        f"more liters are added, how many liters are in it?"
    )
    return question, capacity * percent / 100 + added


def _medium_trip(rng: random.Random) -> tuple[str, float]:
    speed = rng.randint(40, 90)
    hours = rng.randint(2, 5)
    extra = rng.randint(15, 120)
    question = (
        f"A car travels at {speed} km/h for {hours} hours, rests, then drives "
        f"{extra} km more. What is the total distance in km?"
    )
    return question, speed * hours + extra


def _medium_discount_ship(rng: random.Random) -> tuple[str, float]:
    price = rng.choice([80, 120, 160, 200, 240, 400])
    percent = rng.choice([10, 20, 25, 50, 75])
    shipping = rng.randint(5, 30)
    question = (
        f"A ${price} item is discounted {percent}%, then ${shipping} shipping is "
        f"added. What is the total cost in dollars?"
    )
    return question, price * (1 - percent / 100) + shipping


def _hard_discount_tax(rng: random.Random) -> tuple[str, float]:
    price = rng.choice([120, 150, 180, 240, 320, 450])
    discount = rng.choice([10, 15, 20, 25])
    tax = rng.choice([5, 8, 10])
    question = (
        f"A ${price} item is discounted {discount}%, then a {tax}% tax is applied "
        f"to the discounted price. What is the final price in dollars (2 decimals)?"
    )
    return question, price * (1 - discount / 100) * (1 + tax / 100)


def _hard_combined_average(rng: random.Random) -> tuple[str, float]:
    n_a = rng.randint(10, 30)
    avg_a = rng.randint(60, 90)
    n_b = rng.randint(10, 30)
    avg_b = rng.randint(60, 90)
    question = (
        f"Class A has {n_a} students averaging {avg_a} points; class B has {n_b} "
        f"students averaging {avg_b} points. What is the combined average of all "
        f"students (round to 2 decimals)?"
    )
    return question, (n_a * avg_a + n_b * avg_b) / (n_a + n_b)


def _hard_compound_growth(rng: random.Random) -> tuple[str, float]:
    population = rng.choice([1000, 1200, 1500, 2000, 2500, 4000])
    percent = rng.choice([5, 10, 20, 25])
    question = (
        f"A town of {population} people grows {percent}% each year. What is its "
        f"population after 2 years (round to the nearest whole number)?"
    )
    return question, round(population * (1 + percent / 100) ** 2)


_TEMPLATES: dict[str, tuple[Callable[[random.Random], tuple[str, float]], ...]] = {
    "easy": (_easy_boxes, _easy_percent_plus, _easy_wage),
    "medium": (_medium_tank, _medium_trip, _medium_discount_ship),
    "hard": (_hard_discount_tax, _hard_combined_average, _hard_compound_growth),
}

_PER_TIER = 50


def _build_bank(*, seed: int = 20240517, per_tier: int = _PER_TIER) -> list[MathProblem]:
    """Deterministically generate a balanced, de-duplicated tiered problem bank.

    Round-robins each tier's templates, drawing parameters from a seeded RNG and
    keeping the first ``per_tier`` *unique* questions per tier. Because answers are
    computed here (never typed), the bank is correct by construction; because the
    RNG is seeded, it is identical across runs and processes.
    """
    rng = random.Random(seed)  # noqa: S311 - deterministic bank, not cryptographic
    problems: list[MathProblem] = []
    for difficulty in DIFFICULTIES:
        templates = _TEMPLATES[difficulty]
        seen: set[str] = set()
        index = 0
        while len(seen) < per_tier:
            template = templates[index % len(templates)]
            index += 1
            question, answer = template(rng)
            if question in seen:
                continue
            seen.add(question)
            problems.append(MathProblem(question, _fmt(answer), difficulty))
    return problems


HARD_MATH_PROBLEMS: list[MathProblem] = _build_bank()
"""The generated tiered bank: :data:`_PER_TIER` problems per tier (easy/medium/hard)."""


def build_bench_records(
    problems: list[MathProblem],
    *,
    system_instruction: str = NEUTRAL_SYSTEM_INSTRUCTION,
) -> list[Record]:
    """Turn :class:`MathProblem` items into RLFT records under a given framing.

    Like :func:`geap_tuning.rlft.data.build_rlft_records`, but (a) takes the system
    instruction as a parameter (defaulting to the marker-free
    :data:`NEUTRAL_SYSTEM_INSTRUCTION`) and (b) stashes each problem's
    ``difficulty`` in ``references`` alongside ``ground_truth_answer``. The extra
    key is harmless to :func:`geap_tuning.rlft.reward.evaluate` (which reads only
    ``ground_truth_answer``) and lets the eval break correctness down by tier.
    """
    return [
        rlft_example(
            user_text=problem.question,
            references={
                "ground_truth_answer": problem.answer,
                "difficulty": problem.difficulty,
            },
            system_instruction=system_instruction,
        )
        for problem in problems
    ]


def split_stratified(
    problems: list[MathProblem],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> tuple[list[MathProblem], list[MathProblem], list[MathProblem]]:
    """Split into (train, val, test) with each difficulty tier balanced per split.

    Unlike :func:`geap_tuning.rlft.data.split_dataset` (a single global shuffle),
    this shuffles *within* each tier and applies ``ratios`` per tier, so every
    split keeps the same easy/medium/hard mix. Deterministic for a given ``seed``.
    """
    rng = random.Random(seed)  # noqa: S311 - deterministic split, not cryptographic
    train: list[MathProblem] = []
    val: list[MathProblem] = []
    test: list[MathProblem] = []
    for difficulty in DIFFICULTIES:
        tier = [problem for problem in problems if problem.difficulty == difficulty]
        rng.shuffle(tier)
        total = len(tier)
        n_train = int(total * ratios[0])
        n_val = int(total * ratios[1])
        train.extend(tier[:n_train])
        val.extend(tier[n_train : n_train + n_val])
        test.extend(tier[n_train + n_val :])
    return train, val, test


def build_bench_dataset(
    out_dir: str | Path,
    *,
    system_instruction: str = NEUTRAL_SYSTEM_INSTRUCTION,
) -> dict[str, str]:
    """Write train/val/test JSONL for the ranking bench; return a name->path map.

    Uses the stratified split so every file keeps a balanced tier mix.
    """
    out_dir = Path(out_dir)
    train, val, test = split_stratified(HARD_MATH_PROBLEMS)
    paths: dict[str, str] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(build_bench_records(split, system_instruction=system_instruction), path)
        paths[name] = str(path)
    return paths
