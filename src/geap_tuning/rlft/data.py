"""Verifiable-math dataset for the RLFT example.

Where SFT teaches *what* to answer and DPO teaches *how* to phrase it, RLFT
rewards *correctness*: the model generates a full reasoning trace and is scored
by a reward function that checks the final answer against ground truth. Each
record is a grade-school word problem whose unambiguous numeric answer is stored
in ``references={"ground_truth_answer": "<n>"}`` — there is no gold model turn.

The dataset is small, self-contained, and text-only. Records use the shared RLFT
builder in :mod:`geap_tuning.schemas`; the answers are parsed at reward time by
:func:`geap_tuning.rlft.reward.extract_answer`, so every question nudges the
"Answer: <number>" format that reward expects.
"""

from __future__ import annotations

import random
from pathlib import Path

from geap_tuning.schemas import Record, rlft_example, write_jsonl

SYSTEM_INSTRUCTION = (
    "You are a careful math tutor. Think step by step, then end with the final "
    "answer on its own line as 'Answer: <number>'."
)

# (question, ground_truth_answer) pairs. Answers are strings with unambiguous
# integer/decimal values; the reward normalizes commas and trailing zeros.
MATH_PROBLEMS: list[tuple[str, str]] = [
    ("What is 17 * 23? End with 'Answer: <number>'.", "391"),
    (
        "A store sells pencils at 3 for $1.20. How much do 12 pencils cost in dollars?",
        "4.80",
    ),
    (
        (
            "A train travels 60 miles in 45 minutes. How many miles does it travel in 3 "
            "hours at the same speed?"
        ),
        "240",
    ),
    ("If a rectangle is 8 cm wide and 14 cm long, what is its area in square cm?", "112"),
    (
        (
            "Maria had 45 apples. She sold 18 and then bought 30 more. How many apples "
            "does she have now?"
        ),
        "57",
    ),
    ("What is 15% of 240?", "36"),
    (
        "A recipe needs 2.5 cups of flour per loaf. How many cups are needed for 6 loaves?",
        "15",
    ),
    (
        "Tom reads 25 pages a day. How many pages will he read in 2 weeks?",
        "350",
    ),
    ("What is the sum of the first 10 positive integers?", "55"),
    (
        "A car uses 8 liters of fuel per 100 km. How many liters for a 350 km trip?",
        "28",
    ),
    (
        "A jacket costs $80 and is discounted by 25%. What is the sale price in dollars?",
        "60",
    ),
    (
        "There are 7 boxes with 24 bottles each. How many bottles are there in total?",
        "168",
    ),
    ("What is 144 divided by 12?", "12"),
    (
        (
            "A worker earns $18 per hour and works 37.5 hours in a week. What are the "
            "weekly earnings in dollars?"
        ),
        "675",
    ),
    (
        "A tank holds 500 liters and is 40% full. How many liters of water are in it?",
        "200",
    ),
    ("If 5 notebooks cost $32.50, how much does one notebook cost in dollars?", "6.50"),
    (
        "A garden has 6 rows of 15 tomato plants. If 12 plants die, how many remain?",
        "78",
    ),
    ("What is 2 to the power of 8?", "256"),
    (
        "A phone plan costs $35 a month. How much is that per year in dollars?",
        "420",
    ),
    (
        "Sara ran 3.2 km on Monday and twice that on Tuesday. How many km did she run in total?",
        "9.6",
    ),
    (
        "A theater has 28 rows with 32 seats each. How many seats are there in total?",
        "896",
    ),
    ("What is 1000 minus 347?", "653"),
    (
        "A recipe serves 4 and uses 6 eggs. How many eggs are needed to serve 10 people?",
        "15",
    ),
    (
        (
            "A bike shop repaired 9 bikes on average per day for 6 days. How many bikes "
            "were repaired in total?"
        ),
        "54",
    ),
    ("What is 23 * 6?", "138"),
    ("What is 30% of 90?", "27"),
    (
        "A book has 320 pages. If Lena reads 40 pages a day, how many days to finish it?",
        "8",
    ),
    (
        "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices remain?",
        "2",
    ),
    ("What is the average of 12, 18, and 30?", "20"),
    (
        "A shirt costs $24 and is marked up by 50%. What is the new price in dollars?",
        "36",
    ),
    (
        "A bus holds 52 passengers. How many passengers can 4 full buses carry?",
        "208",
    ),
    ("What is 7 squared minus 9?", "40"),
    (
        "A farmer has 96 eggs and packs them into cartons of 12. How many cartons?",
        "8",
    ),
    (
        "Water flows at 15 liters per minute. How many liters flow in 12 minutes?",
        "180",
    ),
    (
        "A movie starts at 7:15 pm and lasts 130 minutes. What time does it end (24-hour, HHMM)?",
        "2125",
    ),
    ("What is 3/4 of 64?", "48"),
    (
        "A team scored 3, 5, and 7 points in three games. What was their total score?",
        "15",
    ),
    (
        "A $1200 laptop is paid in 6 equal monthly installments. How much is each in dollars?",
        "200",
    ),
    (
        "A rope is 18 meters long and cut into 3 equal pieces. How long is each in meters?",
        "6",
    ),
    ("What is 45 + 67 - 12?", "100"),
]


def build_rlft_records(problems: list[tuple[str, str]]) -> list[Record]:
    """Turn ``(question, ground_truth_answer)`` pairs into RLFT records."""
    return [
        rlft_example(
            user_text=question,
            references={"ground_truth_answer": answer},
            system_instruction=SYSTEM_INSTRUCTION,
        )
        for question, answer in problems
    ]


def split_dataset(
    problems: list[tuple[str, str]],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """Deterministically shuffle and split ``problems`` into (train, val, test)."""
    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - deterministic split, not cryptographic
    total = len(shuffled)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_rlft_dataset(out_dir: str | Path) -> dict[str, str]:
    """Write train/val/test JSONL under ``out_dir``; return a name->path mapping."""
    out_dir = Path(out_dir)
    train, val, test = split_dataset(MATH_PROBLEMS)
    paths: dict[str, str] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(build_rlft_records(split), path)
        paths[name] = str(path)
    return paths
