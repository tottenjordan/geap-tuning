"""Constrained text-generation dataset for the graded-reward RLFT demo.

Each example is a short writing prompt that spells out **four** kinds of
constraint at once — required keywords, forbidden filler words, a word-count
band, and a sentence-count band — paired with a ``references`` dict the graded
reward in :mod:`geap_tuning.rlft.constraint_reward` reads. Satisfying every
constraint simultaneously is hard even for a strong base (headroom), while the
*graded* reward gives partial credit so the training signal has variance — the
combination is what answers the two prior RLFT null results (a base that
saturates a verifiable task, and a binary reward with no gradient; see
``docs/doe/rlft-reward-ranking/README.md``).

The bank is generated deterministically (seeded) and is correct by construction:
every band is jointly satisfiable, and ``references`` values are all strings (the
RLFT record format requires it). Records reuse the shared ``rlft_example`` builder
in :mod:`geap_tuning.schemas`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from geap_tuning.schemas import rlft_example, write_jsonl

if TYPE_CHECKING:
    from geap_tuning.schemas import Record

NEUTRAL_SYSTEM_INSTRUCTION = "You are a precise writer who follows every instruction exactly."

# Writing topics, each with a description used in the prompt and a pool of on-topic
# keywords the model can naturally weave in. Keywords are lowercase so they appear
# verbatim in both the prompt and the (lowercased) reward keyword check.
_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "announcement for a product launch",
        ("launch", "roadmap", "release", "feature", "customers", "timeline"),
    ),
    (
        "invitation to a team offsite",
        ("offsite", "agenda", "travel", "workshop", "schedule", "team"),
    ),
    (
        "weekly project status update",
        ("status", "milestone", "blocker", "progress", "deadline", "update"),
    ),
    (
        "welcome note for a new hire",
        ("welcome", "onboarding", "mentor", "handbook", "workspace", "team"),
    ),
    (
        "summary of a customer feedback session",
        ("feedback", "session", "insights", "requests", "priorities", "themes"),
    ),
    (
        "reminder about an upcoming maintenance window",
        ("maintenance", "downtime", "window", "backup", "service", "schedule"),
    ),
)

# Filler words the model should avoid — none appear in the prompt scaffolding or in
# any topic keyword, so forbidding them never conflicts with a required keyword.
_FILLER = ("very", "really", "basically", "actually", "just", "stuff")

# (min_words, max_words) bands — wide enough to be jointly satisfiable with the
# sentence bands below.
_WORD_BANDS = ((40, 90), (50, 100), (60, 110))
# (min_sentences, max_sentences) bands.
_SENTENCE_BANDS = ((3, 6), (4, 7), (3, 5))

_N = 150


class ConstraintSpec(NamedTuple):
    """A writing prompt and the string-valued ``references`` the reward reads."""

    prompt: str
    references: dict[str, str]


def _difficulty(n_keywords: int) -> str:
    if n_keywords <= 2:  # noqa: PLR2004 - inline tier thresholds
        return "easy"
    if n_keywords == 3:  # noqa: PLR2004 - inline tier thresholds
        return "medium"
    return "hard"


def _build_bank(*, seed: int = 20240517, n: int = _N) -> list[ConstraintSpec]:
    """Deterministically generate ``n`` unique, jointly-satisfiable constraint specs."""
    rng = random.Random(seed)  # noqa: S311 - deterministic bank, not cryptographic
    specs: list[ConstraintSpec] = []
    seen: set[str] = set()
    while len(specs) < n:
        description, pool = _TOPICS[len(specs) % len(_TOPICS)]
        n_keywords = rng.randint(2, 4)
        keywords = rng.sample(pool, n_keywords)
        n_forbidden = rng.randint(1, 3)
        forbidden = rng.sample(_FILLER, n_forbidden)
        min_words, max_words = rng.choice(_WORD_BANDS)
        min_sentences, max_sentences = rng.choice(_SENTENCE_BANDS)
        prompt = (
            f"Write a short {description}. "
            f"It must include these words: {', '.join(keywords)}. "
            f"Do not use these words: {', '.join(forbidden)}. "
            f"Keep it between {min_words} and {max_words} words and between "
            f"{min_sentences} and {max_sentences} sentences."
        )
        if prompt in seen:
            continue
        seen.add(prompt)
        references = {
            "required_keywords": ",".join(keywords),
            "forbidden_words": ",".join(forbidden),
            "min_words": str(min_words),
            "max_words": str(max_words),
            "min_sentences": str(min_sentences),
            "max_sentences": str(max_sentences),
            "difficulty": _difficulty(n_keywords),
        }
        specs.append(ConstraintSpec(prompt=prompt, references=references))
    return specs


CONSTRAINT_SPECS: list[ConstraintSpec] = _build_bank()


def build_records(specs: list[ConstraintSpec]) -> list[Record]:
    """Turn constraint specs into RLFT ``contents`` records (user turn + references)."""
    return [
        rlft_example(
            user_text=spec.prompt,
            references=spec.references,
            system_instruction=NEUTRAL_SYSTEM_INSTRUCTION,
        )
        for spec in specs
    ]


def split_dataset(
    specs: list[ConstraintSpec],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> tuple[list[ConstraintSpec], list[ConstraintSpec], list[ConstraintSpec]]:
    """Deterministically shuffle and split into (train, val, test).

    The default ``0.2`` test ratio yields ~30 held-out specs — enough for a
    bootstrap confidence interval on the full-satisfaction rate to be meaningful.
    """
    shuffled = list(specs)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - deterministic split, not cryptographic
    total = len(shuffled)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_constrained_dataset(out_dir: str | Path) -> dict[str, str]:
    """Write train/val/test JSONL under ``out_dir``; return a name->path mapping."""
    out_dir = Path(out_dir)
    train, val, test = split_dataset(CONSTRAINT_SPECS)
    paths: dict[str, str] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(build_records(split), path)
        paths[name] = str(path)
    return paths
