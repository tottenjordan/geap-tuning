"""Structured JSON-extraction dataset for the generative SFT example.

A programmatic, correct-by-construction dataset: each example is a messy,
natural-language order line paired with the strict JSON object it should extract
to. Generation is deterministic (seeded) and needs no network or extra
dependency. The records reuse the shared ``contents``-format builders in
:mod:`geap_tuning.schemas`.

Unlike the classification datasets in this package, this is a *generative* task —
and, deliberately, one that teaches a **house normalization standard the base
model cannot guess**. Plain field extraction saturates a modern base (it scored a
perfect ``accuracy`` in a live run, leaving no headroom). So the gold object here
is not the raw text: each field is normalized by an internal convention that is
signalled — but *not spelled out* — in :data:`SYSTEM_INSTRUCTION`:

* ``order_id`` — strip the ``ord-`` prefix and upper-case it (``ord-a1234`` → ``A1234``).
* ``quantity`` — an integer, mapping spelled-out counts (``a dozen`` → ``12``).
* ``city`` — expand the shipping abbreviation to its canonical name (``NYC`` → ``New York``).
* ``priority`` — map the urgency word to our P-code scale (``urgent`` → ``P0`` … ``low`` → ``P3``).

An untuned base, told only to "apply our internal normalization standard", cannot
know the arbitrary P-code scale or which abbreviations to expand, so it misses
those fields — real, measurable headroom. SFT teaches the standard from the
labeled examples, which is exactly what supervised fine-tuning is for.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

from geap_tuning.schemas import Record, sft_example, write_jsonl

SCHEMA_FIELDS: tuple[str, ...] = ("order_id", "item", "quantity", "city", "priority")

SYSTEM_INSTRUCTION = (
    "Extract the order into a JSON object with EXACTLY these keys: "
    "order_id, item, quantity, city, priority. "
    "Apply our internal normalization standard for every field. "
    "Respond with only the JSON object, no prose, no code fences."
)

_ITEMS = (
    "wireless mouse",
    "mechanical keyboard",
    "usb-c cable",
    "laptop stand",
    "noise-cancelling headphones",
    "webcam",
    "monitor arm",
    "desk lamp",
    "ergonomic chair",
    "phone charger",
    "hdmi adapter",
    "portable ssd",
    "bluetooth speaker",
    "standing desk",
    "cable organizer",
)

# The house convention the base cannot guess: expand the shipping abbreviation used
# in the raw line to the canonical city name in the gold object.
_CITY_MAP: dict[str, str] = {
    "NYC": "New York",
    "LA": "Los Angeles",
    "SF": "San Francisco",
    "ATX": "Austin",
    "PHL": "Philadelphia",
    "CHI": "Chicago",
    "SEA": "Seattle",
    "DEN": "Denver",
    "BOS": "Boston",
    "PDX": "Portland",
}

# Map the urgency word to our arbitrary P-code scale (the clearest base-unknown field).
_PRIORITY_MAP: dict[str, str] = {
    "urgent": "P0",
    "high": "P1",
    "normal": "P2",
    "low": "P3",
}

# Quantity tokens as they appear in the raw line, paired with the integer gold value.
# A mix of digits (base usually gets these) and spelled-out counts.
_QUANTITY_TOKENS: tuple[tuple[str, int], ...] = (
    ("1", 1),
    ("2", 2),
    ("3", 3),
    ("5", 5),
    ("8", 8),
    ("a couple", 2),
    ("a pair", 2),
    ("a few", 3),
    ("half a dozen", 6),
    ("a dozen", 12),
)

# Messy, natural-language templates round-robined over the value pools. Each names
# every field in a different order / phrasing so the model must actually parse. The
# raw tokens (ord- id, abbreviation, urgency word, count token) are what the base
# sees; the gold object carries the normalized values.
_TEMPLATES = (
    "hey can u rush {order_id} — {quantity}x {item} to {city}, mark it {priority} thanks!",
    "{order_id}: please send {quantity} {item} over to {city} ({priority} priority)",
    "need {quantity} {item} shipped to {city} asap, ref {order_id}, priority is {priority}",
    "{priority} request!! {order_id} wants {quantity} {item} delivered in {city} pls",
)


class ExtractionExample(NamedTuple):
    """A messy input line and the strict, normalized JSON object it extracts to."""

    line: str
    target_json: str


def _build_bank(*, seed: int = 20240517, n: int = 200) -> list[ExtractionExample]:
    """Deterministically generate ``n`` unique extraction examples."""
    rng = random.Random(seed)  # noqa: S311 - deterministic data gen, not cryptographic
    examples: list[ExtractionExample] = []
    seen: set[str] = set()
    abbreviations = list(_CITY_MAP)
    priority_words = list(_PRIORITY_MAP)
    # Generate generously, then dedup by line and keep the first ``n`` unique.
    while len(examples) < n:
        letter = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        digits = f"{rng.randint(0, 9999):04d}"
        raw_order_id = f"ord-{letter.lower()}{digits}"
        gold_order_id = f"{letter}{digits}"
        item = rng.choice(_ITEMS)
        quantity_token, quantity_value = rng.choice(_QUANTITY_TOKENS)
        abbreviation = rng.choice(abbreviations)
        city = _CITY_MAP[abbreviation]
        priority_word = rng.choice(priority_words)
        priority_code = _PRIORITY_MAP[priority_word]
        template = _TEMPLATES[len(examples) % len(_TEMPLATES)]
        line = template.format(
            order_id=raw_order_id,
            quantity=quantity_token,
            item=item,
            city=abbreviation,
            priority=priority_word,
        )
        if line in seen:
            continue
        seen.add(line)
        fields = {
            "order_id": gold_order_id,
            "item": item,
            "quantity": quantity_value,
            "city": city,
            "priority": priority_code,
        }
        target_json = json.dumps(fields, sort_keys=True)
        examples.append(ExtractionExample(line=line, target_json=target_json))
    return examples


EXTRACTION_EXAMPLES: list[ExtractionExample] = _build_bank()


def build_records(examples: list[ExtractionExample]) -> list[Record]:
    """Turn extraction examples into SFT ``contents`` records."""
    return [
        sft_example(
            user_text=ex.line,
            model_text=ex.target_json,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        for ex in examples
    ]


def split_dataset(
    examples: list[ExtractionExample],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[list[ExtractionExample], list[ExtractionExample], list[ExtractionExample]]:
    """Deterministically shuffle and split into (train, val, test)."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - deterministic split, not cryptographic
    total = len(shuffled)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_extraction_dataset(out_dir: str | Path) -> dict[str, str]:
    """Write train/val/test JSONL under ``out_dir``; return a name→path mapping."""
    out_dir = Path(out_dir)
    train, val, test = split_dataset(EXTRACTION_EXAMPLES)
    paths: dict[str, str] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(build_records(split), path)
        paths[name] = str(path)
    return paths
