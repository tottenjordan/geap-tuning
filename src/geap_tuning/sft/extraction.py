"""Structured JSON-extraction dataset for the generative SFT example.

A programmatic, correct-by-construction dataset: each example is a messy,
natural-language order line paired with the strict JSON object it should extract
to. Generation is deterministic (seeded) and needs no network or extra
dependency. The records reuse the shared ``contents``-format builders in
:mod:`geap_tuning.schemas`.

Unlike the classification datasets in this package, this is a *generative*
task: the target is a JSON string with an exact schema, so an untuned base
model has real headroom (it tends to add prose, code fences, or emit
``quantity`` as a string ``"3"`` instead of the integer ``3``).
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

_CITIES = (
    "Seattle",
    "Austin",
    "Denver",
    "Chicago",
    "Boston",
    "Portland",
    "Miami",
    "Atlanta",
    "Phoenix",
    "Nashville",
)

_PRIORITIES = ("low", "normal", "high", "urgent")

# Messy, natural-language templates round-robined over the value pools. Each names
# every field in a different order / phrasing so the model must actually parse.
_TEMPLATES = (
    "hey can u rush order {order_id} — {quantity}x {item} to {city}, mark it {priority} thanks!",
    "order {order_id}: please send {quantity} {item} over to {city} ({priority} priority)",
    "need {quantity}x {item} shipped to {city} asap, ref {order_id}, priority={priority}",
    "{priority} request!! {order_id} wants {quantity} {item} delivered in {city} pls",
)


class ExtractionExample(NamedTuple):
    """A messy input line and the strict JSON object it extracts to."""

    line: str
    target_json: str


def _build_bank(*, seed: int = 20240517, n: int = 200) -> list[ExtractionExample]:
    """Deterministically generate ``n`` unique extraction examples."""
    rng = random.Random(seed)  # noqa: S311 - deterministic data gen, not cryptographic
    examples: list[ExtractionExample] = []
    seen: set[str] = set()
    # Generate generously, then dedup by line and keep the first ``n`` unique.
    while len(examples) < n:
        letter = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        order_id = f"{letter}{rng.randint(0, 9999):04d}"
        item = rng.choice(_ITEMS)
        quantity = rng.randint(1, 12)
        city = rng.choice(_CITIES)
        priority = rng.choice(_PRIORITIES)
        template = _TEMPLATES[len(examples) % len(_TEMPLATES)]
        line = template.format(
            order_id=order_id,
            quantity=quantity,
            item=item,
            city=city,
            priority=priority,
        )
        if line in seen:
            continue
        seen.add(line)
        fields = {
            "order_id": order_id,
            "item": item,
            "quantity": quantity,
            "city": city,
            "priority": priority,
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
