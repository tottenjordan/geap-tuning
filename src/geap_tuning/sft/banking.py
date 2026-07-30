"""Load the banking77 fine-grained intent-classification benchmark as SFT records.

banking77 (PolyAI/banking77, CC-BY-4.0) is 13,083 customer-service queries over 77
banking intents. Unlike the 5-intent demo dataset in :mod:`geap_tuning.sft.data` it
does NOT saturate, so a hyperparameter sweep on it produces a real, discriminating
signal (see ``docs/notes/banking77-dataset.md``).

Sourced as two plain ``text,category`` CSVs via stdlib (no extra dependency). Every
record shares a ``systemInstruction`` listing the candidate labels, so both the untuned
baseline and the tuned models emit valid labels and the before/after comparison is fair.
"""

from __future__ import annotations

import csv
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from geap_tuning.schemas import Record, sft_example, write_jsonl

_RAW = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
BANKING77_URLS = {"train": f"{_RAW}/train.csv", "test": f"{_RAW}/test.csv"}

type Pair = tuple[str, str]  # (customer message, intent label)


def load_pairs_from_csv(path: str | Path) -> list[Pair]:
    """Read a banking77 ``text,category`` CSV into ``(text, label)`` pairs."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return [(row["text"], row["category"]) for row in csv.DictReader(fh)]


def banking_labels(pairs: list[Pair]) -> tuple[str, ...]:
    """Return the sorted unique intent labels present in ``pairs``."""
    return tuple(sorted({label for _, label in pairs}))


def sample_balanced(pairs: list[Pair], per_class: int, *, seed: int = 42) -> list[Pair]:
    """Deterministically take up to ``per_class`` examples of each label.

    Uses a fixed-seed :class:`random.Random` (never the global RNG) so runs are
    reproducible and the display-name/idempotency story holds. Classes with fewer than
    ``per_class`` examples contribute all they have.
    """
    by_label: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        by_label[pair[1]].append(pair)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    out: list[Pair] = []
    for label in sorted(by_label):
        bucket = by_label[label][:]
        rng.shuffle(bucket)
        out.extend(bucket[:per_class])
    return out


def build_system_instruction(labels: tuple[str, ...]) -> str:
    """Return a constrained-classification instruction listing every allowed label."""
    return (
        "You are a banking customer-service intent classifier. Classify the customer "
        "message into exactly one of the following intents and respond with only that "
        "intent label, nothing else.\nIntents: " + ", ".join(labels)
    )


def build_banking_records(pairs: list[Pair], *, system_instruction: str) -> list[Record]:
    """Build single-turn SFT records (user message -> gold intent label)."""
    return [
        sft_example(user_text=text, model_text=label, system_instruction=system_instruction)
        for text, label in pairs
    ]


def parse_banking_prediction(text: str, labels: tuple[str, ...]) -> str:
    """Canonicalize a raw model reply to one of ``labels`` (else the normalized text).

    Handles trailing punctuation, casing, and space-vs-underscore; falls back to a
    substring match (longest label first) before giving up (which scores as wrong).
    """
    norm = text.strip().splitlines()[0].strip().lower().replace(" ", "_").strip(".!?\"' ")
    if norm in labels:
        return norm
    for label in sorted(labels, key=len, reverse=True):
        if label in norm:
            return label
    return norm


def download_banking77(cache_dir: str | Path) -> dict[str, Path]:  # pragma: no cover
    """Download and cache the train/test CSVs; return ``{'train': path, 'test': path}``."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in BANKING77_URLS.items():
        dest = cache / f"{split}.csv"
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https URL
        paths[split] = dest
    return paths


def build_banking_dataset(
    out_dir: str | Path,
    *,
    csv_dir: str | Path | None = None,
    per_class: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, str]:
    """Sample a balanced subset and write train/val/test JSONL; return their paths.

    ``train`` + ``val`` are carved from banking77's train split (disjoint per label);
    ``test`` is sampled from its held-out test split. ``csv_dir`` uses pre-downloaded
    CSVs (offline); otherwise the CSVs are fetched and cached under ``out_dir/raw``.
    """
    per_class = per_class or {"train": 10, "val": 2, "test": 5}
    out = Path(out_dir)
    if csv_dir is None:
        src = download_banking77(out / "raw")
    else:
        src = {"train": Path(csv_dir) / "train.csv", "test": Path(csv_dir) / "test.csv"}
    train_pairs = load_pairs_from_csv(src["train"])
    test_pairs = load_pairs_from_csv(src["test"])
    system = build_system_instruction(banking_labels(train_pairs))

    # train + val disjoint per label: take (train+val) balanced, then split each class.
    pool = sample_balanced(train_pairs, per_class["train"] + per_class["val"], seed=seed)
    by_label: dict[str, list[Pair]] = defaultdict(list)
    for pair in pool:
        by_label[pair[1]].append(pair)
    train_sel: list[Pair] = []
    val_sel: list[Pair] = []
    for label in sorted(by_label):
        bucket = by_label[label]
        val_sel.extend(bucket[: per_class["val"]])
        train_sel.extend(bucket[per_class["val"] :])
    test_sel = sample_balanced(test_pairs, per_class["test"], seed=seed)

    paths: dict[str, str] = {}
    for name, sel in (("train", train_sel), ("val", val_sel), ("test", test_sel)):
        dest = out / f"{name}.jsonl"
        write_jsonl(build_banking_records(sel, system_instruction=system), dest)
        paths[name] = str(dest)
    return paths
