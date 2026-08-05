"""Tests for the concise-email preference (DPO) dataset module."""

import re
from pathlib import Path

from geap_tuning.preference.email import (
    EMAIL_DRAFTS,
    SYSTEM_INSTRUCTION,
    build_preference_dataset,
    build_preference_records,
    split_dataset,
)


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def test_bank_is_nonempty_triples() -> None:
    assert len(EMAIL_DRAFTS) >= 55  # sized so a 0.25 test split gives a usable n
    for triple in EMAIL_DRAFTS:
        assert len(triple) == 3
        assert all(isinstance(part, str) and part.strip() for part in triple)


def test_drafts_unique() -> None:
    drafts = [draft for draft, _, _ in EMAIL_DRAFTS]
    assert len(drafts) == len(set(drafts))


def test_preferred_shorter_than_dispreferred() -> None:
    # The concision signal: the preferred rewrite is materially shorter every time.
    for draft, preferred, dispreferred in EMAIL_DRAFTS:
        assert _words(preferred) < _words(dispreferred), draft


def test_build_records_shape() -> None:
    records = build_preference_records(EMAIL_DRAFTS[:3])
    for record, (draft, preferred, dispreferred) in zip(records, EMAIL_DRAFTS[:3], strict=True):
        assert record["contents"][0]["parts"][0]["text"] == draft
        completions = record["completions"]
        assert completions[0]["score"] == 1
        assert completions[1]["score"] == 0
        assert completions[0]["completion"]["parts"][0]["text"] == preferred
        assert completions[1]["completion"]["parts"][0]["text"] == dispreferred
        assert record["systemInstruction"]["parts"][0]["text"] == SYSTEM_INSTRUCTION


def test_split_dataset_is_deterministic_partition() -> None:
    train, val, test = split_dataset(EMAIL_DRAFTS)
    again = split_dataset(EMAIL_DRAFTS)
    assert (train, val, test) == again
    all_drafts = [t[0] for t in train + val + test]
    assert set(all_drafts) == {t[0] for t in EMAIL_DRAFTS}
    assert train
    assert val
    # The test split must be large enough for a meaningful head-to-head win-rate CI.
    assert len(test) >= 12


def test_build_preference_dataset_writes_splits(tmp_path: Path) -> None:
    paths = build_preference_dataset(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    for path in paths.values():
        assert Path(path).exists()
