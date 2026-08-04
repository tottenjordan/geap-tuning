"""Tests for the constrained-generation RLFT dataset module."""

from pathlib import Path

from geap_tuning.rlft.constrained import (
    CONSTRAINT_SPECS,
    NEUTRAL_SYSTEM_INSTRUCTION,
    ConstraintSpec,
    _build_bank,
    build_constrained_dataset,
    build_records,
    split_dataset,
)


def test_build_bank_is_deterministic() -> None:
    assert _build_bank() == _build_bank()
    assert len(_build_bank()) == 150


def test_prompts_unique() -> None:
    prompts = [spec.prompt for spec in CONSTRAINT_SPECS]
    assert len(prompts) == len(set(prompts))


def test_specs_are_constraint_specs() -> None:
    assert CONSTRAINT_SPECS
    assert all(isinstance(spec, ConstraintSpec) for spec in CONSTRAINT_SPECS)


def test_all_reference_values_are_strings() -> None:
    for spec in CONSTRAINT_SPECS:
        for key, value in spec.references.items():
            assert isinstance(value, str), (key, value)


def test_bands_are_satisfiable() -> None:
    for spec in CONSTRAINT_SPECS:
        refs = spec.references
        assert int(refs["min_words"]) <= int(refs["max_words"])
        assert int(refs["min_sentences"]) <= int(refs["max_sentences"])


def test_prompt_names_every_constraint() -> None:
    # A spot check: the prompt text mentions its keywords, forbidden words, and bands.
    spec = CONSTRAINT_SPECS[0]
    for keyword in spec.references["required_keywords"].split(","):
        assert keyword in spec.prompt


def test_build_records_shape() -> None:
    records = build_records(CONSTRAINT_SPECS[:3])
    for record, spec in zip(records, CONSTRAINT_SPECS[:3], strict=True):
        assert list(record["contents"]) == [record["contents"][0]]  # user turn only
        assert record["contents"][0]["role"] == "user"
        assert record["contents"][0]["parts"][0]["text"] == spec.prompt
        assert record["references"] == spec.references
        assert record["systemInstruction"]["parts"][0]["text"] == NEUTRAL_SYSTEM_INSTRUCTION
        assert "completions" not in record


def test_split_dataset_is_deterministic_partition() -> None:
    train, val, test = split_dataset(CONSTRAINT_SPECS)
    again = split_dataset(CONSTRAINT_SPECS)
    assert (train, val, test) == again
    all_prompts = [s.prompt for s in train + val + test]
    assert set(all_prompts) == {s.prompt for s in CONSTRAINT_SPECS}
    # A ~20% test split of 150 gives ~30, enough for a bootstrap CI.
    assert len(test) >= 25


def test_build_constrained_dataset_writes_splits(tmp_path: Path) -> None:
    paths = build_constrained_dataset(tmp_path)
    assert set(paths) == {"train", "val", "test"}
    for path in paths.values():
        assert Path(path).exists()
