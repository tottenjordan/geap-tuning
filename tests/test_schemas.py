"""Tests for JSONL record builders."""

import json
from pathlib import Path

from geap_tuning.schemas import (
    file_part,
    model_turn,
    sft_example,
    text_part,
    user_turn,
    write_jsonl,
)


def test_text_and_file_parts() -> None:
    assert text_part("hi") == {"text": "hi"}
    assert file_part("image/jpeg", "gs://b/x.jpg") == {
        "fileData": {"mimeType": "image/jpeg", "fileUri": "gs://b/x.jpg"}
    }


def test_turn_helpers_wrap_parts() -> None:
    assert user_turn(text_part("q")) == {"role": "user", "parts": [{"text": "q"}]}
    assert model_turn(text_part("a")) == {"role": "model", "parts": [{"text": "a"}]}


def test_sft_example_shape() -> None:
    ex = sft_example(user_text="Hi", model_text="Argh!", system_instruction="Be a pirate.")
    assert ex["systemInstruction"]["parts"] == [{"text": "Be a pirate."}]
    assert ex["contents"][0] == {"role": "user", "parts": [{"text": "Hi"}]}
    assert ex["contents"][1] == {"role": "model", "parts": [{"text": "Argh!"}]}


def test_sft_example_omits_system_when_absent() -> None:
    ex = sft_example(user_text="Hi", model_text="Yo")
    assert "systemInstruction" not in ex


def test_write_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    count = write_jsonl([{"a": 1}, {"b": 2}], path)
    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
