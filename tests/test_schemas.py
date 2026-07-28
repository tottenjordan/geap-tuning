"""Tests for JSONL record builders."""

import json
from pathlib import Path

from geap_tuning.schemas import (
    completion_turn,
    file_part,
    model_turn,
    preference_example,
    rlft_example,
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


def test_completion_turn_shape() -> None:
    assert completion_turn(1, "Hi there!") == {
        "score": 1,
        "completion": {"role": "model", "parts": [{"text": "Hi there!"}]},
    }


def test_preference_example_shape() -> None:
    ex = preference_example(
        user_text="My order is late.",
        preferred_text="So sorry! I'll track it now.",
        dispreferred_text="Not my problem.",
    )
    assert ex["contents"] == [{"role": "user", "parts": [{"text": "My order is late."}]}]
    assert [c["score"] for c in ex["completions"]] == [1, 0]
    assert ex["completions"][0]["completion"]["parts"][0]["text"] == "So sorry! I'll track it now."
    assert "systemInstruction" not in ex


def test_preference_example_includes_system_instruction() -> None:
    ex = preference_example(
        user_text="Hi",
        preferred_text="A",
        dispreferred_text="B",
        system_instruction="You are a support agent.",
    )
    assert ex["systemInstruction"]["parts"] == [{"text": "You are a support agent."}]


def test_rlft_example_shape() -> None:
    ex = rlft_example(user_text="What is 2+2?", references={"ground_truth_answer": "4"})
    assert ex["contents"] == [{"role": "user", "parts": [{"text": "What is 2+2?"}]}]
    assert ex["references"] == {"ground_truth_answer": "4"}
    assert "systemInstruction" not in ex
    assert "completions" not in ex  # RLFT has no scored completions and no gold model turn


def test_rlft_example_includes_system_instruction() -> None:
    ex = rlft_example(
        user_text="Q",
        references={"ground_truth_answer": "1"},
        system_instruction="You are a math tutor.",
    )
    assert ex["systemInstruction"]["parts"] == [{"text": "You are a math tutor."}]


def test_write_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    count = write_jsonl([{"a": 1}, {"b": 2}], path)
    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
