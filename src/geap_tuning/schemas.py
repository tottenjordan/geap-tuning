"""Builders for the ``contents``-format tuning JSONL used across GEAP services.

Every GEAP tuning dataset is JSONL where each line is a ``GenerateContent``
example. Supervised fine-tuning uses ``contents`` (a ``user`` turn followed by
a ``model`` turn) plus an optional ``systemInstruction``. Preference tuning and
RLFT layer extra fields (``completions``/``references``) on the same base shape,
so the part/turn helpers here are deliberately service-agnostic and will be
reused when those services are added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

type Part = dict[str, Any]
type Turn = dict[str, Any]
type Record = dict[str, Any]


def text_part(text: str) -> Part:
    """Return a text ``part``."""
    return {"text": text}


def file_part(mime_type: str, file_uri: str) -> Part:
    """Return a ``fileData`` part referencing a GCS object (for multimodal tuning)."""
    return {"fileData": {"mimeType": mime_type, "fileUri": file_uri}}


def user_turn(*parts: Part) -> Turn:
    """Return a ``user`` turn wrapping one or more parts."""
    return {"role": "user", "parts": list(parts)}


def model_turn(*parts: Part) -> Turn:
    """Return a ``model`` turn wrapping one or more parts."""
    return {"role": "model", "parts": list(parts)}


def sft_example(
    *,
    user_text: str,
    model_text: str,
    system_instruction: str | None = None,
) -> Record:
    """Build a single-turn supervised fine-tuning record.

    ``systemInstruction`` is omitted entirely when not provided (its presence is
    optional and its ``role`` is ignored by the tuning service).
    """
    record: Record = {
        "contents": [
            user_turn(text_part(user_text)),
            model_turn(text_part(model_text)),
        ],
    }
    if system_instruction is not None:
        record["systemInstruction"] = {"parts": [text_part(system_instruction)]}
    return record


def write_jsonl(records: list[Record], path: str | Path) -> int:
    """Write ``records`` as one JSON object per line; return the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
