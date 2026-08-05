"""Verifiable-math reward for the RLFT demo.

This module is BOTH imported (offline eval + unit tests) AND shipped verbatim to
the GEAP code-execution reward sandbox as ``python_code_snippet``. The sandbox
runs this source, then calls ``evaluate(example, response) -> float`` with
camelCase ProtoJSON dicts (``example`` carries ``references``; ``response`` is a
``Content`` with ``parts``). It must therefore stay self-contained — stdlib only,
no ``geap_tuning`` imports. Rewards are clipped to [-1, 1] by the platform.
"""

from __future__ import annotations

import re
from typing import Any

_CORRECT = 1.0
_WRONG = -1.0
_ANSWER_RE = re.compile(r"answer\s*[:=]\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)


def normalize_number(value: str) -> str:
    """Normalize a numeric string for comparison (drop commas/trailing '.'/'.0')."""
    text = value.strip().replace(",", "").rstrip(".")
    return text.removesuffix(".0")


def extract_answer(text: str) -> str | None:
    """Return the normalized final number after the last 'Answer:' marker, or None."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def _response_text(response: dict[str, Any]) -> str:
    content = response.get("content", response)
    parts = content.get("parts", [])
    return " ".join(part.get("text", "") for part in parts)


def evaluate(example: dict[str, Any], response: dict[str, Any]) -> float:
    """Reward 1.0 when the model's final answer matches the ground truth, else -1.0."""
    truth = example.get("references", {}).get("ground_truth_answer")
    predicted = extract_answer(_response_text(response))
    if truth is None or predicted is None:
        return _WRONG
    return _CORRECT if predicted == normalize_number(truth) else _WRONG
