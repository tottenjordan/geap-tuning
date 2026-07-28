"""Call a tuned model endpoint.

After a tuning job succeeds you invoke the tuned model by its endpoint resource
name, exactly like the base model. For thinking-capable models (Gemini 2.5+),
thinking is disabled by default here: supervised fine-tuning trains the model to
emit the ground-truth answer directly, so a thinking trace adds cost and latency
without benefit.
"""

from __future__ import annotations

from typing import Any


def generate(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    endpoint: str,
    contents: Any,  # noqa: ANN401 - str | Part | list[Part], per the SDK
    *,
    thinking_budget: int = 0,
    **config: Any,  # noqa: ANN401 - forwarded to GenerateContentConfig
) -> str:
    """Generate against ``endpoint`` and return the stripped response text."""
    response = client.models.generate_content(
        model=endpoint,
        contents=contents,
        config={"thinking_config": {"thinking_budget": thinking_budget}, **config},
    )
    return response.text.strip()
