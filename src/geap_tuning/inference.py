"""Call a tuned model endpoint.

After a tuning job succeeds you invoke the tuned model by its endpoint resource
name, exactly like the base model. For thinking-capable models (Gemini 2.5+),
thinking is disabled by default here: supervised fine-tuning trains the model to
emit the ground-truth answer directly, so a thinking trace adds cost and latency
without benefit.
"""

from __future__ import annotations

from typing import Any


def _no_text_message(response: Any, endpoint: str) -> str:  # noqa: ANN401 - SDK response
    """Explain an empty response, naming the finish reason when the SDK supplies one."""
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    detail = f" (finish_reason={reason})" if reason else ""
    return (
        f"Model returned no text for endpoint {endpoint}{detail}. "
        f"This usually means the response was blocked by a safety filter or hit the "
        f"output token limit."
    )


def generate(
    client: Any,  # noqa: ANN401 - SDK client type is dynamic
    endpoint: str,
    contents: Any,  # noqa: ANN401 - str | Part | list[Part], per the SDK
    *,
    thinking_budget: int = 0,
    **config: Any,  # noqa: ANN401 - forwarded to GenerateContentConfig
) -> str:
    """Generate against ``endpoint`` and return the stripped response text.

    Raises :class:`RuntimeError` when the response carries no text. ``response.text``
    is ``Optional[str]`` — it is ``None`` for a safety block, a ``MAX_TOKENS`` stop,
    or an empty candidate — so returning it unchecked produced a bare
    ``AttributeError: 'NoneType' object has no attribute 'strip'`` partway through an
    eval loop, with nothing to say the response had been blocked.
    """
    response = client.models.generate_content(
        model=endpoint,
        contents=contents,
        config={"thinking_config": {"thinking_budget": thinking_budget}, **config},
    )
    text = response.text
    if text is None:
        raise RuntimeError(_no_text_message(response, endpoint))
    return text.strip()
