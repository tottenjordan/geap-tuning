"""Tests for the tuned-endpoint inference wrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from geap_tuning.inference import generate


def test_generate_passes_endpoint_and_disables_thinking() -> None:
    client = MagicMock()
    client.models.generate_content.return_value.text = "  hello  "

    out = generate(client, "projects/../endpoints/1", "Q")

    assert out == "hello"
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "projects/../endpoints/1"
    assert kwargs["contents"] == "Q"
    assert kwargs["config"]["thinking_config"]["thinking_budget"] == 0


def test_generate_forwards_extra_config() -> None:
    client = MagicMock()
    client.models.generate_content.return_value.text = "x"

    generate(client, "ep", "Q", temperature=0.2, max_output_tokens=64)

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config["temperature"] == 0.2
    assert config["max_output_tokens"] == 64


def test_generate_raises_a_clear_error_when_blocked() -> None:
    # response.text is None on a safety block / MAX_TOKENS; previously this was a
    # bare AttributeError partway through an eval loop.
    client = MagicMock()
    response = client.models.generate_content.return_value
    response.text = None
    response.candidates = [SimpleNamespace(finish_reason="SAFETY")]

    with pytest.raises(RuntimeError, match="no text") as excinfo:
        generate(client, "projects/p/locations/us/endpoints/1", "Q")

    message = str(excinfo.value)
    assert "SAFETY" in message  # the reason travels with the error
    assert "endpoints/1" in message  # and so does the endpoint


def test_generate_reports_missing_text_without_candidates() -> None:
    client = MagicMock()
    response = client.models.generate_content.return_value
    response.text = None
    response.candidates = []

    with pytest.raises(RuntimeError, match="no text"):
        generate(client, "ep", "Q")


def test_generate_allows_empty_string_response() -> None:
    # "" is a real (if unhelpful) answer, not a blocked response: it must not raise.
    client = MagicMock()
    client.models.generate_content.return_value.text = "   "
    assert generate(client, "ep", "Q") == ""
