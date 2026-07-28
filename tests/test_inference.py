"""Tests for the tuned-endpoint inference wrapper."""

from unittest.mock import MagicMock

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
