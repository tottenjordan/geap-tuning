"""Working examples of Gemini Enterprise Agent Platform (GEAP) model tuning services."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    """Return the installed distribution version, or a placeholder when not installed."""
    try:
        return version("geap-tuning")
    except PackageNotFoundError:  # pragma: no cover - only when run from a source tree
        return "unknown"


def main() -> None:
    """Print the installed version and how the resolved config looks.

    This is the ``geap-tuning`` console script. It deliberately does **no** tuning:
    the package is a collection of runnable examples, so the useful thing a bare
    invocation can do is confirm the install and tell you whether ``.env`` resolves
    — the most common first-run failure — then point at the examples.
    """
    print(f"geap-tuning {_version()}")

    # Imported lazily: this pulls in the Gen AI SDK, and `--version`-style use
    # should not pay for that.
    from geap_tuning.config import load_config  # noqa: PLC0415

    try:
        cfg = load_config()
    except ValueError as exc:
        print(f"config: NOT resolved ({exc})")
        print("        copy .env.example to .env and fill it in")
    else:
        print(f"config: project={cfg.project} location={cfg.location} bucket={cfg.bucket}")

    print("Run an example, e.g.: uv run python examples/run_sft.py")
