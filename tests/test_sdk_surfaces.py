"""Guards on which SDK surfaces this package is allowed to touch.

``vertexai.Client`` is deprecated in google-cloud-aiplatform 2.1.0 in favour of
``agentplatform.Client`` (which ships vendored inside google-cloud-aiplatform,
not as a separate PyPI package). This repo has never used it, and these tests
keep it that way — a contributor copying from Google's tuning docs, which
``docs/notes/geap-tuning-overview.md`` points at as "SDK path 2", could
otherwise reintroduce it silently.

Only ``vertexai.Client`` is banned. ``vertexai.tuning.sft`` and ``vertexai.init``
emit no deprecation warning as of 2.1.3 and remain a legitimate documented
alternative to the ``google.genai`` path this repo actually uses. Note also that
``genai.Client(vertexai=True, ...)`` in ``config.py`` is a routing *keyword* on a
different SDK, not this class — the AST checks below cannot confuse the two.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEARCH_ROOTS = (_REPO_ROOT / "src" / "geap_tuning", _REPO_ROOT / "examples")


def _vertexai_client_hits(tree: ast.AST) -> list[int]:
    """Return the line numbers at which ``tree`` reaches for ``vertexai.Client``.

    Catches **both** spellings on purpose. The equivalent guard in the reference
    migration (tottenjordan/gepa-prompt-wrangler#47) matched only the attribute
    form and missed six of seven real sites written as ``from vertexai import
    Client``.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        # form 1: `from vertexai import Client`
        if isinstance(node, ast.ImportFrom) and node.module == "vertexai":
            lines.extend(node.lineno for alias in node.names if alias.name == "Client")
        # form 2: `vertexai.Client(...)`
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "Client"
            and isinstance(node.value, ast.Name)
            and node.value.id == "vertexai"
        ):
            lines.append(node.lineno)
    return sorted(lines)


def test_no_module_uses_deprecated_vertexai_client() -> None:
    """No shipped module may construct the deprecated client."""
    offenders: list[str] = []
    for root in _SEARCH_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            offenders.extend(
                f"{path.relative_to(_REPO_ROOT)}:{line}" for line in _vertexai_client_hits(tree)
            )
    assert not offenders, (
        "vertexai.Client is deprecated; use agentplatform.Client instead. Found at: "
        + ", ".join(offenders)
    )


def test_guard_detects_both_spellings() -> None:
    """The detector itself must catch the import form, not just the call form."""
    call_form = ast.parse("import vertexai\nx = vertexai.Client(project='p')\n")
    import_form = ast.parse("from vertexai import Client\nx = Client(project='p')\n")
    assert _vertexai_client_hits(call_form) == [2]
    assert _vertexai_client_hits(import_form) == [1]


def test_guard_ignores_the_genai_routing_keyword() -> None:
    """``genai.Client(vertexai=True)`` is a different SDK and must not trip the guard."""
    benign = ast.parse(
        "from google import genai\nc = genai.Client(vertexai=True, project='p')\n",
    )
    assert _vertexai_client_hits(benign) == []
