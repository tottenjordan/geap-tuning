# Toolchain & standards

How this project's Python toolchain is wired, and the non-obvious decisions behind it. Rules to follow are in [CODE_STANDARDS.md](../../CODE_STANDARDS.md); this note explains *why*, so future sessions don't re-litigate. Set up 2026-07-28.

## Stack

- **uv** (0.11.x) manages deps + venv. `src/` layout, build backend `uv_build`. Runtime deps: `google-genai`, `google-cloud-aiplatform`, `python-dotenv`.
- **ruff** lint+format, **ty** type check, **pytest**+**pytest-cov** tests. All in `[dependency-groups]` (`lint`, `test`, `dev` includes both) — not `[project.optional-dependencies]`.
- `Makefile` wraps flows: `make dev|lint|format|test`. `make lint` = `ruff format --check` + `ruff check` + `ty check src/`.

## Non-obvious decisions (the reason this note exists)

- **Python pinned to 3.12**, not uv's default. `uv init` wrote `.python-version` = `3.14` and `requires-python = ">=3.14"`, but the box only has CPython 3.12.3. Pinned `.python-version` → `3.12`, `requires-python` → `>=3.12`, `target-version`/ty → `py312`. If you bump Python, change all four.
- **ruff `select = ["ALL"]`** with deliberate ignores in `pyproject.toml`: `D` (docstrings), `COM812` (formatter owns commas), `CPY001` (no copyright headers), `T201` (`print()` is expected in runnable examples). Tests ignore `S101` (asserts) via `per-file-ignores`.
- **`docs/` is excluded from ruff** (`extend-exclude = ["docs"]`). Without this, `ruff format` rewrites the Python snippets inside note Markdown fenced blocks (ruff 0.16 formats embedded code) and `make lint` fails. Keep doc snippets illustrative, not lint-clean.
- `[tool.ty.terminal] error-on-warning = true` — ty warnings fail the build; treat them as errors.

## Verified green

`make lint` and `make test` both pass on 3.12.3 as of the setup date (1 smoke test, 100% coverage on the placeholder entry point).
