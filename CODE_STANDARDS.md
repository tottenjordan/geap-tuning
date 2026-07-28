# CODE_STANDARDS.md

Standards that **must** be followed when writing code or making environment changes in this repo. When in doubt, prefer the [`modern-python`](https://github.com/trailofbits/cookiecutter-python) conventions this project is built on.

## Git / commits

- **Never add `Co-Authored-By` trailers** to commits or PRs.

## Python tooling

- **Package management: `uv` for everything.** Never call bare `pip` or `python`.
  - Add deps with `uv add <pkg>` (runtime) or `uv add --group <lint|test> <pkg>` (dev). Never hand-edit `[project.dependencies]` / `[dependency-groups]`.
  - Remove with `uv remove`. Sync with `uv sync --all-groups`. Run anything with `uv run <cmd>` — never activate `.venv` manually.
  - Commit `uv.lock`.
- **Lint + format: `ruff` only.** Never black, flake8, or isort. `uv run ruff check .` and `uv run ruff format .`. Config: `select = ["ALL"]` with explicit `ignore`s in `pyproject.toml`.
- **Type checking: `ty`** (Astral). Never mypy or pyright. `uv run ty check src/`.
- **Standalone scripts:** use PEP 723 inline metadata + `uv run script.py`, not `requirements.txt`.
- Use the `src/` layout; target `requires-python = ">=3.12"`.

## Testing

- **`pytest`** for tests (`uv run pytest`); a single test: `uv run pytest tests/test_x.py::test_name`.
- **`ty`** for type checks — treat type errors as test failures.
- Before considering work done, `make lint && make test` must pass. (`make` targets wrap the `uv run` commands above.)

## Environment / secrets

- All config comes from `.env` (git-ignored). Keep [`.env.example`](.env.example) in sync when adding a new variable — add the key with a safe placeholder, never a real value.
- Never commit `.env` or real credentials. See [environment notes](docs/notes/environment.md) for the variable groups and region/bucket gotchas.
