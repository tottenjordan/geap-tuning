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

`make lint`, `make test` and `make build` all pass on CPython 3.12.3 (305 tests,
98% coverage; sdist + wheel build with the `uv_build` backend). Re-verified
2026-09-18 after upgrading every dependency to its latest compatible version —
notably `google-cloud-aiplatform` 1.x → 2.x.

CI runs the same three commands as one job (`.github/workflows/ci.yml`). The
repo is **public**, so Actions minutes are free and unlimited — the job is kept
minimal for signal, not for billing.

## Test-suite performance (measured 2026-09-18)

**Almost none of the runtime is the tests.** Of a ~7.5s run before optimization:
~5.9s was module imports at collection, ~1.1s coverage, and **~0.5s actual test
execution**. The slowest single test is 0.04s, so optimizing test bodies is
pointless. Two things follow:

- **`google.cloud.aiplatform` is lazy-imported** in `experiments.py` behind
  `_import_aiplatform()`. It costs ~4.3s to import (roughly doubled in the 1.x →
  2.x bump) and `doe.py` imports `experiments`, so every test paid it. Deferring
  it took `make test` from 7.5s to ~4.4s. `google.genai` (~1.6s) is deliberately
  left eager: it spans six modules and its types appear throughout the launcher
  signatures, which is code that doubles as teaching material.
- **Do not add `pytest-xdist`.** Measured: serial 6.37s, `-n 2` 10.34s, `-n 4`
  8.48s, `-n 8` 9.10s. With ~0.5s of parallelizable work and a multi-second
  import cost every worker re-pays, parallelism is slower at every worker count.

Also ruled out as bottlenecks: no database, no Docker, no network, no file
logging. Tests write ~450 tmp files per run (mostly `sft_vision/test_data.py`
synthetic images) at ~0.1s total; local disk measures 300 MB/s, so a hosted
runner would be slower, not faster, thanks to cold dependency downloads.
