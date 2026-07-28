# geap-tuning

Working, runnable examples of **Gemini Enterprise Agent Platform (GEAP)** model tuning services — supervised fine-tuning, preference tuning, checkpoints, and continuous tuning — using the Google Gen AI SDK and the Vertex/Agent Platform Python SDK.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
cp .env.example .env   # then fill in your GCP project, region, and bucket
make dev               # uv sync --all-groups
```

## Common commands

| Task | Command |
|------|---------|
| Install everything | `make dev` |
| Lint + format check + types | `make lint` |
| Auto-format | `make format` |
| Run tests | `make test` |
| Single test | `uv run pytest tests/test_smoke.py::test_main_runs` |
| Run the SFT example | `uv run python examples/run_sft.py` (requires live GCP + incurs tuning cost) |

## Conventions

Read [CODE_STANDARDS.md](CODE_STANDARDS.md) before writing code or changing the environment. Session notes live in [`docs/notes/`](docs/notes/README.md); guidance for AI agents is in [CLAUDE.md](CLAUDE.md).
