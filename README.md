<div align="center">

<!-- Banner: drop your image at docs/imgs/banner.png, then uncomment the line below.
<a href="docs/imgs/banner.png"><img src="docs/imgs/banner.png" alt="GEAP Tuning — working examples of Gemini Enterprise Agent Platform model tuning"></a>
-->

<h1>🎛️ GEAP Tuning 🔧</h1>

<p>
<a href="https://www.python.org/"><img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white"></a>
<a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white"></a>
<a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/lint%20%26%20format-ruff-D7FF64?logo=ruff&logoColor=black"></a>
<a href="https://github.com/astral-sh/ty"><img alt="ty" src="https://img.shields.io/badge/types-ty-261230"></a>
<a href="https://googleapis.github.io/python-genai/"><img alt="Google Gen AI SDK" src="https://img.shields.io/badge/Google%20Gen%20AI%20SDK-4285F4?logo=google&logoColor=white"></a>
<a href="https://cloud.google.com/vertex-ai"><img alt="Vertex AI" src="https://img.shields.io/badge/Vertex%20AI-4285F4?logo=googlecloud&logoColor=white"></a>
<a href="https://deepmind.google/technologies/gemini/"><img alt="Gemini 2.5 Flash" src="https://img.shields.io/badge/Gemini%202.5%20Flash-8E75B2?logo=googlegemini&logoColor=white"></a>
<a href="https://docs.pytest.org/"><img alt="pytest" src="https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white"></a>
</p>

<p><em>Working, runnable examples of <b>Gemini Enterprise Agent Platform (GEAP)</b> model tuning services —<br><b>supervised fine-tuning (SFT)</b>, <b>preference tuning (DPO)</b>, and <b>reinforcement learning fine-tuning (RLFT)</b> —<br>built on the Google Gen AI SDK against the Vertex/Agent Platform backend.</em></p>

</div>

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
| Provision GCP resources | `./scripts/bootstrap_gcp.sh` (enables APIs + creates the region-matched bucket; idempotent, needs `gcloud auth login` first) |
| Run the SFT example | `uv run python examples/run_sft.py` (requires live GCP + incurs tuning cost) |

## Conventions

Read [CODE_STANDARDS.md](CODE_STANDARDS.md) before writing code or changing the environment. Session notes live in [`docs/notes/`](docs/notes/README.md); guidance for AI agents is in [CLAUDE.md](CLAUDE.md).
