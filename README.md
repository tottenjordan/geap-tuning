<div align="center">

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
| Run the DPO example | `uv run python examples/run_preference.py` (requires live GCP + incurs tuning cost) |
| Run the RLFT example | `uv run python examples/run_rlft.py` (requires live GCP + incurs tuning cost) |

## Experiment tracking

GEAP tracks tuning experiments at two levels — one you get for free, one you opt into.

### 1. Built-in tuning metrics (automatic, no code)

Every Gemini tuning job launched here — SFT, DPO, and RLFT — automatically emits
training/validation metrics that stream to the Cloud console in real time. View them
under **Agent Platform Studio → Tune and Distill → _your tuned model_ → Monitor tab**.
The metric set depends on the method:

| Method | Training metrics | Validation metrics (only if a validation set is provided) |
|--------|------------------|-----------------------------------------------------------|
| SFT | `/train_total_loss`, `/train_fraction_of_correct_next_step_preds`, `/train_num_predictions` | `/eval_total_loss`, `/eval_fraction_of_correct_next_step_preds`, `/eval_num_predictions` |
| DPO (preference) | `/preference_optimization_train_loss` | `/eval_total_loss` |
| RLFT (reinforcement) | reward/loss curves over training steps | reward curves over the validation set |

This is why each launcher (`launch_sft_job`, `launch_preference_job`, `launch_rlft_job`)
accepts an optional `val_uri`: **passing a validation split is what unlocks the `/eval_*`
curves** — without it you only get the training curves. No SDK calls beyond launching the
job are needed; the metrics appear once the job starts running.

### 2. Vertex AI Experiments (opt-in, for comparing runs)

To compare *multiple* tuning runs — a hyperparameter sweep over `adapter_size`, `epochs`,
`learning_rate_multiplier`, or (RLFT) `samples_per_prompt` — and to record your own offline
eval numbers (e.g. the held-out accuracy from `run_rlft_eval`) alongside each run, wrap the
launch in a **Vertex AI Experiment**. This lives on the `google-cloud-aiplatform` SDK (already
a dependency), *complementing* the `google.genai` tuning call — Experiments is orchestration
around the job, so mixing the two SDKs here is intentional, not a violation of the
"one SDK path per example" rule (see [CLAUDE.md](CLAUDE.md)).

```python
from google.cloud import aiplatform

from geap_tuning.config import load_config, genai_client
from geap_tuning.rlft.tune import launch_rlft_job
from geap_tuning.jobs import wait_for_tuning_job, tuned_endpoint
from geap_tuning.rlft.evaluate import run_rlft_eval

cfg = load_config()
aiplatform.init(project=cfg.project, location=cfg.location, experiment="geap-rlft-math")
client = genai_client(cfg)  # tuning stays regional

with aiplatform.start_run("v1-adapter16"):
    aiplatform.log_params({"base_model": "gemini-3.5-flash", "adapter_size": 16, "epochs": 5})
    job = launch_rlft_job(client, train_uri=train_uri, val_uri=val_uri,
                          display_name="geap-rlft-math-v1", base_model="gemini-3.5-flash")
    job = wait_for_tuning_job(client, job.name)
    metrics = run_rlft_eval(test_records,
                            generate_fn=lambda u: generate(client, tuned_endpoint(job), u))
    aiplatform.log_metrics({"held_out_accuracy": metrics["accuracy"], "n": metrics["n"]})
```

Each `start_run(...)` is one trackable run; `log_params`/`log_metrics` attach a
key→value snapshot (summary metrics). Compare runs side by side in
**console → Experiments**, or pull them programmatically with
`aiplatform.Experiment("geap-rlft-math").get_data_frame()` (and per run,
`run.get_params()` / `run.get_metrics()`). Longitudinal *time-series* metrics additionally
require a Vertex AI TensorBoard instance. Experiment runs themselves incur no extra
charge — you pay only for the tuning/eval resources they wrap.

See [`docs/notes/experiment-tracking.md`](docs/notes/experiment-tracking.md) for the full
API surface, the automatic-vs-opt-in split, and gotchas.

## Conventions

Read [CODE_STANDARDS.md](CODE_STANDARDS.md) before writing code or changing the environment. Session notes live in [`docs/notes/`](docs/notes/README.md); guidance for AI agents is in [CLAUDE.md](CLAUDE.md).
