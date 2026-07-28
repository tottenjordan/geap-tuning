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
| Run the checkpointing demo | `uv run python examples/run_checkpoints.py` (requires live GCP + incurs tuning cost) |
| Run the continuous-tuning demo | `uv run python examples/run_continuous_tuning.py` (requires live GCP + incurs tuning cost) |

## Checkpointing & continuous tuning

Two cross-cutting sub-features layer on top of all three services (not separate
services). They're demonstrated by dedicated demos — `examples/run_checkpoints.py`
+ [`notebooks/04_checkpoints.ipynb`](notebooks/04_checkpoints.ipynb) and
`examples/run_continuous_tuning.py` +
[`notebooks/05_continuous_tuning.ipynb`](notebooks/05_continuous_tuning.ipynb) —
built on shared helpers in `geap_tuning.jobs` plus three keyword-only launcher
params (`export_last_checkpoint_only`, `evaluation_config`,
`pre_tuned_model_checkpoint_id`).

- **Checkpointing** — tune with `export_last_checkpoint_only=False` (the default)
  to keep one checkpoint per epoch, list them, run per-checkpoint inference, and
  reassign the model's default checkpoint.
- **Continuous tuning** — pass a prior tuned model's resource name as
  `base_model` to continue-tune from it; the demo chains **SFT → RLFT** on the
  same math domain and reports the accuracy lift.

See [`docs/notes/checkpoints-and-continuous-tuning.md`](docs/notes/checkpoints-and-continuous-tuning.md)
for the verified SDK surface and gotchas (Gen AI SDK only; auto-eval
`us-central1` only; base-model 2025-07-11 cutoff; tuning stays regional).

## Evaluation

GEAP exposes a managed **Gen AI Evaluation service**, and this repo pairs it with
its own **offline** scoring. They answer different questions — use both:

| | GEAP Evaluation service (managed) | Offline `evaluate.py` (this repo) |
|---|---|---|
| Runs where | Inside the tuning job, on GEAP | Locally, against the tuned endpoint |
| When | After each checkpoint, automatically | Whenever you call it, post-tuning |
| Metrics | LLM-as-judge / computation metrics from the eval catalog | Task-specific: SFT accuracy, DPO win-rate, RLFT reward accuracy |
| Output | JSONL/summary written to Cloud Storage | Python dict returned in-process |
| Availability | Preview, **`us-central1` only** | Anywhere |

### Using the managed Evaluation service

The service is reached **through the tuning call** — there is no standalone
`client.evals` surface in the Gen AI SDK (2.14.0). You attach an
`EvaluationConfig` to the job; GEAP then evaluates each exported checkpoint and
writes the results to GCS. Build the config with
[`geap_tuning.autoeval.build_evaluation_config`](src/geap_tuning/autoeval.py) and
pass it to any launcher's `evaluation_config` parameter:

```python
from geap_tuning.autoeval import build_evaluation_config
from geap_tuning.config import genai_client, load_config
from geap_tuning.sft.tune import launch_sft_job

cfg = load_config()  # location must be us-central1 for auto-eval
client = genai_client(cfg)

eval_config = build_evaluation_config(cfg.bucket, prefix="sft_eval")

job = launch_sft_job(
    client,
    train_uri=train_uri,
    val_uri=val_uri,
    display_name="geap-sft-support-intent",
    evaluation_config=eval_config,     # <- runs after each checkpoint
    export_last_checkpoint_only=False, # keep checkpoints so each gets evaluated
)
```

Under the hood `build_evaluation_config` assembles the SDK types:

```python
types.EvaluationConfig(
    metrics=[types.Metric(name="FLUENCY", prompt_template="Evaluate the fluency of this response: {prediction}")],
    output_config=types.OutputConfig(
        gcs_destination=types.GcsDestination(output_uri_prefix="gs://<bucket>/sft_eval"),
    ),
)
```

- **Metrics** — each `types.Metric` is either an LLM-as-judge metric (supply a
  `prompt_template`, optional `judge_model_system_instruction`) or a computation
  metric (`custom_function`). Pass your own list to `build_evaluation_config(...,
  metrics=[...])`; the default is a single pointwise fluency metric as a starting
  point. **Verify metric names/templates against the live eval catalog before a
  real run** — the catalog evolves, and the SDK lowercases `Metric.name`.
- **Autorater** — `EvaluationConfig.autorater_config` (`types.AutoraterConfig`)
  tunes the judge: `autorater_model`, `sampling_count`, `flip_enabled` (mitigates
  position bias), `generation_config`.
- **Cadence** — the launcher evaluates each checkpoint; `CreateTuningJobConfig`
  also exposes `evaluate_interval` for step-based cadence if you thread it through.
- **Results** — land under `output_uri_prefix` in Cloud Storage; view them there
  or in **Agent Platform Studio → Tune and Distill → _your tuned model_**.

> Auto-eval is Preview and available in **`us-central1` only**, so keep
> `GCP_REGION`/`GOOGLE_CLOUD_LOCATION` on `us-central1` when using it.

### Offline evaluation (complementary)

Each service ships a small, unit-tested offline scorer that runs against the
tuned endpoint and returns a metrics dict — used by the `examples/run_*.py`
drivers and notebooks:

- SFT — [`sft/evaluate.py`](src/geap_tuning/sft/evaluate.py) `run_eval` →
  `accuracy`, `macro_f1`, per-label `report`.
- DPO — [`preference/evaluate.py`](src/geap_tuning/preference/evaluate.py)
  `score_winrate` → `win_rate` + `wins`/`losses`/`ties`/`n`.
- RLFT — [`rlft/evaluate.py`](src/geap_tuning/rlft/evaluate.py) `run_rlft_eval`
  → `accuracy` + `correct`/`n`, reusing the training reward.

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
    job = launch_rlft_job(
        client,
        train_uri=train_uri,
        val_uri=val_uri,
        display_name="geap-rlft-math-v1",
        base_model="gemini-3.5-flash",
    )
    job = wait_for_tuning_job(client, job.name)
    metrics = run_rlft_eval(
        test_records, generate_fn=lambda u: generate(client, tuned_endpoint(job), u)
    )
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
