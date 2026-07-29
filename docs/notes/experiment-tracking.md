# Experiment tracking

How GEAP tracks tuning experiments. Two independent layers — verified 2026-07-29 against
the GEAP docs. See [tuning APIs](tuning-apis.md) for the launch calls these wrap.

**Implemented in this repo:** Layer 2 is wrapped by
[`src/geap_tuning/experiments.py`](../../src/geap_tuning/experiments.py) (`init_experiment`,
`get_or_create_tensorboard`, `track_run`, `log_summary_metrics`, `log_timeseries_metrics`,
`experiment_dataframe`) and demoed by
[`examples/run_experiment_tracking.py`](../../examples/run_experiment_tracking.py) /
[`notebooks/08_experiment_tracking.ipynb`](../../notebooks/08_experiment_tracking.ipynb).
Layer 1 needs no code.

## Layer 1 — built-in tuning metrics (automatic)

Every Gemini tuning job (SFT/DPO/RLFT via `client.tunings.tune(...)`) auto-collects
train/validation metrics with **no extra code**. They stream in real time to the Cloud
console: **Agent Platform Studio → Tune and Distill → _tuned model_ → Monitor tab**.

- **SFT** (Gemini 2.x/translation): `/train_total_loss`,
  `/train_fraction_of_correct_next_step_preds`, `/train_num_predictions`; validation
  `/eval_total_loss`, `/eval_fraction_of_correct_next_step_preds`, `/eval_num_predictions`.
- **DPO**: `/preference_optimization_train_loss`; validation `/eval_total_loss`.
- **RLFT**: reward/loss curves over training steps (+ validation curves). Exact metric-key
  names not confirmed in docs — re-verify before quoting them.

**Key dependency:** the `/eval_*` curves only appear if a **validation dataset** was passed
at launch. That is the reason the launchers (`launch_sft_job`, `launch_preference_job`,
`launch_rlft_job`) all take an optional `val_uri` — omit it and you get training curves only.

This layer is console-first; the metrics are backed by an internal TensorBoard the job
manages. No `aiplatform` calls required.

## Layer 2 — Vertex AI Experiments (opt-in, cross-run comparison)

For comparing many runs (hyperparameter sweeps) and logging your **own** offline metrics
(e.g. `run_eval`/`run_rlft_eval` accuracy), use Vertex AI Experiments. This is on the
**`google-cloud-aiplatform`** SDK, NOT `google.genai`. The `experiments.py` helper wraps the
raw calls below one-to-one (`init_experiment` → `aiplatform.init`, `track_run` →
`start_run` + `log_params`, `log_summary_metrics` → `log_metrics`, etc.).

**Cost pattern used in the demo:** a multi-job hyperparameter sweep is expensive, so
`run_experiment_tracking.py` instead reuses a **single** SFT-with-checkpoints job and derives
one Experiment run per exported checkpoint (evaluate each, log its accuracy) — cross-run
comparison without paying for multiple tuning jobs.

- SDK-path note: this is the one sanctioned place to mix SDKs. Experiments wraps/orchestrates
  the job; the actual tune call stays on the `genai` path. Not a violation of the repo's
  "one SDK path per example" rule (that rule is about the tuning call itself).

Core API:
- `aiplatform.init(project=..., location=..., experiment="<name>", experiment_tensorboard=<tb>)`
  — create/select the experiment context. Preferred creation path (vs.
  `aiplatform.Experiment.create`, which makes the resource but does not set it globally, so
  `start_run` won't attach to it). Pass `experiment_tensorboard=<resource-name>` to attach a
  Managed TensorBoard (enables time-series); leave it `None` for summary-only. The helper's
  `init_experiment(..., tensorboard=...)` and `get_or_create_tensorboard(...)` cover this.
- `with aiplatform.start_run("<run-name>") as run:` — one trackable run.
- `aiplatform.log_params({...})` — input snapshot (base_model, adapter_size, epochs,
  learning_rate_multiplier, samples_per_prompt, beta, …).
- `aiplatform.log_metrics({...})` — **summary metrics**: one value per key per run
  (e.g. held-out accuracy). Use for our offline eval outputs.
- `aiplatform.log_time_series_metrics({...})` — **time-series metrics** (per-step); these
  are stored in **Vertex AI TensorBoard** and require a TensorBoard instance attached to the
  experiment. Summary metrics do NOT need TensorBoard.
- Read back: `aiplatform.Experiment("<name>").get_data_frame()` for a cross-run table;
  per run, `run.get_params()` / `run.get_metrics()`; `run.delete()` to remove.
- Resume an existing run: `aiplatform.start_run("<run-name>", resume=True)`.

Cost: experiment runs incur no extra charge — only the wrapped tuning/eval resources.

## Gotchas

- **Two things called "tracking".** Layer 1 (console Monitor tab, automatic, tuning-loss
  curves) is distinct from Layer 2 (Experiments SDK, opt-in, your params/metrics). They are
  complementary, not the same feature.
- **Validation set gates eval curves** (Layer 1) — see above.
- **TensorBoard only for time-series** (Layer 2). Summary metrics work without it.
- **Autologging** (`enable_autolog=True` on `CustomJob.from_local_script`) is a *custom
  training* feature (Fastai/Keras/sklearn/XGBoost/etc.), not applicable to the managed
  Gemini tuning jobs this repo launches — don't reach for it here.
- Region: keep the experiment `location` aligned with the tuning region (`us-central1` here);
  Experiments is a regional resource like the tuning job.
