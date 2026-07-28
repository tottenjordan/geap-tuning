# Checkpointing & continuous tuning — verified SDK surface

Durable reference for GEAP's two cross-cutting tuning sub-features, as
implemented in this repo. Verified against `google-genai` **2.14.0** on
2026-07-28. Re-verify SDK symbols before acting — this surface moves. See
[[tuning-apis]] for the per-service call shapes and [[geap-tuning-overview]] for
the method matrix.

Both features are **Gen AI SDK only** (not the older Agent Platform / `vertexai`
SDK) and demonstrated by dedicated demos: `examples/run_checkpoints.py` +
`notebooks/04_checkpoints.ipynb` and `examples/run_continuous_tuning.py` +
`notebooks/05_continuous_tuning.ipynb`. The shared helpers live in `jobs.py`; the
launcher params are on all three `launch_*_job` functions.

## Checkpointing

A tuning job can export intermediate checkpoints (one per epoch up to ~10, evenly
spaced beyond that), each independently servable — so you can compare epochs and
roll the default back if the last epoch overfit.

- **Config:** `types.CreateTuningJobConfig.export_last_checkpoint_only: bool`
  (default `False` → intermediate checkpoints **on**). RLFT also has
  `checkpoint_interval: int` (reinforcement-tuning only) for steps-between-saves.
- **Read:** `job.tuned_model.checkpoints` → `list[TunedModelCheckpoint]`, each
  with `.checkpoint_id`, `.epoch`, `.step`, `.endpoint`. Wrapped by
  `jobs.list_checkpoints(job)` (empty list when none exported).
- **Per-checkpoint inference:** `generate_content(model=checkpoint.endpoint, ...)`
  — each checkpoint has its own endpoint. Wrapped by
  `jobs.checkpoint_endpoint(job, checkpoint_id)`.
- **Default checkpoint** (the one the model's base endpoint serves):
  - read: `client.models.get(model=<tuned model name>).default_checkpoint_id`
    → `jobs.get_default_checkpoint_id(client, job)`.
  - change: `client.models.update(model=..., config=types.UpdateModelConfig(
    default_checkpoint_id=...))` → `jobs.set_default_checkpoint(client, job, id)`.
- `jobs.tuned_model_name(job)` returns `job.tuned_model.model`
  (`projects/.../models/id@ver`) — the value both `models.get/update` and
  continuous tuning need.

## Continuous tuning

Continue-tune from an already-tuned model instead of a base model — the
documented best practice is **SFT first, then RLFT/DPO** on top.

- **How:** pass the prior tuned model's resource name
  (`projects/.../models/id@ver`) as `base_model` to `client.tunings.tune(...)`.
  The SDK auto-detects the `projects/` prefix and wraps it as a `PreTunedModel`
  — **no** extra plumbing on the launchers (they already forward `base_model`).
- **Pick a source checkpoint:** `types.CreateTuningJobConfig
  .pre_tuned_model_checkpoint_id` (threaded through every launcher). `None` uses
  the model's default checkpoint.
- **Supported chains:** SFT→SFT, SFT→DPO, SFT→RLFT, RLFT→RLFT. This repo ships
  **SFT→RLFT** as the concrete demo (`run_continuous_tuning.py`); the RLFT stage
  reuses the SFT-seed data via `rlft.data.build_math_sft_dataset` (RLFT records
  carry no gold completion, so the SFT stage needs its own gold `Answer: <n>`
  turn built from the same `MATH_PROBLEMS`).

## Gotchas

- **Gen AI SDK only** — not the Agent Platform / `vertexai` SDK.
- **Auto-eval is `us-central1` only** (Preview) — the `evaluation_config`
  launcher param (built by `autoeval.build_evaluation_config`) runs eval after
  each checkpoint. The SDK **lowercases** `Metric.name`.
- **Base-model cutoff:** the base SFT model for a continuous run must have been
  tuned **on/after 2025-07-11**.
- **Same region:** both stages of a continuous chain run in the same region;
  tuning stays **regional** (never `global` — `global` is inference-only for
  Gemini 3.x).
- **Supported models** (checkpoints + continuous): Gemini 2.5 Flash / Flash-Lite
  / Pro, 3.1 Flash-Lite, 3.5 Flash.
