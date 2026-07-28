# GEAP Tuning APIs — call shapes, dataset schemas, hyperparameters

Durable reference for the three GEAP tuning services. Verified 2026-07-28.
Re-verify SDK symbols and base-model strings before acting — this surface moves.
See [[environment]] for config/auth and [[toolchain]] for tooling.

## Service matrix

| Service | Launch call | Dataset extra fields | Key hyperparameters | SDK maturity |
|---|---|---|---|---|
| **SFT** | `client.tunings.tune(base_model, training_dataset, config)` | none (just `contents`) | `epoch_count`, `adapter_size`, `learning_rate_multiplier` | GA, Gen AI SDK |
| **Preference (DPO)** | same `tunings.tune(...)` + `method="PREFERENCE_TUNING"` | `completions` list with `score` (0/1) | above + `beta` | GA, Gen AI SDK |
| **RLFT** | same `tunings.tune(...)` + `method="REINFORCEMENT_TUNING"` | `references` (no target completion) | `reward_config` + `samples_per_prompt` | Pre-GA, Gen AI SDK |

The launch method is **`tune`**, not `create`. Construct the Vertex-backed
client with `genai.Client(vertexai=True, project=..., location=...)`.

## Dataset JSONL schemas (all share the `contents` base)

Base record (SFT) — one user turn + the ground-truth model turn:

```json
{"contents": [
  {"role": "user",  "parts": [{"text": "..."}]},
  {"role": "model", "parts": [{"text": "..."}]}
], "systemInstruction": {"parts": [{"text": "..."}]}}
```

`systemInstruction` is optional. Multimodal parts use
`{"fileData": {"mimeType": "...", "fileUri": "gs://..."}}` (see
`schemas.file_part`).

**DPO** adds a `completions` array of candidate responses, each with a binary
`score` (1 = preferred, 0 = not). **RLFT** drops the target model turn entirely
(`contents` ends on the user turn) and adds a `references` string→string dict of
ground-truth metadata that the **reward function** reads to score generations —
there is no gold answer to match. See the RLFT section below for the full shape.

## SFT hyperparameters (this repo's `launch_sft_job`)

- `epoch_count` — default 2 here; small datasets overfit past a few epochs.
- `adapter_size` — LoRA rank; enum `ADAPTER_SIZE_{ONE,TWO,FOUR,EIGHT,SIXTEEN}`
  (map in `sft/tune.py:ADAPTER_MAP`). Larger = more capacity + cost + overfit risk.
- `learning_rate_multiplier` — scales the default LR.
- `validation_dataset` — optional `TuningDataset(gcs_uri=...)`.

Config object: `types.CreateTuningJobConfig(...)`;
datasets: `types.TuningDataset(gcs_uri=...)`.

## DPO (preference tuning) record + hyperparameters

Implemented in `preference/` (`launch_preference_job`, builders in
`schemas.preference_example`). Same `client.tunings.tune(...)` as SFT — both
`method="PREFERENCE_TUNING"` **and** `beta` are fields on
`types.CreateTuningJobConfig` (verified against the installed SDK, not assumed).

Record shape — `contents` ends on a **user** turn (no gold model turn); the pair
of scored responses lives in `completions`:

```json
{
  "contents": [{"role": "user", "parts": [{"text": "..."}]}],
  "completions": [
    {"score": 1, "completion": {"role": "model", "parts": [{"text": "<preferred>"}]}},
    {"score": 0, "completion": {"role": "model", "parts": [{"text": "<dispreferred>"}]}}
  ],
  "systemInstruction": {"parts": [{"text": "..."}]}
}
```

- `score` is binary: **1 = preferred, 0 = dispreferred**; exactly one of each.
  Only the `completions` turns are trained on. `systemInstruction` optional.
- **Text-only** — DPO does not support multimodal parts.
- `beta` — recommended **0.01–0.5**; lower = more aggressive updates toward the
  preferred response, `0` = no learning. All the SFT hyperparameters above
  (`epoch_count`, `adapter_size`, `learning_rate_multiplier`, `validation_dataset`)
  also apply.
- **Supported base models:** Gemini 2.5 Flash / 2.5 Flash-Lite.
- **Best practice:** SFT on the preferred responses first, then *continuous-tune*
  from that checkpoint with DPO. This repo's demo tunes the base model directly to
  stay self-contained.
- **Eval:** no single gold label → autorater pairwise **win-rate** (tuned reply
  vs. the dispreferred reference in a blind A/B judgment); see
  `preference/evaluate.py`.

## RLFT (reinforcement tuning) record + hyperparameters

Implemented in `rlft/` (`launch_rlft_job`, builders in `schemas.rlft_example`).
Same `client.tunings.tune(...)` as SFT/DPO — `method` is passed as the **string**
`"REINFORCEMENT_TUNING"` (the `TuningMethod` enum only lists SFT/PREFERENCE/
DISTILLATION, but the SDK is case-insensitive; confirmed by the SDK's own
`tests/tunings/test_tune.py`). `reward_config` (and `composite_reward_config`,
`samples_per_prompt`, `thinking_level`, `evaluate_interval`, `checkpoint_interval`,
`max_output_tokens`, `batch_size`) are fields on `types.CreateTuningJobConfig`
(verified against the installed `google-genai` 2.14.0, not assumed).

Record shape — `contents` ends on a **user** turn; ground truth lives in
`references` (a string→string dict), and there is **no** completion:

```json
{
  "systemInstruction": {"parts": [{"text": "..."}]},
  "contents": [{"role": "user", "parts": [{"text": "What is 17 * 23? End with 'Answer: <number>'."}]}],
  "references": {"ground_truth_answer": "391"}
}
```

- **Reward scorers** — one of four on `SingleReinforcementTuningRewardConfig`
  (`reward_name` + a scorer): `string_match_reward_scorer`, `autorater_scorer`,
  **`code_execution_reward_scorer`** (used here), or `cloud_run_reward_scorer`.
  `CompositeReinforcementTuningRewardConfig` combines several.
- **Code-execution contract** — the sandbox runs `python_code_snippet`, then calls
  `evaluate(example, response) -> float` with **camelCase ProtoJSON** dicts
  (`example` carries `references`/`systemInstruction`; `response` is a `Content`
  with `parts`). Rewards are **clipped to `[-1, 1]`**. The snippet must be
  **self-contained** (stdlib + numpy/pandas/sympy in the sandbox, no repo imports).
  A job **auto-stops if >80%** of reward calls error or return `NaN`. This repo
  ships `rlft/reward.py` verbatim via `inspect.getsource` — one tested function
  used both as the training reward and for offline eval.
- **Preflight** — `client.tunings.validate_reward(parent, sample_response, example,
  single_reward_config=...)` scores one example before launch; a non-null `error`
  or `NaN` means the reward is broken (`rlft/tune.py:validate_reward_config`).
- **Hyperparameters** — `samples_per_prompt` (candidate generations per prompt for
  reward comparison), plus `epoch_count`, `adapter_size`, `learning_rate_multiplier`,
  `validation_dataset` as with SFT; `thinking_level` (`HIGH`/`MINIMAL`).
- **Limits (docs):** ≤5,000 train / ≤500 val examples, ≤32,768 input/output tokens.
- **Supported base model / regions:** docs specify `gemini-3.5-flash`; tuning in
  `us-central1` / `europe-west4`; `v1beta1`. This repo's launcher defaults to
  `gemini-2.5-flash` for consistency but documents the `gemini-3.5-flash`
  recommendation — verify availability before running live.
- **Best practice:** SFT first, then *continuous-tune* with RLFT. This repo's demo
  tunes the base model directly to stay self-contained.
- **Eval:** no gold label → reuse the reward. Generate per held-out prompt, score
  with the same `evaluate`, report fraction with positive reward (accuracy); see
  `rlft/evaluate.py`.

## Job lifecycle → endpoint

- States: `JOB_STATE_{PENDING,RUNNING,SUCCEEDED,FAILED,CANCELLED}` (constants in
  `jobs.py`). Poll `client.tunings.get(name=...)`; reuse by display name via
  `client.tunings.list()` for cost control.
- Output is an **endpoint**: `job.tuned_model.endpoint` (fall back to
  `.model`). Call `client.models.generate_content(model=endpoint, ...)`.
- For thinking models, disable thinking (`thinking_budget=0`) on the tuned
  endpoint — SFT trains direct ground-truth output, so a trace only adds cost.

## Checkpointing & continuous tuning (cross-cutting, implemented)

Both are sub-features of the three services above, demonstrated by dedicated
demos (`04_checkpoints`, `05_continuous_tuning`) via shared helpers in `jobs.py`
and three keyword-only launcher params (`export_last_checkpoint_only`,
`evaluation_config`, `pre_tuned_model_checkpoint_id`; RLFT also
`checkpoint_interval`). Continuous tuning needs no base-model plumbing — pass a
tuned-model resource name as `base_model` and the SDK treats it as pre-tuned.
Full verified surface + gotchas (auto-eval `us-central1` only, 2025-07-11
base-model cutoff, Gen AI SDK only) in
[[checkpoints-and-continuous-tuning]].

## Pre-GA / drift caveats

- RLFT **is** supported in the installed `google-genai` SDK via
  `method="REINFORCEMENT_TUNING"` + `reward_config` — it is *not* REST-only. It is
  still `v1beta1`/Pre-GA, so re-verify the reward types and `method` string (and the
  base-model string) before running.
- Verify the base-model string at build time (e.g. `gemini-2.5-flash` vs a newer
  `gemini-3.x`); availability differs per tuning service and region.
- Job region must match the GCS bucket region (see [[environment]]).
