# GEAP Tuning APIs — call shapes, dataset schemas, hyperparameters

Durable reference for the three GEAP tuning services. Verified 2026-07-28.
Re-verify SDK symbols and base-model strings before acting — this surface moves.
See [[environment]] for config/auth and [[toolchain]] for tooling.

## Service matrix

| Service | Launch call | Dataset extra fields | Key hyperparameters | SDK maturity |
|---|---|---|---|---|
| **SFT** | `client.tunings.tune(base_model, training_dataset, config)` | none (just `contents`) | `epoch_count`, `adapter_size`, `learning_rate_multiplier` | GA, Gen AI SDK |
| **Preference (DPO)** | same `tunings.tune(...)` + `method="PREFERENCE_TUNING"` | `completions` list with `score` (0/1) | above + `beta` | GA, Gen AI SDK |
| **RLFT** | REST `v1beta1` tuningJobs (no stable high-level SDK) | `references` (no target completion) | reward function + reward config | Pre-GA, REST-first |

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
`score` (1 = preferred, 0 = not). **RLFT** replaces the target model turn with a
`references` field (context the model conditions on) and scores generations with
a reward function instead of matching a gold answer.

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

## Job lifecycle → endpoint

- States: `JOB_STATE_{PENDING,RUNNING,SUCCEEDED,FAILED,CANCELLED}` (constants in
  `jobs.py`). Poll `client.tunings.get(name=...)`; reuse by display name via
  `client.tunings.list()` for cost control.
- Output is an **endpoint**: `job.tuned_model.endpoint` (fall back to
  `.model`). Call `client.models.generate_content(model=endpoint, ...)`.
- For thinking models, disable thinking (`thinking_budget=0`) on the tuned
  endpoint — SFT trains direct ground-truth output, so a trace only adds cost.

## Pre-GA / drift caveats

- Pin RLFT to `v1beta1`; treat it as REST-only until a high-level SDK lands.
- Verify the base-model string at build time (e.g. `gemini-2.5-flash` vs a newer
  `gemini-3.x`); availability differs per tuning service and region.
- Job region must match the GCS bucket region (see [[environment]]).
