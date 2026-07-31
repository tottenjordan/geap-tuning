# Resource labels

Every Google Cloud resource this repo creates carries a **resource label** so
spend and assets can be attributed. The label is a single key/value pair sourced
from two `.env` vars — `LABEL_KEY` / `LABEL_VALUE` (defaults `project` /
`geap-tuning`) — resolved once into `TuningConfig.labels` (a `{key: value}` map,
empty when either var is unset) by `config.load_config`. Examples pass
`labels=cfg.labels` at each creation site.

Verified 2026-07-30 against `google-genai` 2.14.0, `google-cloud-aiplatform`
1.162.0.

## What actually accepts labels (installed SDK versions)

| Resource | How labels attach | Supported? |
|---|---|---|
| **Tuning jobs** (`client.tunings.tune`) | `CreateTuningJobConfig(labels=...)` | **Yes** — Vertex/GEAP path only. Labels **propagate to the generated Model + Endpoint**, so one label on the job covers job + model + endpoint. |
| **Managed TensorBoard** | `aiplatform.Tensorboard.create(labels=...)` | **Yes** — applied only when a new instance is created; a reused one keeps its existing labels. |
| `aiplatform.init` / `start_run` / `Experiment.create` | — | **No** `labels` param in this version. TensorBoard is the only labelable Experiments resource. |
| Auto-eval `EvaluationConfig` | — | Not a standalone resource; it rides on the tuning job, which already carries the labels. |
| **Cloud Storage** | — | **Out of scope.** This repo never creates a bucket; objects take only free-form `blob.metadata`, not resource labels. |

## Rules & caveats

- **Vertex/GEAP only.** `CreateTuningJobConfig(labels=...)` raises `ValueError`
  in Developer-API mode (`GOOGLE_GENAI_USE_VERTEXAI` unset/false). This repo runs
  the Vertex path, so it's fine — but a Developer-API example must not pass labels.
- **Charset (enforced server-side / by `aiplatform.utils.validate_labels`):**
  keys and values are ≤64 chars, lowercase letters, digits, `_`, `-`. `project` /
  `geap-tuning` comply, so no custom validation is needed.
- **Set at creation only.** Labels are applied when a resource is created;
  **reused** jobs/TensorBoards (found by display name) are **not** relabeled
  retroactively. Only newly created resources gain the label.
- **Backward-compatible.** With the env vars unset, `cfg.labels == {}` and every
  launcher/TensorBoard call receives `labels=None` — a no-op, unchanged behavior.
- **Not logged as Experiments params.** `run_sweep` forwards `labels` to the
  launched jobs but does not log them via `aiplatform.log_params` (that stays
  scalar input hyperparameters only).
- **DOE-managed labels.** On top of the caller's `labels`, `run_sweep` merges two
  self-describing labels onto every job it launches (`doe._run_labels`):
  `tuning_method` (the `SweepConfig.method` **lowercased** — e.g. `rlft` — to
  satisfy the charset rule above) and `experiment` (the shared Experiment name,
  only when `run_sweep(..., experiment=...)` is set). The managed keys take
  precedence over any same-named caller key, so a job's labels always reflect how
  it was actually launched. This means DOE jobs are labeled even when `cfg.labels`
  is empty (`tuning_method` is always present).

## Where it's wired

- `config.py` — `TuningConfig.labels` + `LABEL_KEY`/`LABEL_VALUE` resolution.
- `sft/tune.py`, `preference/tune.py`, `rlft/tune.py` — `labels` kwarg →
  `CreateTuningJobConfig`.
- `doe.py` — `_launch_*` and `run_sweep` thread `labels` to the launchers.
- `experiments.py` — `get_or_create_tensorboard(..., labels=...)`.
- `examples/run_*.py` — pass `labels=cfg.labels`; the config print line shows
  `labels={cfg.labels}`.

See [environment.md](environment.md) for the full `.env` variable groups.
