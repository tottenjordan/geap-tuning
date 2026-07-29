# DOE (design of experiments) & multi-run visualization

Declarative hyperparameter sweeps and cross-run charts for tuning. Verified
2026-07-29. Builds directly on [experiment tracking](experiment-tracking.md) (the
sweep logs each run there) and [checkpointing](checkpoints-and-continuous-tuning.md)
(per-checkpoint curves).

**Implemented in this repo:**
[`src/geap_tuning/doe.py`](../../src/geap_tuning/doe.py) (sweep orchestration) and
[`src/geap_tuning/viz.py`](../../src/geap_tuning/viz.py) (plots), demoed by
[`examples/run_doe.py`](../../examples/run_doe.py) /
[`notebooks/10_doe.ipynb`](../../notebooks/10_doe.ipynb) (SFT sweep),
[`examples/run_doe_dpo.py`](../../examples/run_doe_dpo.py) +
[`examples/run_doe_rlft.py`](../../examples/run_doe_rlft.py) /
[`notebooks/12_doe_dpo_rlft.ipynb`](../../notebooks/12_doe_dpo_rlft.ipynb) (DPO &
RLFT sweeps), and the read-only
[`examples/run_multi_run_viz.py`](../../examples/run_multi_run_viz.py) /
[`notebooks/11_multi_run_viz.ipynb`](../../notebooks/11_multi_run_viz.ipynb) (charts
an already-tracked experiment — **zero tuning cost**).

## DOE module (`doe.py`)

Declare a full-factorial grid once; run it as a unit.

- `SweepConfig(name, base_model, method, grid, fixed)` — `method` (`"SFT"` /
  `"DPO"` / `"RLFT"`, default `"SFT"`) selects the launcher via `_LAUNCHERS`. `grid`
  maps a launcher keyword to the values to cross (SFT/DPO: `epochs`, `adapter_size`,
  `learning_rate_multiplier`; DPO also `beta`; RLFT also `samples_per_prompt`);
  `fixed` is constant kwargs applied to every run — carry **non-scalar** kwargs
  (e.g. an RLFT `reward_config`) here.
- `expand_grid(grid)` — full cross-product over **sorted** keys (deterministic);
  empty grid → `[{}]`.
- `run_spec_slug(point)` — deterministic, **resource-ID-safe** slug (sorted
  `key<value>`, lowercased, every non-`[a-z0-9-]` char → `-`, so `adapter_size` →
  `adapter-size` and `1.0` → `1-0`); empty → `"default"`. Underscores are **not**
  allowed: the slug feeds `display_name`, reused as both the tuning-job display name
  and the Vertex AI Experiments run name, which must match `[a-z0-9][a-z0-9-]{0,127}`.
- `build_run_specs(sweep)` — one `RunSpec` per point; `display_name =
  f"geap-doe-{sweep.name}-{slug}"` is the **idempotency key**; `params = fixed | point`.
- `run_sweep(client, sweep, *, train_uri, val_uri, evaluate_fn, experiment=..., ...)`
  — per spec: reuse a job matching its display name (via
  `find_tuning_job_by_display_name`) **or** launch (default is the `sweep.method`
  launcher in `_LAUNCHERS`, each called with `export_last_checkpoint_only=False` so
  per-checkpoint endpoints exist for curves), wait, resolve `tuned_endpoint`, score
  with `evaluate_fn(endpoint)`, and — when `experiment` is set — log **scalar**
  params + numeric metrics to Vertex AI Experiments. Injectable seams (`launch_fn`,
  `wait_fn`, `find_fn`) keep it unit-testable without live GCP.
- `_LAUNCHERS = {"SFT": ..., "DPO": ..., "RLFT": ...}` — method → launcher adapter
  (`launch_sft_job` / `launch_preference_job` / `launch_rlft_job`); an unknown
  `method` raises `KeyError`.
- `HEADLINE_METRIC` / `METRICS_BY_METHOD` — per-method metric keys (SFT/RLFT →
  `accuracy`, **DPO → `win_rate`**). Examples/notebooks read these so the winner
  selection and plots stay method-correct without hardcoding.
- `select_best_run(results, *, metric)` — max metric, sorted-name tie-break
  (generalizes `sft_vision.evaluate.select_best_experiment`).
- `aggregate_results(results, *, metrics)` — flatten runs to one row each
  (`{"run", "base_model", **scalar params, **metrics}`); the **single shaped
  structure** feeding both the table and the plots. Non-scalar params (RLFT
  `reward_config`) are dropped by `_scalar_params`.
- `collect_checkpoint_curve(job, evaluate_fn, *, metric)` — `(epoch, metric)` per
  exported checkpoint (needs `export_last_checkpoint_only=False`), sorted by epoch.

**SDK boundary:** the tune call stays on the `google.genai` path; the only mix is
Experiments logging, routed through [`experiments.py`](../../src/geap_tuning/experiments.py)
(the one sanctioned mix point). `init_experiment` is called by the example/notebook,
**not** by `run_sweep`.

**Methods (SFT / DPO / RLFT):** the same `run_sweep` drives all three; set
`SweepConfig.method`. Per-method specifics:
- **DPO** (`examples/run_doe_dpo.py`) — grid crosses `beta` x `epochs` on the
  preference dataset; headline is **`win_rate`** (there is no accuracy), computed
  offline by a **base-model autorater** (tuned reply vs. the dispreferred
  reference), so the driver needs a `judge_fn`.
- **RLFT** (`examples/run_doe_rlft.py`) — grid crosses `epochs` x
  `samples_per_prompt`; the **`reward_config` is a non-scalar object carried in
  `sweep.fixed`** (a declarative string-match reward here), and is **preflighted**
  once with `validate_reward_config` before the sweep spends money. Base model is
  `gemini-3.5-flash` (regional client only — the global endpoint excludes tuning).

## Visualization module (`viz.py`)

matplotlib + pandas live in the optional **`viz`** group (`uv sync --group viz`)
and are **lazy-imported** behind `_import_matplotlib` / `_import_pandas` (modeled on
`sft_vision.data._import_kagglehub`), so importing the package never requires them;
a missing dep raises `RuntimeError("... run: uv sync --group viz")`. Every plot fn
takes already-shaped rows and returns a matplotlib `Figure` (no GCP).

- `rows_to_dataframe(rows)` / `dataframe_to_rows(df)` — pandas bridges.
- `normalize_experiment_rows(rows)` — strips the `metric.`/`param.` prefixes and
  renames `run_name` → `run` from an `experiment_dataframe`, so a **read-back**
  dataframe plots identically to `aggregate_results` output. This is what makes the
  zero-cost notebook 11 path work.
- `drop_rows_missing_metrics(rows, metrics)` — drops rows with no usable value for
  **any** requested metric. A mixed experiment can hold time-series-only runs whose
  summary `accuracy`/`macro_f1` are `NaN`; without this they'd plot as empty bar
  clusters. `run_multi_run_viz.py` applies it right after `normalize_experiment_rows`.
- `plot_metric_bars(rows, *, metric)`, `plot_grouped_metric_bars(rows, *, metrics)`,
  `plot_curves(series, *, metric)` (one line per run over `(x, value)` points, fed by
  `collect_checkpoint_curve`).

Tests never render — they monkeypatch the import shims with a `MagicMock` and assert
the shaped data reaches `ax.bar` / `ax.plot`, plus that the shims raise on
`ImportError`.

## Where each plottable metric comes from (design crux)

| Source | Granularity | Fetch | Feeds |
|---|---|---|---|
| `experiment_dataframe(exp)` | one value/metric/run | programmatic, no TensorBoard | cross-run table, bars |
| `aggregate_results` rows | one value/metric/run | programmatic, TB-free | bars, table |
| `collect_checkpoint_curve` | value/epoch/run | programmatic (`list_checkpoints` + `checkpoint_endpoint`) | curve overlays |
| Managed TensorBoard time-series | value/step/run | heavy read-back API | avoided (write-only in demos) |
| **Layer-1 Monitor curves** (train/val loss) | value/step/run | **console-only — NOT via the SDK** | screenshot only |

The last row is the constraint that shapes everything: cross-run **loss** curves
can't be pulled programmatically, so cross-run *summary* bars come from
`aggregate_results` / `experiment_dataframe` and cross-run *accuracy* curves come
from our own offline per-checkpoint eval — never from Layer-1.

## Gotchas

- **`experiment_dataframe` columns are prefixed** (`metric.`, `param.`, `run_name`)
  — plot directly and `metric="accuracy"`/`label_key="run"` won't match. Run it
  through `normalize_experiment_rows` first.
- **Slugs must be Vertex-resource-ID-safe — no underscores.** The run slug feeds
  `display_name`, reused as both the tuning-job display name *and* the Experiments
  run name, which are validated against `[a-z0-9][a-z0-9-]{0,127}`. An underscore
  (from a key like `adapter_size` or a `.`→`_` float) makes `aiplatform.start_run`
  400 with `resource ID … must match the regular expression`. `run_spec_slug`
  lowercases and maps every non-`[a-z0-9-]` char to `-` for this reason.
- **Idempotency is by display name.** Re-running a sweep reuses finished jobs; to
  force a re-tune, change `sweep.name` or the grid (both change the slug).
- **Curves need all checkpoints.** `run_sweep`'s default launcher sets
  `export_last_checkpoint_only=False`; a custom `launch_fn` must too, or
  `collect_checkpoint_curve` sees only the final checkpoint.
- **Non-scalar params go in `sweep.fixed`, never the grid.** An RLFT
  `reward_config` is an object: it must not reach `run_spec_slug` (does
  `f"{k}{v}"`), the aggregate rows, or `aiplatform.log_params` (scalars only). It
  rides in `sweep.fixed` (splatted to the launcher) and is filtered out of the rows
  + Experiments params by `_scalar_params` (keeps only `str | int | float`; `bool`
  is a valid scalar). Only grid points are slugged, so `fixed` never affects the
  display name / idempotency key.
- **DPO headlines `win_rate`, not `accuracy`.** Read `HEADLINE_METRIC[method]` /
  `METRICS_BY_METHOD[method]` — calling `select_best_run` / `aggregate_results` with
  the default `accuracy` on DPO results `KeyError`s (DPO metrics are
  `{win_rate, wins, losses, ties, n}`).
