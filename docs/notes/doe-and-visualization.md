# DOE (design of experiments) & multi-run visualization

Declarative hyperparameter sweeps and cross-run charts for tuning. Verified
2026-07-29. Builds directly on [experiment tracking](experiment-tracking.md) (the
sweep logs each run there) and [checkpointing](checkpoints-and-continuous-tuning.md)
(per-checkpoint curves).

**Implemented in this repo:**
[`src/geap_tuning/doe.py`](../../src/geap_tuning/doe.py) (sweep orchestration) and
[`src/geap_tuning/viz.py`](../../src/geap_tuning/viz.py) (plots), demoed by
[`examples/run_doe.py`](../../examples/run_doe.py) /
[`notebooks/10_doe.ipynb`](../../notebooks/10_doe.ipynb) (runs a sweep) and the
read-only [`examples/run_multi_run_viz.py`](../../examples/run_multi_run_viz.py) /
[`notebooks/11_multi_run_viz.ipynb`](../../notebooks/11_multi_run_viz.ipynb) (charts
an already-tracked experiment — **zero tuning cost**).

## DOE module (`doe.py`)

Declare a full-factorial grid once; run it as a unit.

- `SweepConfig(name, base_model, grid, fixed)` — `grid` maps a `launch_sft_job`
  keyword (`epochs`, `adapter_size`, `learning_rate_multiplier`) to the values to
  cross; `fixed` is constant kwargs applied to every run.
- `expand_grid(grid)` — full cross-product over **sorted** keys (deterministic);
  empty grid → `[{}]`.
- `run_spec_slug(point)` — deterministic, display-name-safe slug (sorted `key<value>`,
  non-alnum → `_`, so `1.0` → `1_0`); empty → `"default"`.
- `build_run_specs(sweep)` — one `RunSpec` per point; `display_name =
  f"geap-doe-{sweep.name}-{slug}"` is the **idempotency key**; `params = fixed | point`.
- `run_sweep(client, sweep, *, train_uri, val_uri, evaluate_fn, experiment=..., ...)`
  — per spec: reuse a job matching its display name (via
  `find_tuning_job_by_display_name`) **or** launch (default dispatches
  `launch_sft_job(..., export_last_checkpoint_only=False)` so per-checkpoint
  endpoints exist for curves), wait, resolve `tuned_endpoint`, score with
  `evaluate_fn(endpoint)`, and — when `experiment` is set — log params + numeric
  metrics to Vertex AI Experiments. Injectable seams (`launch_fn`, `wait_fn`,
  `find_fn`) keep it unit-testable without live GCP.
- `select_best_run(results, *, metric)` — max metric, sorted-name tie-break
  (generalizes `sft_vision.evaluate.select_best_experiment`).
- `aggregate_results(results, *, metrics)` — flatten runs to one row each
  (`{"run", "base_model", **params, **metrics}`); the **single shaped structure**
  feeding both the table and the plots.
- `collect_checkpoint_curve(job, evaluate_fn, *, metric)` — `(epoch, metric)` per
  exported checkpoint (needs `export_last_checkpoint_only=False`), sorted by epoch.

**SDK boundary:** the tune call stays on the `google.genai` path; the only mix is
Experiments logging, routed through [`experiments.py`](../../src/geap_tuning/experiments.py)
(the one sanctioned mix point). `init_experiment` is called by the example/notebook,
**not** by `run_sweep`.

**Scope:** SFT only. DPO/RLFT sweeps are a follow-up — the sketched design is a
launcher registry keyed on the sweep's `method` dispatching to
`launch_preference_job` / `launch_rlft_job`.

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
- **Idempotency is by display name.** Re-running a sweep reuses finished jobs; to
  force a re-tune, change `sweep.name` or the grid (both change the slug).
- **Curves need all checkpoints.** `run_sweep`'s default launcher sets
  `export_last_checkpoint_only=False`; a custom `launch_fn` must too, or
  `collect_checkpoint_curve` sees only the final checkpoint.
