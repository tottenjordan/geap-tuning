# DOE results

Results-focused writeups for each **design-of-experiments (DOE)** sweep this repo
runs. Each DOE gets its own subfolder holding an explainer, its results table, and
its chart. The **shared framework mechanics** — `SweepConfig`, `run_sweep`,
idempotent reuse, `_scalar_params`, per-method headline metrics, plotting — live
once in [`../notes/doe-and-visualization.md`](../notes/doe-and-visualization.md);
these docs link to it rather than repeat it.

Writeups are added as each DOE is actually run live (a doc with a real results
table, not a placeholder). The table below is the full map of DOEs in the repo;
the **Writeup** column links the ones that have landed.

| DOE | Question it answers | Example | Notebook | Writeup |
|---|---|---|---|---|
| SFT hyperparameters | Which `epochs` × `adapter_size` tunes the best classifier? | `examples/run_doe.py` | `notebooks/10_doe.ipynb` | _pending live run_ |
| DPO hyperparameters | Which `beta` × `epochs` wins the most A/B judgments? | `examples/run_doe_dpo.py` | `notebooks/12_doe_dpo_rlft.ipynb` | _pending live run_ |
| RLFT hyperparameters | Which `epochs` × `samples_per_prompt` maximizes reward accuracy? | `examples/run_doe_rlft.py` | `notebooks/12_doe_dpo_rlft.ipynb` | _pending live run_ |
| **RLFT reward shapes** | Which **reward function** produces the most accurate math model? | `examples/run_doe_rlft_rewards.py` | `notebooks/15_doe_reward_types.ipynb` | [rlft-reward-shapes](rlft-reward-shapes/README.md) |

## Layout convention

```
docs/doe/
├── README.md                    # this index
└── <doe-name>/
    ├── README.md                # the writeup: question, how to run, results
    └── metrics.png              # the chart (from the example's --plot flag)
```

Charts are written into the DOE's own subfolder (the examples' `PLOT_PATH` points
here), keeping the repo root clean.

## Cross-references

- [`../notes/doe-and-visualization.md`](../notes/doe-and-visualization.md) — the DOE/viz framework and its design crux.
- [`../notes/environment.md`](../notes/environment.md) — region/endpoint rules, including that a tuned Gemini 3.x model lands on the `us`/`eu` multi-region endpoint.
- [`../notes/resource-labels.md`](../notes/resource-labels.md) — every DOE job is labeled with `tuning_method` + `experiment` on top of `cfg.labels`.
