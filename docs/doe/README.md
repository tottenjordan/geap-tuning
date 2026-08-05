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
| **SFT convention-teaching** | Can SFT teach a house normalization standard, and does `epochs` × `adapter_size` change *which* conventions it learns? | `examples/run_sft_extraction.py` | `notebooks/17_sft_extraction.ipynb` | [sft-extraction-convention](sft-extraction-convention/README.md) (discriminates *and* dissociates: v3 lifts rule-based fields to 1.0; the arbitrary `priority` relabel stays 0.0) |
| DPO hyperparameters | Which `beta` × `epochs` wins the most A/B judgments? | `examples/run_doe_dpo.py` | `notebooks/12_doe_dpo_rlft.ipynb` | _pending live run_ |
| RLFT hyperparameters | Which `epochs` × `samples_per_prompt` maximizes reward accuracy? | `examples/run_doe_rlft.py` | `notebooks/12_doe_dpo_rlft.ipynb` | _pending live run_ |
| **DPO concise email** | Which axis does DPO move on a strong base — objective concision or the subjective judge? | `examples/run_preference_email.py` | `notebooks/18_preference_email.ipynb` | [dpo-concise-email](dpo-concise-email/README.md) (modest objective gain: `mean_compression` 1.13→1.04; subjective judge flat) |
| **RLFT constrained generation** | Can a graded reward close real headroom on a strong base? | `examples/run_rlft_constrained.py` | `notebooks/19_rlft_constrained.ipynb` | [rlft-constrained-generation](rlft-constrained-generation/README.md) (third null: only the near-saturated `keywords` moved; exact `word_count` stayed 0/30) |
| **RLFT reward shapes** | Which **reward function** produces the most accurate math model? | `examples/run_doe_rlft_rewards.py` | `notebooks/15_doe_reward_types.ipynb` | [rlft-reward-shapes](rlft-reward-shapes/README.md) (null result — the cautionary tale) |
| **RLFT reward ranking** | Which reward shape best teaches an absent output behavior (ranks `format_rate`, since RLFT's only base saturates correctness) | `examples/run_doe_rlft_reward_ranking.py` | `notebooks/16_doe_reward_ranking.ipynb` | [rlft-reward-ranking](rlft-reward-ranking/README.md) (second null: headroom opened, but RLFT can't teach a ≈0%-base behavior without an SFT warm-start) |

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
