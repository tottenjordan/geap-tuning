# Notebooks

Thin, runnable walkthroughs of the GEAP tuning services. Every notebook is a
**thin demo** — each cell calls into the tested [`geap_tuning`](../src/geap_tuning)
package rather than re-implementing logic — and mirrors the matching
[`examples/run_*.py`](../examples) driver. See the top-level
[README](../README.md) for setup and the reference-architecture / workflow
diagrams.

> **Requires live GCP and incurs tuning cost.** Fill in `.env`, run
> `gcloud auth login`, and (for managed evaluation) keep the region on
> `us-central1`. Notebooks are committed **output-stripped**.

## Contents

| # | Notebook | What it covers | Pairs with |
|---|----------|----------------|------------|
| 01 | [`01_sft.ipynb`](01_sft.ipynb) | **Supervised fine-tuning (SFT)** — teach a skill from labeled `contents` records | [`run_sft.py`](../examples/run_sft.py) |
| 02 | [`02_preference_tuning.ipynb`](02_preference_tuning.ipynb) | **Preference tuning (DPO)** — learn *how* to answer from preference pairs (`completions` + `score`) | [`run_preference.py`](../examples/run_preference.py) |
| 03 | [`03_rlft.ipynb`](03_rlft.ipynb) | **Reinforcement learning fine-tuning (RLFT)** — train against a code-execution reward over `references` | [`run_rlft.py`](../examples/run_rlft.py) |
| 04 | [`04_checkpoints.ipynb`](04_checkpoints.ipynb) | **Checkpointing** — export per-epoch checkpoints, run per-checkpoint inference, reassign the default | [`run_checkpoints.py`](../examples/run_checkpoints.py) |
| 05 | [`05_continuous_tuning.ipynb`](05_continuous_tuning.ipynb) | **Continuous tuning** — chain **SFT → RLFT** by continuing from a tuned model | [`run_continuous_tuning.py`](../examples/run_continuous_tuning.py) |
| 06 | [`06_rlft_reward_types.ipynb`](06_rlft_reward_types.ipynb) | **RLFT reward types** — string-match, autorater, cloud-run (documented-only), and a weighted **composite** reward | [`run_rlft_reward_types.py`](../examples/run_rlft_reward_types.py) |
| 07 | [`07_advanced_eval.ipynb`](07_advanced_eval.ipynb) | **Advanced managed evaluation** — LLM-judge / computation / predefined metrics + autorater & inference config | [`run_advanced_eval.py`](../examples/run_advanced_eval.py) |
| 08 | [`08_experiment_tracking.ipynb`](08_experiment_tracking.ipynb) | **Experiment tracking** — log params + offline metrics to Vertex AI Experiments; opt-in Managed TensorBoard time-series | [`run_experiment_tracking.py`](../examples/run_experiment_tracking.py) |

## Suggested order

Start with **01 → 02 → 03** for the three core tuning methods, then the
cross-cutting features: **04** (checkpointing) and **05** (continuous tuning).
**06** and **07** go deeper on RLFT reward design and the managed Evaluation
service respectively, and can be read on their own once you've done 03. **08**
adds opt-in Vertex AI Experiments / Managed TensorBoard tracking on top of any
tuning run and reuses the checkpoint flow from **04**.

## Reference

- API call shapes, JSONL schemas, hyperparameters — [`docs/notes/tuning-apis.md`](../docs/notes/tuning-apis.md)
- Checkpointing & continuous tuning surface — [`docs/notes/checkpoints-and-continuous-tuning.md`](../docs/notes/checkpoints-and-continuous-tuning.md)
- All session notes — [`docs/notes/`](../docs/notes/README.md)
