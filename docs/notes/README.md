# Session Notes — Index

Curated links to durable notes for this repo. Keep this file **under 200 lines**: links + one-line hooks only, no detail. See [CLAUDE.md](../../CLAUDE.md) for the note-taking rules.

## Topics

- [GEAP tuning overview](geap-tuning-overview.md) — tuning methods, supported models, SDK paths, job→endpoint flow.
- [Tuning APIs](tuning-apis.md) — per-service call shapes, JSONL schemas (SFT/DPO/RLFT), the four RLFT reward scorers + composite, managed-evaluation config (metric kinds, autorater, `evaluate_interval`), hyperparameters, Pre-GA caveats.
- [Checkpointing & continuous tuning](checkpoints-and-continuous-tuning.md) — `export_last_checkpoint_only`, per-checkpoint endpoints, default-checkpoint reassignment, SFT→RLFT chain.
- [Multimodal (image) SFT](multimodal-sft.md) — `fileData` record shape, reuses `launch_sft_job`, kagglehub `vision` group, GCS↔local eval mapping, sweep→val-select→test.
- [banking77 dataset](banking77-dataset.md) — 77-intent CC-BY-4.0 benchmark sourced via stdlib (no dep); the discriminating SFT dataset that replaces the saturated 5-intent demo for DOE.
- [Environment & config](environment.md) — `.env` var groups, redundant aliases, region/bucket gotchas, auth.
- [Tuned endpoints & cost](endpoints-and-cost.md) — tuned Gemini endpoints are serverless/per-token (idle ≠ hourly bill); cleanup is tidiness/quota, not runaway cost; manual teardown.
- [Toolchain & standards](toolchain.md) — uv/ruff/ty/pytest setup, ruff `ALL` ignores, non-obvious config decisions.
- [Experiment tracking](experiment-tracking.md) — automatic tuning-metric curves vs. opt-in Vertex AI Experiments (SFT + RLFT tracked demos); the `val_uri`→`/eval_*` link.
- [DOE & visualization](doe-and-visualization.md) — declarative SFT/DPO/RLFT sweep + idempotent orchestration; cross-run bars/curves, which metric comes from where, the `viz` group; the banking77 discriminating sweep + untuned baseline (before→after).
- [Resource labels](resource-labels.md) — `LABEL_KEY`/`LABEL_VALUE` → `cfg.labels`; which SDK objects accept `labels` (tuning jobs + TensorBoard only), job labels propagate to Model/Endpoint, set-at-creation-only.
- [Generative tuning demos in fresh domains](generative-tuning-domains.md) — before→after SFT (JSON extraction), DPO (concise email), RLFT (graded constrained generation); the graded-reward design that answers the two prior RLFT nulls.

## Diagrams

Reference-architecture and workflow diagrams live in [`docs/imgs/`](../imgs/) and are embedded in the [README](../../README.md) and the notes above: `reference-architecture.png`, `tuning-workflow.png`, `rlft-reward-types.png`, `evaluation.png`. Regenerate them with the PaperBanana MCP tool from the source contexts in the README sections they illustrate.

<!-- Add one line per new note: - [Title](file.md) — hook -->
