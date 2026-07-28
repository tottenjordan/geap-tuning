# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of **working, runnable examples** demonstrating **Gemini Enterprise Agent Platform (GEAP)** model tuning services. GEAP is Google Cloud's rebrand/evolution of the Vertex AI Agent Platform surface; its tuning APIs are reached through the same `aiplatform.googleapis.com` endpoints and the Python SDKs below.

## Code standards (read first)

**Always refer to [CODE_STANDARDS.md](CODE_STANDARDS.md) before writing code or making environment changes**, and link to it when you do. Non-negotiables: `uv` for all package management (never bare `pip`/`python`), `ruff` for lint+format, `ty` for type checks, `pytest` for tests, and **never** add `Co-Authored-By` trailers to commits/PRs.

## Layout & commands

Python `src/` layout, package `geap_tuning`, managed by `uv` (Python 3.12+). Config and tool settings live in `pyproject.toml`; `Makefile` wraps the common flows:

- `make dev` — `uv sync --all-groups` (install runtime + lint + test groups)
- `make lint` — `ruff format --check` + `ruff check` + `ty check src/`
- `make format` — `ruff format .`
- `make test` — `uv run pytest` (coverage on `geap_tuning`)
- Single test: `uv run pytest tests/test_smoke.py::test_main_runs`

Runtime deps: `google-genai`, `google-cloud-aiplatform` (Vertex/Agent Platform SDK), `python-dotenv`. The repo is still early — one placeholder entry point (`geap_tuning.main`) and a smoke test; add tuning examples as their own modules under `src/geap_tuning/`.

## Tuning surface being demonstrated

This repo demonstrates GEAP's **three** tuning services (one example subpackage each under `src/geap_tuning/`):
- **Supervised fine-tuning (SFT)** — the core method; teaches a new skill/behavior from labeled JSONL examples (`contents` records). Implemented (`sft/`).
- **Preference tuning (DPO)** — builds on SFT using preference pairs; same `client.tunings.tune(...)` call with `method="PREFERENCE_TUNING"` + a `beta` hyperparameter, and `completions`/`score` in each record. Planned (`preference/`).
- **Reinforcement learning fine-tuning (RLFT)** — trains against a programmable **reward function** over a `references` dataset (no target completion). Pre-GA, REST-first (`v1beta1`); no stable high-level SDK yet. Planned (`rlft/`).

Checkpoints and continuous tuning are cross-cutting sub-features of these services, not separate services. Data modalities: text, image, audio, document. Datasets are JSONL files staged in Cloud Storage. Full API shapes are in [docs/notes/tuning-apis.md](docs/notes/tuning-apis.md).

## Two SDK paths (pick deliberately; do not mix in one example)

Examples generally follow one of two client styles. Keep each example on one path so it reads cleanly:

1. **Google Gen AI SDK** — `from google import genai`; `client.tunings.tune/get/list(...)` (the launch method is `tune`, not `create`). Routes to Vertex when constructed with `vertexai=True` (or `GOOGLE_GENAI_USE_VERTEXAI=true`). This is the newer, preferred surface and the one this repo uses.
2. **Agent Platform / Vertex SDK for Python** — `import vertexai` + `from vertexai.tuning import sft`; `sft.SupervisedTuningJob(...)`. Older but appears throughout Google's tuning docs.

A tuning job's output is an **endpoint** (`tuning_job.tuned_model.endpoint`); you call `generate_content` against that endpoint, not a model name. For thinking models, disable thinking / set the minimum thinking budget when calling a tuned model — SFT trains the model to mimic ground truth without a thinking trace.

## Environment / configuration

Config lives in `.env` (present, git-ignored — never commit it). Load it before running any example. Key groups:
- **GCP project**: `PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_PROJECT_NUMBER`, `GCP_REGION`, `GOOGLE_CLOUD_LOCATION`.
- **API routing / auth**: `GOOGLE_GENAI_USE_VERTEXAI` (true → Vertex/GEAP path), `GOOGLE_API_KEY` / `GEMINI_API_KEY` (AI Studio / Developer API path).
- **Cloud Storage** (dataset + checkpoint staging): `GCS_BUCKET_NAME`, `BUCKET`, `GOOGLE_CLOUD_STORAGE_BUCKET`.

Note the redundant aliases (`PROJECT_ID`/`GOOGLE_CLOUD_PROJECT`, three bucket vars). When writing an example, decide which var name it reads and be consistent; don't assume all three buckets point at the same place.

Most tuning docs default examples to `us-central1`. The Gen AI evaluation service that can auto-run after a job is (per docs) available in `us-central1` — prefer it for examples that chain tuning → eval.

Auth for CLI/REST calls: `gcloud auth print-access-token` (user must have run `gcloud auth login`). If you need the user to authenticate, suggest they run `! gcloud auth login` in the prompt.

## Session notes (required workflow)

Persist durable findings under `docs/notes/`, organized by topic:
- **One topic per file.** Group related sub-topics; split when a file covers genuinely distinct topics. Cross-link with relative Markdown links.
- **`docs/notes/README.md` is the top-level index** — a curated list of links to the key note files. Keep it **under 200 lines**; it points to detail, it does not contain it.
- Save a note only when the info **outlives this conversation and isn't recoverable** from the repo, git history, this file, or existing docs (e.g. "the tuning job region must match the bucket region", "SDK X's `list()` returns objects not names").
- **Check for an existing note on the topic and update it** instead of creating a duplicate. **Delete notes that turn out wrong or stale.**
- Notes reflect what was true when written. If a note names a file, flag, or SDK symbol, **re-verify it still exists before acting on it.**
