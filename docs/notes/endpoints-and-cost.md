# Tuned endpoints & what they actually cost

Durable correction to a common misconception about post-tuning cleanup.
Established 2026-07-29 while inventorying deployed endpoints for this project.
See [[checkpoints-and-continuous-tuning]] for how per-checkpoint endpoints arise
and [[environment]] for region/bucket setup.

## The cost model (the part that's easy to get wrong)

A GEAP tuning job produces a **tuned Gemini model** served on **Google-managed,
serverless infrastructure** — billed **per token**, exactly like the base model.
There is **no dedicated node** behind a tuned Gemini endpoint, so an **idle
endpoint is not accruing hourly charges**. You pay when you call it, not while it
sits.

This is the opposite of a **dedicated custom-model deployment** (e.g. a container
served via `vertex-custom-serve-v2` on a Vertex `Endpoint`), which provisions
real machines and **does bill per node-hour whether or not it serves traffic**.
Do not carry that mental model over to tuned Gemini endpoints.

## Why clean up at all, then

Cleanup is about **tidiness and quota**, not a runaway bill:

- **Tuned-model / endpoint quota** — every succeeded job leaves a tuned model +
  endpoint, and the checkpoint demos (`run_checkpoints.py`, `run_continuous_tuning.py`)
  leave **one endpoint per exported checkpoint**. These count against per-region
  limits and clutter `gcloud ai endpoints list`.
- **Inventory hygiene** — fewer stale endpoints means the display-name →
  live-endpoint mapping stays legible for the next person.

So skipping cleanup does **not** "cost real money" the way the earlier runbook
framing (`/home/user/.claude/plans/*` Step 3) implied. Treat Step 3 as a
tidiness/quota task, not an urgent money-stopper.

## What still does cost money

- **Per-token inference** against any endpoint you actually call (base, tuned, or
  checkpoint) — the normal Gemini token price.
- **The tuning run itself** — one-time training cost when the job runs.
- **GCS storage** — staged datasets + exported checkpoints under the bucket sit
  at storage rates until pruned (small, but non-zero).
- **Any dedicated custom-model deployment** — hourly per-node, as above. This repo
  ships none, but if one is ever added, *that* is the real hourly-cost case.

## Teardown reality

The repo has **create-only tooling** (`scripts/bootstrap_gcp.sh`) and **no
teardown script**, so removing endpoints is manual:

```bash
gcloud ai endpoints list --region="$GCP_REGION" --project="$PROJECT_ID"
gcloud ai endpoints undeploy-model <ENDPOINT_ID> --region="$GCP_REGION" \
  --deployed-model-id=<DEPLOYED_MODEL_ID> --project="$PROJECT_ID"
gcloud ai endpoints delete <ENDPOINT_ID> --region="$GCP_REGION" --project="$PROJECT_ID"
```

Terminal tuning **jobs** are just metadata — cheap to keep, and keeping them is
what makes the examples' reuse-by-display-name work. Don't delete the jobs.
