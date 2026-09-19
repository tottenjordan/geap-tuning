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

`scripts/cleanup_endpoints.sh` is the teardown counterpart to
`scripts/bootstrap_gcp.sh`. It reads the same `.env`, is **dry-run by default**,
and only ever considers display names matching `--prefix` (default `geap-`), so
unrelated workloads in the project are never candidates:

```bash
./scripts/cleanup_endpoints.sh                            # dry run, all geap-*
./scripts/cleanup_endpoints.sh --prefix geap-doe-         # narrow the blast radius
./scripts/cleanup_endpoints.sh --keep geap-sft-support-intent
./scripts/cleanup_endpoints.sh --older-than 30            # only endpoints >30 days old
./scripts/cleanup_endpoints.sh --prefix geap-doe- --yes   # actually delete
```

It undeploys each model before deleting its endpoint, which is the ordering the
API requires. Deletion is irreversible: getting a checkpoint back means re-running
its tuning job.

> **After a cleanup, reuse-by-display-name yields a dangling endpoint.** Verified
> 2026-09-19 after deleting 65 endpoints: the tuning **jobs** survive (they are not
> deletable), `find_tuning_job_by_display_name` still matches them, and
> `tuned_endpoint(job)` still returns the endpoint resource name — which no longer
> exists. The example therefore skips launching and then **404s at inference**
> rather than transparently re-tuning. There is no cheap fix: detecting it would
> cost an endpoint-existence API call on every reuse. The remedy is to **change the
> display name** (for a sweep, `sweep.name`) so a fresh job is launched. The
> cleanup script prints this warning when it finishes.

**Why you will need it.** Every job runs with `export_last_checkpoint_only=False`
(the default, so `collect_checkpoint_curve` can score each checkpoint) and GEAP
deploys **one endpoint per exported checkpoint**. An 8-epoch DOE grid point leaves
~8 endpoints. Measured on this project 2026-09-18: **78 endpoints in
`us-central1`, 65 of them from this repo**, with 26 created on a single heavy
sweep day. Since endpoints are serverless (`automaticResources`, no machine type —
verified on a live endpoint) this costs nothing idle; the risk is the **per-region
endpoint quota**, where hitting the ceiling mid-sweep surfaces as a confusing
deployment failure rather than an obvious quota error.

The equivalent manual sequence, if you would rather do it by hand:

```bash
gcloud ai endpoints list --region="$GCP_REGION" --project="$PROJECT_ID"
gcloud ai endpoints undeploy-model <ENDPOINT_ID> --region="$GCP_REGION" \
  --deployed-model-id=<DEPLOYED_MODEL_ID> --project="$PROJECT_ID"
gcloud ai endpoints delete <ENDPOINT_ID> --region="$GCP_REGION" --project="$PROJECT_ID"
```

Terminal tuning **jobs** are just metadata — cheap to keep, and keeping them is
what makes the examples' reuse-by-display-name work. Don't delete the jobs — and
in fact you **can't**: `tuningJobs` exposes **no `delete` method** (a REST
`DELETE …/tuningJobs/<id>` returns `404 Method not found` on both `v1` and
`v1beta1`; only a *running* job can be *cancelled*). So "cleaning up a job"
always means deleting the **model + endpoint it produced** — fetch them from
`client.tunings.get(name=…).tuned_model` (`.model`, `.endpoint`), then either the
two-step `gcloud` above or a one-shot REST delete of the endpoint with
`?force=true` (undeploys before deleting) followed by `…/models/<id>` delete.
