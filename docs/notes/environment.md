# Environment & config

The `.env` file (present at repo root, git-ignored, **do not commit**) holds all GCP/Gemini config. Observed on 2026-07-28. See [GEAP tuning overview](geap-tuning-overview.md) for how these feed tuning jobs.

## Variable groups

- **GCP project:** `PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_PROJECT_NUMBER`, `GCP_REGION`, `GOOGLE_CLOUD_LOCATION`.
- **API routing / auth:** `GOOGLE_GENAI_USE_VERTEXAI` (true → Vertex/GEAP path; false → AI Studio Developer API), `GOOGLE_API_KEY`, `GEMINI_API_KEY`.
- **Cloud Storage:** `GCS_BUCKET_NAME`, `BUCKET`, `GOOGLE_CLOUD_STORAGE_BUCKET`.

## Gotchas (the reason this note exists)

- **Redundant aliases.** Project id has two names (`PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`); the bucket has three (`GCS_BUCKET_NAME`, `BUCKET`, `GOOGLE_CLOUD_STORAGE_BUCKET`). They exist because different SDKs/samples read different names. Don't assume they hold the same value — pick the canonical one per example and be explicit about which you read.
- **Two locations vars** (`GCP_REGION`, `GOOGLE_CLOUD_LOCATION`) — same aliasing risk.
- **Region alignment.** Tuning docs default to `us-central1`; the auto-eval-after-tuning service is `us-central1`-only. Keep tuning region, dataset bucket region, and eval region consistent to avoid cross-region failures.
- **Gemini 3.x → `global` for INFERENCE only; tuning is regional.** Gemini 3.x models (e.g. `gemini-3.5-flash`) are served for *inference* (`generateContent`) only from the `global` endpoint — a regional client rejects the base model. BUT the `global` endpoint **does not support tuning**: per the [locations doc](https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations#global-endpoint) Global-endpoint Limitations, "Tuning" (and by extension `validate_reward` — `ValidateReinforcementTuningReward` returns `400 ... not available for your project or region` on `global`) is excluded. So **tune Gemini 3.x in a region** (`us-central1`/`europe-west4`); the resulting tuned endpoint is also regional. In this repo, `config.resolve_location(cfg, base_model)` / `genai_client(cfg, base_model=...)` encode the *inference* rule (return `global` for `gemini-N`, N≥3); **do not** pass `base_model` when building a tuning client — leave it regional. GCS upload is location-independent (`storage.Client()`). Verified 2026-07-28 against a live `validate_reward` call failing on `global`.
- **Auth for REST/CLI:** `gcloud auth print-access-token` (needs prior `gcloud auth login`). SDK paths use ADC or the API key depending on `GOOGLE_GENAI_USE_VERTEXAI`.
