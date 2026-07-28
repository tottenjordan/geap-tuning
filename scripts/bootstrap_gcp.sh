#!/usr/bin/env bash
#
# Bootstrap the GCP resources the GEAP tuning examples need:
#   1. enable the aiplatform + storage APIs
#   2. create the GCS bucket (in the SAME region as the tuning location)
#
# Idempotent: safe to re-run. Reads configuration from .env, resolving the same
# redundant aliases that geap_tuning.config does. Requires an authenticated
# gcloud (`gcloud auth login`); this script does not authenticate for you.
#
# Usage:
#   ./scripts/bootstrap_gcp.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

# Load .env (auto-export every assignment).
set -a
# shellcheck disable=SC1090  # path is dynamic but validated above
source "${ENV_FILE}"
set +a

# first_set VAR1 VAR2 ... -> echo the first non-empty value among those env vars.
first_set() {
  local name
  for name in "$@"; do
    if [[ -n "${!name:-}" ]]; then
      printf '%s' "${!name}"
      return 0
    fi
  done
  return 1
}

PROJECT="$(first_set PROJECT_ID GOOGLE_CLOUD_PROJECT || true)"
LOCATION="$(first_set GOOGLE_CLOUD_LOCATION GCP_REGION || true)"
BUCKET="$(first_set GCS_BUCKET_NAME BUCKET GOOGLE_CLOUD_STORAGE_BUCKET || true)"
LOCATION="${LOCATION:-us-central1}"  # matches geap_tuning.config default

if [[ -z "${PROJECT}" ]]; then
  echo "ERROR: no project set (expected PROJECT_ID or GOOGLE_CLOUD_PROJECT in .env)." >&2
  exit 1
fi
if [[ -z "${BUCKET}" ]]; then
  echo "ERROR: no bucket set (expected GCS_BUCKET_NAME, BUCKET, or GOOGLE_CLOUD_STORAGE_BUCKET in .env)." >&2
  exit 1
fi

# Normalize the bucket to a bare name: strip a gs:// prefix and any object path.
BUCKET="${BUCKET#gs://}"
BUCKET="${BUCKET%%/*}"
BUCKET_URI="gs://${BUCKET}"

echo "Project:  ${PROJECT}"
echo "Location: ${LOCATION}"
echo "Bucket:   ${BUCKET_URI}"
echo

# 1. Enable the required APIs (no-op if already enabled).
echo "==> Enabling APIs (aiplatform, storage)..."
gcloud services enable aiplatform.googleapis.com storage.googleapis.com \
  --project="${PROJECT}"

# 2. Create the bucket only if it does not already exist.
echo "==> Ensuring bucket ${BUCKET_URI} exists in ${LOCATION}..."
if gcloud storage buckets describe "${BUCKET_URI}" --project="${PROJECT}" >/dev/null 2>&1; then
  echo "    Bucket already exists; leaving it as-is."
else
  gcloud storage buckets create "${BUCKET_URI}" \
    --project="${PROJECT}" \
    --location="${LOCATION}" \
    --uniform-bucket-level-access
  echo "    Created ${BUCKET_URI}."
fi

echo
echo "Done. The bucket region (${LOCATION}) must match the tuning location used by"
echo "the examples; keep GOOGLE_CLOUD_LOCATION consistent with it."
