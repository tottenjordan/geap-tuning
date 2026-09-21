#!/usr/bin/env bash
#
# Undeploy and delete tuned-model endpoints left behind by this repo's examples.
#
# Why this exists: every tuning job runs with `export_last_checkpoint_only=False`
# (the default, needed so `collect_checkpoint_curve` can score each checkpoint), and
# GEAP deploys ONE ENDPOINT PER EXPORTED CHECKPOINT. An 8-epoch DOE grid point
# therefore leaves ~8 endpoints behind. They are serverless (`automaticResources`,
# no machine type) so they cost nothing idle — this is a QUOTA problem, not a bill:
# Vertex caps endpoints per region, and a busy sweep day can add 25+.
#
# SAFETY: dry-run by default. It prints what it would delete and changes nothing
# unless you pass --yes. It only ever touches endpoints whose display name starts
# with --prefix (default "geap-"), so unrelated workloads in the project are never
# candidates. Deletion is irreversible; a deleted endpoint means re-tuning to get
# that checkpoint back.
#
# Usage:
#   ./scripts/cleanup_endpoints.sh                          # dry run, all geap-*
#   ./scripts/cleanup_endpoints.sh --prefix geap-doe-       # narrow the blast radius
#   ./scripts/cleanup_endpoints.sh --keep geap-sft-support-intent
#   ./scripts/cleanup_endpoints.sh --older-than 30          # only endpoints >30 days old
#   ./scripts/cleanup_endpoints.sh --prefix geap-doe- --yes # actually delete
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

PREFIX="geap-"
KEEP=""
OLDER_THAN=0
CONFIRMED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --older-than) OLDER_THAN="$2"; shift 2 ;;
    --yes) CONFIRMED=1; shift ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090  # path is dynamic but validated above
source "${ENV_FILE}"
set +a

first_set() {
  local name
  for name in "$@"; do
    if [[ -n "${!name:-}" ]]; then printf '%s' "${!name}"; return 0; fi
  done
  return 1
}

PROJECT="$(first_set PROJECT_ID GOOGLE_CLOUD_PROJECT || true)"
LOCATION="$(first_set GOOGLE_CLOUD_LOCATION GCP_REGION || true)"
LOCATION="${LOCATION:-us-central1}"  # matches geap_tuning.config default

if [[ -z "${PROJECT}" ]]; then
  echo "ERROR: no project set (expected PROJECT_ID or GOOGLE_CLOUD_PROJECT in .env)." >&2
  exit 1
fi

echo "Project:  ${PROJECT}"
echo "Location: ${LOCATION}"
echo "Prefix:   ${PREFIX}"
[[ -n "${KEEP}" ]] && echo "Keeping:  ${KEEP}"
[[ "${OLDER_THAN}" -gt 0 ]] && echo "Age:      older than ${OLDER_THAN} day(s)"
echo

CUTOFF=""
if [[ "${OLDER_THAN}" -gt 0 ]]; then
  CUTOFF="$(date -u -d "${OLDER_THAN} days ago" +%Y-%m-%dT%H:%M:%SZ)"
fi

TOTAL=$(gcloud ai endpoints list --project="${PROJECT}" --region="${LOCATION}" \
  --format="value(name)" | wc -l)
echo "==> ${TOTAL} endpoint(s) in ${LOCATION}; selecting display names starting with '${PREFIX}'..."

CANDIDATES=()
while IFS=$'\t' read -r id display created; do
  [[ -z "${id}" ]] && continue
  [[ "${display}" != "${PREFIX}"* ]] && continue
  [[ -n "${KEEP}" && "${display}" == "${KEEP}"* ]] && continue
  [[ -n "${CUTOFF}" && "${created}" > "${CUTOFF}" ]] && continue
  CANDIDATES+=("${id}|${display}|${created}")
done < <(gcloud ai endpoints list --project="${PROJECT}" --region="${LOCATION}" \
           --format="value(name,displayName,createTime)")

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "    Nothing matches; nothing to do."
  exit 0
fi

echo "    ${#CANDIDATES[@]} endpoint(s) selected:"
for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r id display created <<<"${entry}"
  echo "      ${display}  (${id}, created ${created})"
done
echo

if [[ "${CONFIRMED}" -ne 1 ]]; then
  echo "DRY RUN — nothing was deleted."
  echo "Re-run with --yes to undeploy and delete the ${#CANDIDATES[@]} endpoint(s) above."
  echo "Deletion is irreversible: recovering a checkpoint means re-running its tuning job."
  exit 0
fi

echo "==> Deleting ${#CANDIDATES[@]} endpoint(s)..."
DELETED=0
FAILED=0
for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r id display _created <<<"${entry}"
  echo "    ${display} (${id})"
  # A model must be undeployed before its endpoint can be deleted.
  while read -r deployed_id; do
    [[ -z "${deployed_id}" ]] && continue
    gcloud ai endpoints undeploy-model "${id}" --project="${PROJECT}" \
      --region="${LOCATION}" --deployed-model-id="${deployed_id}" --quiet \
      || echo "      WARN: undeploy ${deployed_id} failed; continuing"
  done < <(gcloud ai endpoints describe "${id}" --project="${PROJECT}" \
             --region="${LOCATION}" --format="value(deployedModels[].id)" | tr ';' '\n')

  if gcloud ai endpoints delete "${id}" --project="${PROJECT}" --region="${LOCATION}" --quiet; then
    DELETED=$((DELETED + 1))
  else
    echo "      ERROR: delete failed"
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "Done. Deleted ${DELETED}, failed ${FAILED}."
echo
echo "HEADS UP: the tuning JOBS remain (tuningJobs exposes no delete), and each one"
echo "still reports the endpoint you just deleted. So reuse-by-display-name will"
echo "find the job, skip launching, and then 404 at inference. To re-run any of"
echo "these examples, bump GEAP_RUN_SUFFIX in .env (e.g. -v2) so every driver gets a"
echo "fresh name at once. See docs/notes/endpoints-and-cost.md."
