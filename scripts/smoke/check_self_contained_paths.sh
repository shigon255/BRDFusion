#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

bad=$(
  rg -n \
    "/project/yi-ray|/project2/yi-ray/drivestudio" \
    README.md docs scripts tools configs datasets models utils third_party \
    --glob '!scripts/smoke/check_self_contained_paths.sh' \
    2>/dev/null || true
)

if [[ -n "${bad}" ]]; then
  echo "Found stale absolute paths to original workbenches:" >&2
  echo "${bad}" >&2
  exit 1
fi

echo "Self-contained path check OK"
