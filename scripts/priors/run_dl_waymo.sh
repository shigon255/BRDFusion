#!/usr/bin/env bash
set -euo pipefail

: "${SCENE:?Set SCENE to the Waymo scene index.}"
: "${NUM_TIMESTEPS:?Set NUM_TIMESTEPS to the number of frames to process.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DL_ROOT="${DL_ROOT:-${REPO_ROOT}/third_party/DiffusionLight-Turbo}"
DL_SCRIPT="${DL_SCRIPT:-run_dataset.sh}"

[[ -d "${DL_ROOT}" ]] || { echo "DL_ROOT does not exist: ${DL_ROOT}" >&2; exit 1; }
[[ -f "${DL_ROOT}/${DL_SCRIPT}" ]] || { echo "DiffusionLight script missing: ${DL_ROOT}/${DL_SCRIPT}" >&2; exit 1; }

cd "${DL_ROOT}"
cmd=(bash "${DL_SCRIPT}" "${SCENE}" "${NUM_TIMESTEPS}")
if [[ -n "${INTERVAL:-}" ]]; then
  cmd+=("${INTERVAL}")
fi

exec "${cmd[@]}"
