#!/usr/bin/env bash
set -euo pipefail

: "${SCENE:?Set SCENE to the self scene name.}"
: "${NUM_TIMESTEPS:?Set NUM_TIMESTEPS to the number of frames to process.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DR_ROOT="${DR_ROOT:-${REPO_ROOT}/third_party/cosmos1-diffusion-renderer}"
DR_SCRIPT="${DR_SCRIPT:-inv_dataset_self.sh}"
PATH_NAME="${PATH_NAME:-.}"

[[ -d "${DR_ROOT}" ]] || { echo "DR_ROOT does not exist: ${DR_ROOT}" >&2; exit 1; }
[[ -f "${DR_ROOT}/${DR_SCRIPT}" ]] || { echo "DiffusionRenderer script missing: ${DR_ROOT}/${DR_SCRIPT}" >&2; exit 1; }

cd "${DR_ROOT}"
if [[ -n "${DATA_ROOT:-}" ]]; then
  export DATASET_BASE="${DATA_ROOT}"
fi
cmd=(bash "${DR_SCRIPT}" "${SCENE}" "${PATH_NAME}" "${NUM_TIMESTEPS}")
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
