#!/usr/bin/env bash
set -euo pipefail

: "${SCENE:?Set SCENE to the Waymo scene index.}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-198}"
CAM_IDS="${CAM_IDS:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation dr_prior_generation --exec "${SCRIPT_PATH}" "$@"
fi
resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}
DR_ROOT="${DR_ROOT:-${REPO_ROOT}/third_party/cosmos1-diffusion-renderer}"
DR_SCRIPT="${DR_SCRIPT:-inv_dataset.sh}"

[[ -d "${DR_ROOT}" ]] || { echo "DR_ROOT does not exist: ${DR_ROOT}" >&2; exit 1; }
[[ -f "${DR_ROOT}/${DR_SCRIPT}" ]] || { echo "DiffusionRenderer script missing: ${DR_ROOT}/${DR_SCRIPT}" >&2; exit 1; }

DATA_ROOT="$(resolve_path "${DATA_ROOT:-${REPO_ROOT}/data/waymo/processed/training}")"
SCENE_NUM="$((10#${SCENE}))"
SCENE_PADDED="$(printf "%03d" "${SCENE_NUM}")"
export DATASET_PREFIX="$(resolve_path "${DATASET_PREFIX:-${DATA_ROOT}/${SCENE_PADDED}}")"
export CAM_IDS

cd "${DR_ROOT}"
exec bash "${DR_SCRIPT}" "${SCENE_NUM}" "${NUM_TIMESTEPS}"
