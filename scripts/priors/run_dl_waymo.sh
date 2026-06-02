#!/usr/bin/env bash
set -euo pipefail

: "${SCENE:?Set SCENE to the Waymo scene index.}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-198}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation diffusionlight_prior_generation --exec "${SCRIPT_PATH}" "$@"
fi
resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}
DL_ROOT="${DL_ROOT:-${REPO_ROOT}/third_party/DiffusionLight-Turbo}"
DL_SCRIPT="${DL_SCRIPT:-run_dataset.sh}"
INTERVAL="${INTERVAL:-2}"
export CAM_IDS="${CAM_IDS:-"0,1,2"}"

[[ -d "${DL_ROOT}" ]] || { echo "DL_ROOT does not exist: ${DL_ROOT}" >&2; exit 1; }
[[ -f "${DL_ROOT}/${DL_SCRIPT}" ]] || { echo "DiffusionLight script missing: ${DL_ROOT}/${DL_SCRIPT}" >&2; exit 1; }

DATA_ROOT="$(resolve_path "${DATA_ROOT:-${REPO_ROOT}/data/waymo/processed/training}")"
SCENE_NUM="$((10#${SCENE}))"
export DATASET_ROOT="$(resolve_path "${DATASET_ROOT:-${DATA_ROOT}}")"

if [[ " ${CAM_IDS} " != *" 0 "* ]]; then
  echo "Warning: CAM_IDS does not include camera 0. DiffusionLight will process only the selected cameras, but envmap merging still requires camera 0 calibration in the dataset for anchoring." >&2
fi

cd "${DL_ROOT}"
cmd=(bash "${DL_SCRIPT}" "${SCENE_NUM}" "${NUM_TIMESTEPS}" "${INTERVAL}")

exec "${cmd[@]}"
