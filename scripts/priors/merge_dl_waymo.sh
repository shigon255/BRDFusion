#!/usr/bin/env bash
set -euo pipefail

: "${SCENE:?Set SCENE to the Waymo scene index.}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-51}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation diffusionlight_prior_merge --exec "${SCRIPT_PATH}" "$@"
fi
resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}

DATA_ROOT="$(resolve_path "${DATA_ROOT:-data/waymo/processed/training}")"
DL_ROOT="$(resolve_path "${DL_ROOT:-third_party/DiffusionLight-Turbo}")"
SCENE_NUM="$((10#${SCENE}))"
SCENE_PADDED="$(printf "%03d" "${SCENE_NUM}")"
INTERVAL="${INTERVAL:-2}"
MODE="${MODE:-median}"
CAM_IDS="${CAM_IDS:-0}"
SEEDS="${SEEDS:-0 37 71}"
RESIZE_METHODS="${RESIZE_METHODS:-crop}"
DATASET_CONFIG="${DATASET_CONFIG:-waymo/brdfusion_1cam}"

# shellcheck disable=SC2206
cams=( ${CAM_IDS} )
# shellcheck disable=SC2206
seeds=( ${SEEDS} )
# shellcheck disable=SC2206
resize_methods=( ${RESIZE_METHODS} )

view_times=()
for ((t=0; t<NUM_TIMESTEPS; t+=INTERVAL)); do
  frame_id="$(printf "%03d" "${t}")"
  for cam in "${cams[@]}"; do
    view_times+=("${frame_id}_${cam}")
  done
done

DL_OUTPUT_ROOT="$(resolve_path "${DL_OUTPUT_ROOT:-${DL_ROOT}/outputs/${SCENE_PADDED}}")"
OUTPUT_DIR="$(resolve_path "${OUTPUT_DIR:-${DATA_ROOT}/${SCENE_PADDED}/dlenvmap}")"
OUTPUT="$(resolve_path "${OUTPUT:-${OUTPUT_DIR}/${SCENE_PADDED}_envmap_${MODE}.exr}")"
OUTPUT_VIDEO="$(resolve_path "${OUTPUT_VIDEO:-${OUTPUT_DIR}/${SCENE_PADDED}_envmap_${MODE}.mp4}")"
INPUT_IMAGE_ROOT="$(resolve_path "${INPUT_IMAGE_ROOT:-${DATA_ROOT}/${SCENE_PADDED}/images}")"

if [[ " ${CAM_IDS} " != *" 0 "* ]]; then
  echo "Warning: CAM_IDS does not include camera 0. The merged envmap is still anchored with frame 0, camera 0, so ${DATA_ROOT}/${SCENE_PADDED}/extrinsics and intrinsics must contain camera 0 calibration." >&2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python tools/priors/merge_diffusionlight_envmaps.py \
  --dataset_kind waymo \
  --scene "${SCENE_PADDED}" \
  --cam_ids "${cams[@]}" \
  --view_times "${view_times[@]}" \
  --resize_methods "${resize_methods[@]}" \
  --seeds "${seeds[@]}" \
  --output_root "${DL_OUTPUT_ROOT}" \
  --mode "${MODE}" \
  --output "${OUTPUT}" \
  --input_image_root "${INPUT_IMAGE_ROOT}" \
  --output_video "${OUTPUT_VIDEO}" \
  --config_file "${CONFIG_FILE:-configs/omnire.yaml}" \
  dataset="${DATASET_CONFIG}" \
  data.data_root="${DATA_ROOT}" \
  data.scene_idx="${SCENE_NUM}" \
  data.start_timestep=0 \
  data.end_timestep=-1
