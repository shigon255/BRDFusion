#!/usr/bin/env bash
set -euo pipefail

: "${SCENE:?Set SCENE to the self scene name.}"
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

DATA_ROOT="$(resolve_path "${DATA_ROOT:-data/self/path1_fixed_tree_gamma_full}")"
DL_ROOT="$(resolve_path "${DL_ROOT:-third_party/DiffusionLight-Turbo}")"
PATH_NAME="${PATH_NAME:-.}"
INTERVAL="${INTERVAL:-2}"
MODE="${MODE:-median}"
CAM_IDS="${CAM_IDS:-0}"
SEEDS="${SEEDS:-0 37 71}"
RESIZE_METHODS="${RESIZE_METHODS:-crop}"
DATASET_CONFIG="${DATASET_CONFIG:-self/brdfusion_1cam}"

cam_tag_for_id() {
  case "$1" in
    0|Camera_Center) printf 'Camera_Center' ;;
    1|Cam_Left) printf 'Cam_Left' ;;
    2|Cam_Right) printf 'Cam_Right' ;;
    *) echo "Unsupported self camera '$1'. Use 0 1 2 or Camera_Center Cam_Left Cam_Right." >&2; return 1 ;;
  esac
}
cam_id_for_value() {
  case "$1" in
    0|Camera_Center) printf '0' ;;
    1|Cam_Left) printf '1' ;;
    2|Cam_Right) printf '2' ;;
    *) echo "Unsupported self camera '$1'. Use 0 1 2 or Camera_Center Cam_Left Cam_Right." >&2; return 1 ;;
  esac
}

cam_values="${CAM_NAMES:-${CAM_IDS}}"
# shellcheck disable=SC2206
cam_values_arr=( ${cam_values} )
cams=()
cam_tags=()
for value in "${cam_values_arr[@]}"; do
  cams+=("$(cam_id_for_value "${value}")")
  cam_tags+=("$(cam_tag_for_id "${value}")")
done
# shellcheck disable=SC2206
seeds=( ${SEEDS} )
# shellcheck disable=SC2206
resize_methods=( ${RESIZE_METHODS} )

view_times=()
for ((t=0; t<NUM_TIMESTEPS; t+=INTERVAL)); do
  frame_id="$(printf "%03d" "${t}")"
  for cam_tag in "${cam_tags[@]}"; do
    view_times+=("${cam_tag}_${frame_id}")
  done
done

DL_OUTPUT_ROOT="$(resolve_path "${DL_OUTPUT_ROOT:-${DL_ROOT}/outputs/${PATH_NAME}/${SCENE}}")"
OUTPUT_DIR="$(resolve_path "${OUTPUT_DIR:-${DATA_ROOT}/${SCENE}/dlenvmap}")"
OUTPUT="$(resolve_path "${OUTPUT:-${OUTPUT_DIR}/${SCENE}_envmap_${MODE}.exr}")"
OUTPUT_VIDEO="$(resolve_path "${OUTPUT_VIDEO:-${OUTPUT_DIR}/${SCENE}_envmap_${MODE}.mp4}")"
INPUT_IMAGE_ROOT="$(resolve_path "${INPUT_IMAGE_ROOT:-${DATA_ROOT}/${SCENE}/images}")"

if [[ " ${cam_values} " != *" 0 "* && " ${cam_values} " != *" Camera_Center "* ]]; then
  echo "Warning: selected cameras do not include camera 0 / Camera_Center. The merged envmap is still anchored with frame 0, camera 0, so ${DATA_ROOT}/${SCENE} must contain camera 0 calibration." >&2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python tools/priors/merge_diffusionlight_envmaps.py \
  --dataset_kind self \
  --scene "${SCENE}" \
  --cam_ids "${cams[@]}" \
  --view_times "${view_times[@]}" \
  --resize_methods "${resize_methods[@]}" \
  --seeds "${seeds[@]}" \
  --output_root "${DL_OUTPUT_ROOT}" \
  --mode "${MODE}" \
  --output "${OUTPUT}" \
  --input_image_root "${INPUT_IMAGE_ROOT}" \
  --output_video "${OUTPUT_VIDEO}" \
  --frame_index_base 0 \
  --config_file "${CONFIG_FILE:-configs/omnire.yaml}" \
  dataset="${DATASET_CONFIG}" \
  data.data_root="${DATA_ROOT}" \
  data.scene_idx="${SCENE}" \
  data.start_timestep=0 \
  data.end_timestep=-1
