#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation render --exec "${SCRIPT_PATH}" "$@"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CKPT:?Set CKPT to the checkpoint path.}"

POSTFIX="${POSTFIX:-_eval}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CALIBRATE_ENVMAP_TIMESTEP="${CALIBRATE_ENVMAP_TIMESTEP:-${START:-0}}"
CALIBRATE_ENVMAP_CAM_ID="${CALIBRATE_ENVMAP_CAM_ID:-0}"
cmd=(
  "${PYTHON_BIN}" tools/eval.py
  --resume_from "${CKPT}"
  --postfix "${POSTFIX}"
  --calibrate_envmap_timestep "${CALIBRATE_ENVMAP_TIMESTEP}"
  --calibrate_envmap_cam_id "${CALIBRATE_ENVMAP_CAM_ID}"
)

if [[ -n "${CONFIG_OVERLAYS:-}" ]]; then
  # shellcheck disable=SC2206
  overlays=( ${CONFIG_OVERLAYS} )
  for overlay in "${overlays[@]}"; do
    cmd+=(--config_overlay "${overlay}")
  done
fi
if [[ -n "${RENDER_VIDEO_POSTFIX:-}" ]]; then
  cmd+=(--render_video_postfix "${RENDER_VIDEO_POSTFIX}")
fi
if [[ -n "${VIDEO_OUTPUT_DIR:-}" ]]; then
  cmd+=(--video_output_dir "${VIDEO_OUTPUT_DIR}")
fi
if [[ -n "${METRICS_OUTPUT_DIR:-}" ]]; then
  cmd+=(--metrics_output_dir "${METRICS_OUTPUT_DIR}")
fi
if [[ -n "${START:-}" ]]; then
  cmd+=(--eval_start_timestep "${START}")
fi
if [[ -n "${NUM_FRAMES:-}" ]]; then
  cmd+=(--eval_num_timesteps "${NUM_FRAMES}")
fi
CAM_IDS="${APP_CAM_IDS:-${EVAL_CAM_IDS:-${CAM_IDS:-}}}"
if [[ -n "${CAM_IDS}" ]]; then
  cmd+=(--eval_cam_ids "${CAM_IDS}")
fi
if [[ -n "${DATASET_SOURCE:-}" ]]; then
  cmd+=(--dataset_source "${DATASET_SOURCE}")
fi
if [[ "${CALIBRATE_ENVMAP:-1}" == "1" ]]; then
  cmd+=(--calibrate_envmap)
fi
if [[ "${NO_PBR:-0}" == "1" ]]; then
  cmd+=(--no_pbr)
fi
if [[ "${LOG_METRICS:-0}" == "1" ]]; then
  cmd+=(--log_metrics)
fi

if [[ -n "${EXTERNAL_DATA_ROOT:-}" || -n "${EXTERNAL_SCENE_IDX:-}" || -n "${EXTERNAL_RELIGHT_SCENE_IDX:-}" ]]; then
  cmd+=(data.pixel_source.external_source.enable=true)
  [[ -n "${EXTERNAL_DATA_ROOT:-}" ]] && cmd+=(data.pixel_source.external_source.data_root="${EXTERNAL_DATA_ROOT}")
  [[ -n "${EXTERNAL_SCENE_IDX:-}" ]] && cmd+=(data.pixel_source.external_source.scene_idx="${EXTERNAL_SCENE_IDX}")
  [[ -n "${EXTERNAL_RELIGHT_SCENE_IDX:-}" ]] && cmd+=(data.pixel_source.external_source.relighted_scene_idx="${EXTERNAL_RELIGHT_SCENE_IDX}")
fi

cmd+=(render.render_full=true render.render_test=false render.render_novel=null data.pixel_source.load_images_only=true)

if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

printf '[INFO] Eval command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${PRINT_CMD:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
