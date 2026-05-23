#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CKPT:?Set CKPT to the checkpoint path.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the point-cloud export/render directory.}"
[[ -f "${CKPT}" ]] || { echo "CKPT does not exist: ${CKPT}" >&2; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-python}"
VIS_TIMESTEP="${VIS_TIMESTEP:-0}"
FPS="${FPS:-10}"
RADIUS="${RADIUS:-0.006}"
POINTS_PER_PIXEL="${POINTS_PER_PIXEL:-128}"
MAX_POINTS="${MAX_POINTS:-2000000}"

mkdir -p "${OUTPUT_DIR}"

PLY_PATH="${PLY_PATH:-${OUTPUT_DIR}/scene_t${VIS_TIMESTEP}.ply}"
CAMERA_JSON="${CAMERA_JSON:-${OUTPUT_DIR}/cameras_t${VIS_TIMESTEP}.json}"
RENDER_DIR="${RENDER_DIR:-${OUTPUT_DIR}/render_t${VIS_TIMESTEP}}"

"${PYTHON_BIN}" -m tools.extract \
  --resume_from "${CKPT}" \
  --gaussian_output_path "${PLY_PATH}" \
  --camera_output_path "${CAMERA_JSON}" \
  --vis_timestep "${VIS_TIMESTEP}" \
  ${EXTRACT_OPTS:-}

render_cmd=(
  "${PYTHON_BIN}" -m tools.render_pointcloud_views
  --ply_path "${PLY_PATH}"
  --camera_json "${CAMERA_JSON}"
  --output_dir "${RENDER_DIR}"
  --fps "${FPS}"
  --radius "${RADIUS}"
  --points_per_pixel "${POINTS_PER_PIXEL}"
  --max_points "${MAX_POINTS}"
)
if [[ "${BG_WHITE:-1}" == "1" ]]; then
  render_cmd+=(--bg_white)
fi

exec "${render_cmd[@]}"
