#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CONFIG:?Set CONFIG to the config file path.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the COLMAP text output directory.}"
[[ -f "${CONFIG}" ]] || { echo "CONFIG does not exist: ${CONFIG}" >&2; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGE_DIR_REL="${IMAGE_DIR_REL:-images}"
IMAGE_EXT="${IMAGE_EXT:-jpg}"

cmd=(
  "${PYTHON_BIN}" tools/export_colmap.py
  --config_file "${CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --image_dir_rel "${IMAGE_DIR_REL}"
  --image_ext "${IMAGE_EXT}"
)

if [[ -n "${DATASET:-}" ]]; then
  cmd+=(dataset="${DATASET}")
fi
if [[ -n "${DATA_ROOT:-}" ]]; then
  cmd+=(data.data_root="${DATA_ROOT}")
fi
if [[ -n "${SCENE_IDX:-}" ]]; then
  cmd+=(data.scene_idx="${SCENE_IDX}")
fi
if [[ -n "${START:-}" ]]; then
  cmd+=(data.start_timestep="${START}")
fi
if [[ -n "${END:-}" ]]; then
  cmd+=(data.end_timestep="${END}")
fi
if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
