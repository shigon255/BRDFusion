#!/usr/bin/env bash
set -euo pipefail

# Generic dynamic-asset export wrapper.
# Usage:
#   scripts/assets/export_dynamic_assets.sh [source_ckpt] [output_path]
#
# Optional env vars:
#   PYTHON_BIN=python
#   CLASSES="RigidNodes,SMPLNodes"
#   RIGID_IDS="0,2,5"
#   SMPL_IDS="1,3"
#   DEFORMABLE_IDS=""
#   EXPORT_MODE=per_object
#   MANIFEST_NAME=manifest.json
#   PREVIEW_SIZE=320
#   PREVIEW_ANCHOR_TIMESTEP=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"

SOURCE_CKPT="${SOURCE_CKPT:-${1:-}}"
OUTPUT_PATH="${OUTPUT_PATH:-${2:-}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CLASSES="${CLASSES:-RigidNodes,SMPLNodes}"
RIGID_IDS="${RIGID_IDS:-}"
SMPL_IDS="${SMPL_IDS:-}"
DEFORMABLE_IDS="${DEFORMABLE_IDS:-}"
EXPORT_MODE="${EXPORT_MODE:-per_object}"
MANIFEST_NAME="${MANIFEST_NAME:-manifest.json}"
PREVIEW_SIZE="${PREVIEW_SIZE:-320}"
PREVIEW_ANCHOR_TIMESTEP="${PREVIEW_ANCHOR_TIMESTEP:-0}"

if [[ -z "${SOURCE_CKPT}" || -z "${OUTPUT_PATH}" ]]; then
  echo "[ERR] Usage: scripts/assets/export_dynamic_assets.sh [source_ckpt] [output_path]" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "[ERR] Source checkpoint does not exist: ${SOURCE_CKPT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

CMD=(
  "${PYTHON_BIN}" tools/export_dynamic_assets.py
  --resume_from "${SOURCE_CKPT}"
  --output_path "${OUTPUT_PATH}"
  --classes "${CLASSES}"
  --export_mode "${EXPORT_MODE}"
  --manifest_name "${MANIFEST_NAME}"
  --preview_size "${PREVIEW_SIZE}"
  --preview_anchor_timestep "${PREVIEW_ANCHOR_TIMESTEP}"
)

if [[ -n "${RIGID_IDS}" ]]; then
  CMD+=(--rigid_ids "${RIGID_IDS}")
fi
if [[ -n "${SMPL_IDS}" ]]; then
  CMD+=(--smpl_ids "${SMPL_IDS}")
fi
if [[ -n "${DEFORMABLE_IDS}" ]]; then
  CMD+=(--deformable_ids "${DEFORMABLE_IDS}")
fi

echo "[INFO] Export source checkpoint: ${SOURCE_CKPT}"
echo "[INFO] Export output path: ${OUTPUT_PATH}"
echo "[INFO] Classes: ${CLASSES}"
echo "[INFO] Export mode: ${EXPORT_MODE}"

"${CMD[@]}"

echo "[OK] Dynamic asset export complete: ${OUTPUT_PATH}"
