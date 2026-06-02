#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CKPT:?Set CKPT to the checkpoint path.}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the output PLY path.}"
[[ -f "${CKPT}" ]] || { echo "CKPT does not exist: ${CKPT}" >&2; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
  "${PYTHON_BIN}" tools/export_rigid_canonical.py
  --resume_from "${CKPT}"
  --output_path "${OUTPUT_PATH}"
)

if [[ -n "${INSTANCE_IDS:-}" ]]; then
  cmd+=(--instance_ids "${INSTANCE_IDS}")
fi
if [[ "${SPLIT_BY_INSTANCE:-0}" == "1" ]]; then
  cmd+=(--split_by_instance)
fi
if [[ -n "${ALPHA_THRESH:-}" ]]; then
  cmd+=(--alpha_thresh "${ALPHA_THRESH}")
fi
if [[ -n "${COLOR_SOURCE:-}" ]]; then
  cmd+=(--color_source "${COLOR_SOURCE}")
fi

exec "${cmd[@]}"
