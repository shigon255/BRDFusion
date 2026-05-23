#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${DATA_ROOT:?Set DATA_ROOT to the processed Waymo split root, e.g. data/waymo/processed/training.}"
: "${SEGFORMER_PATH:?Set SEGFORMER_PATH to the SegFormer checkout.}"

CHECKPOINT="${CHECKPOINT:-${SEGFORMER_PATH}/pretrained/segformer.b5.1024x1024.city.160k.pth}"
DEVICE="${DEVICE:-cuda:0}"

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
[[ -d "${SEGFORMER_PATH}" ]] || { echo "SEGFORMER_PATH does not exist: ${SEGFORMER_PATH}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "CHECKPOINT does not exist: ${CHECKPOINT}" >&2; exit 1; }

cmd=(
  python datasets/tools/extract_masks.py
  --data_root "${DATA_ROOT}"
  --segformer_path "${SEGFORMER_PATH}"
  --checkpoint "${CHECKPOINT}"
  --device "${DEVICE}"
)

if [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  scenes=( ${SCENES} )
  cmd+=(--scene_ids "${scenes[@]}")
elif [[ -n "${SPLIT_FILE:-}" ]]; then
  cmd+=(--split_file "${SPLIT_FILE}")
else
  : "${START:?Set START when neither SCENES nor SPLIT_FILE is provided.}"
  : "${NUM_SCENES:?Set NUM_SCENES when neither SCENES nor SPLIT_FILE is provided.}"
  cmd+=(--start_idx "${START}" --num_scenes "${NUM_SCENES}")
fi

if [[ "${PROCESS_DYNAMIC_MASK:-0}" == "1" ]]; then
  cmd+=(--process_dynamic_mask)
fi
if [[ "${IGNORE_EXISTING:-0}" == "1" ]]; then
  cmd+=(--ignore_existing)
fi
if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
