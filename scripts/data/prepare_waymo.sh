#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${RAW_ROOT:?Set RAW_ROOT to the Waymo raw TFRecord directory.}"
: "${TARGET_ROOT:?Set TARGET_ROOT to the processed Waymo output root.}"

SPLIT="${SPLIT:-training}"
WORKERS="${WORKERS:-8}"
PROCESS_KEYS="${PROCESS_KEYS:-images lidar calib pose dynamic_masks objects}"

cmd=(
  python datasets/preprocess.py
  --data_root "${RAW_ROOT}"
  --target_dir "${TARGET_ROOT}"
  --dataset waymo
  --split "${SPLIT}"
  --workers "${WORKERS}"
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

# shellcheck disable=SC2206
keys=( ${PROCESS_KEYS} )
cmd+=(--process_keys "${keys[@]}")

if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
