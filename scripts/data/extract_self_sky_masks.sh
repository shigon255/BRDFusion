#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${DATA_ROOT:?Set DATA_ROOT to the self dataset path root, e.g. data/self/path1.}"
: "${SCENES:?Set SCENES to one or more self scene names.}"
: "${SEGFORMER_PATH:?Set SEGFORMER_PATH to the SegFormer checkout.}"

CHECKPOINT="${CHECKPOINT:-${SEGFORMER_PATH}/pretrained/segformer.b5.1024x1024.city.160k.pth}"
DEVICE="${DEVICE:-cuda:0}"

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
[[ -d "${SEGFORMER_PATH}" ]] || { echo "SEGFORMER_PATH does not exist: ${SEGFORMER_PATH}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "CHECKPOINT does not exist: ${CHECKPOINT}" >&2; exit 1; }

# shellcheck disable=SC2206
scene_list=( ${SCENES} )
for scene in "${scene_list[@]}"; do
  [[ -d "${DATA_ROOT}/${scene}/images" ]] || {
    echo "Missing self image directory: ${DATA_ROOT}/${scene}/images" >&2
    exit 1
  }

  tmp_split="$(mktemp /tmp/brdfusion_self_split.XXXXXX.csv)"
  printf "scene_id\n%s\n" "${scene}" > "${tmp_split}"

  cmd=(
    python datasets/tools/extract_masks.py
    --data_root "${DATA_ROOT}"
    --split_file "${tmp_split}"
    --segformer_path "${SEGFORMER_PATH}"
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    --rgb_dirname images
  )
  if [[ "${IGNORE_EXISTING:-0}" == "1" ]]; then
    cmd+=(--ignore_existing)
  fi
  if [[ -n "${EXTRA_OPTS:-}" ]]; then
    # shellcheck disable=SC2206
    extra=( ${EXTRA_OPTS} )
    cmd+=("${extra[@]}")
  fi

  "${cmd[@]}"
  rm -f "${tmp_split}"
done
