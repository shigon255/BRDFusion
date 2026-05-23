#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--single_cam" ]]; then
  if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 --single_cam <ori_rgb_video> <output_root> [cam_id]" >&2
    exit 1
  fi
  ORI_RGB_VIDEO="${2}"
  OUTPUT_ROOT="${3}"
  CAM_ID="${4:-0}"
  python scripts/prepare_drivestudio_eval_inputs.py intrinsic \
    --ori_rgb_video "${ORI_RGB_VIDEO}" \
    --output_root "${OUTPUT_ROOT}" \
    --num_cams 1 \
    --cam_id "${CAM_ID}"
  exit 0
fi

ORI_RGB_VIDEO="${1:?Usage: $0 <ori_rgb_video> <output_root> [tmp_root]}"
OUTPUT_ROOT="${2:?Usage: $0 <ori_rgb_video> <output_root> [tmp_root]}"
TMP_ROOT="${3:-}"

CMD=(
  python scripts/prepare_drivestudio_eval_inputs.py intrinsic
  --ori_rgb_video "${ORI_RGB_VIDEO}"
  --output_root "${OUTPUT_ROOT}"
)

if [[ -n "${TMP_ROOT}" ]]; then
  CMD+=(--tmp_root "${TMP_ROOT}")
fi

"${CMD[@]}"
