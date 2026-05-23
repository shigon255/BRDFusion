#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--single_cam" ]]; then
  # if [[ $# -lt 8 || $# -gt 9 ]]; then
  #   echo "Usage: $0 --single_cam <albedo> <normal> <normalized_depth> <roughness> <metallic> <envmap> <output_root> [cam_id]" >&2
  #   exit 1
  # fi

  ALBEDO_VIDEO="${2}"
  NORMAL_VIDEO="${3}"
  NORMALIZED_DEPTH_VIDEO="${4}"
  ROUGHNESS_VIDEO="${5}"
  METALLIC_VIDEO="${6}"
  ENVMAP="${7}"
  OUTPUT_ROOT="${8}"
  CAM_ID="${9:-0}"
  SEED="${10:-0}"

  python scripts/prepare_drivestudio_eval_inputs.py relight \
    --albedo_video "${ALBEDO_VIDEO}" \
    --normal_video "${NORMAL_VIDEO}" \
    --normalized_depth_video "${NORMALIZED_DEPTH_VIDEO}" \
    --roughness_video "${ROUGHNESS_VIDEO}" \
    --metallic_video "${METALLIC_VIDEO}" \
    --envmap "${ENVMAP}" \
    --output_root "${OUTPUT_ROOT}" \
    --num_cams 1 \
    --cam_id "${CAM_ID}" \
    --seed "${SEED}"
  exit 0
fi

if [[ $# -lt 9 ]]; then
  echo "Usage: $0 <albedo> <normal> <normalized_depth> <roughness> <metallic> <envmap_0> <envmap_1> <envmap_2> <output_root> [tmp_root]" >&2
  exit 1
fi

ALBEDO_VIDEO="${1}"
NORMAL_VIDEO="${2}"
NORMALIZED_DEPTH_VIDEO="${3}"
ROUGHNESS_VIDEO="${4}"
METALLIC_VIDEO="${5}"
ENVMAP_0="${6}"
ENVMAP_1="${7}"
ENVMAP_2="${8}"
OUTPUT_ROOT="${9}"
TMP_ROOT="${10:-}"

CMD=(
  python scripts/prepare_drivestudio_eval_inputs.py relight
  --albedo_video "${ALBEDO_VIDEO}"
  --normal_video "${NORMAL_VIDEO}"
  --normalized_depth_video "${NORMALIZED_DEPTH_VIDEO}"
  --roughness_video "${ROUGHNESS_VIDEO}"
  --metallic_video "${METALLIC_VIDEO}"
  --envmap_0 "${ENVMAP_0}"
  --envmap_1 "${ENVMAP_1}"
  --envmap_2 "${ENVMAP_2}"
  --output_root "${OUTPUT_ROOT}"
)

if [[ -n "${TMP_ROOT}" ]]; then
  CMD+=(--tmp_root "${TMP_ROOT}")
fi

"${CMD[@]}"
