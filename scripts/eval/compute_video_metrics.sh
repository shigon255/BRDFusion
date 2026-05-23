#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CONFIG:?Set CONFIG to the config file path.}"
: "${START:?Set START to the first timestep.}"
: "${END:?Set END to the inclusive last timestep.}"
: "${TEST_STRIDE:?Set TEST_STRIDE to the held-out timestep stride.}"

cmd=(
  python tools/compute_video_metrics.py
  --config_file "${CONFIG}"
  --start_timestep "${START}"
  --end_timestep "${END}"
  --test_image_stride "${TEST_STRIDE}"
)

if [[ -n "${CONFIG_OVERLAYS:-}" ]]; then
  # shellcheck disable=SC2206
  overlays=( ${CONFIG_OVERLAYS} )
  for overlay in "${overlays[@]}"; do
    cmd+=(--config_overlay "${overlay}")
  done
fi
if [[ -n "${DATASET:-}" ]]; then
  cmd+=(--dataset "${DATASET}")
fi
if [[ -n "${DATASET_SOURCE:-}" ]]; then
  cmd+=(--dataset_source "${DATASET_SOURCE}")
fi
if [[ -n "${VIDEO_ROOT:-}" ]]; then
  cmd+=(--video_root "${VIDEO_ROOT}")
fi
if [[ -n "${VIDEO_NAME:-}" ]]; then
  cmd+=(--video_name "${VIDEO_NAME}")
fi
if [[ -n "${IMAGE_JSON:-}" ]]; then
  cmd+=(--image_output_json "${IMAGE_JSON}")
fi
if [[ -n "${RELIGHT_VIDEO_ROOT:-}" ]]; then
  cmd+=(--relight_video_root "${RELIGHT_VIDEO_ROOT}")
fi
if [[ -n "${RELIGHT_VIDEO_NAME:-}" ]]; then
  cmd+=(--relight_video_name "${RELIGHT_VIDEO_NAME}")
fi
if [[ -n "${RELIGHT_JSON:-}" ]]; then
  cmd+=(--relight_output_json "${RELIGHT_JSON}")
fi
if [[ -n "${INTRINSIC_VIDEO_ROOT:-}" ]]; then
  cmd+=(--intrinsic_video_root "${INTRINSIC_VIDEO_ROOT}")
fi
if [[ -n "${INTRINSIC_JSON:-}" ]]; then
  cmd+=(--intrinsic_output_json "${INTRINSIC_JSON}")
fi
if [[ "${COLOR_CORRECTION:-0}" == "1" ]]; then
  cmd+=(--color_correction)
fi
if [[ "${PASTE_GT_SKY_FOR_IMAGE:-0}" == "1" ]]; then
  cmd+=(--paste_gt_sky_for_image)
fi
if [[ "${GROUND_ONLY_RELIGHT:-0}" == "1" ]]; then
  cmd+=(--ground_only_relight)
fi
if [[ "${PASTE_GT_SKY_FOR_RELIGHT:-0}" == "1" ]]; then
  cmd+=(--paste_gt_sky_for_relight)
fi
if [[ -n "${CAM_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  cams=( ${CAM_IDS} )
  cmd+=(--cam_ids "${cams[@]}")
fi

if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
