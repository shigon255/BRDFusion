#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CKPT:?Set CKPT to the checkpoint path.}"

POSTFIX="${POSTFIX:-_eval}"

cmd=(python tools/eval.py --resume_from "${CKPT}" --postfix "${POSTFIX}")

if [[ -n "${CONFIG_OVERLAYS:-}" ]]; then
  # shellcheck disable=SC2206
  overlays=( ${CONFIG_OVERLAYS} )
  for overlay in "${overlays[@]}"; do
    cmd+=(--config_overlay "${overlay}")
  done
fi
if [[ -n "${RENDER_VIDEO_POSTFIX:-}" ]]; then
  cmd+=(--render_video_postfix "${RENDER_VIDEO_POSTFIX}")
fi
if [[ -n "${START:-}" ]]; then
  cmd+=(--eval_start_timestep "${START}")
fi
if [[ -n "${NUM_FRAMES:-}" ]]; then
  cmd+=(--eval_num_timesteps "${NUM_FRAMES}")
fi
if [[ -n "${DATASET_SOURCE:-}" ]]; then
  cmd+=(--dataset_source "${DATASET_SOURCE}")
fi
if [[ "${CALIBRATE_ENVMAP:-0}" == "1" ]]; then
  cmd+=(--calibrate_envmap)
fi
if [[ "${NO_PBR:-0}" == "1" ]]; then
  cmd+=(--no_pbr)
fi
if [[ "${LOG_METRICS:-0}" == "1" ]]; then
  cmd+=(--log_metrics)
fi

if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
