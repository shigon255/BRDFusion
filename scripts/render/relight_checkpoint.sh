#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CKPT:?Set CKPT to the checkpoint path.}"
: "${NEW_ENVMAP:?Set NEW_ENVMAP to the replacement envmap path.}"

POSTFIX="${POSTFIX:-${RELIGHT_POSTFIX:-_relight}}"
NEW_ENVMAP_RES="${NEW_ENVMAP_RES:-1024}"

cmd=(
  python tools/eval.py
  --resume_from "${CKPT}"
  --postfix "${POSTFIX}"
  --new_envmap_path "${NEW_ENVMAP}"
  --new_envmap_path_res "${NEW_ENVMAP_RES}"
)

if [[ -n "${CONFIG_OVERLAYS:-}" ]]; then
  # shellcheck disable=SC2206
  overlays=( ${CONFIG_OVERLAYS} )
  for overlay in "${overlays[@]}"; do
    cmd+=(--config_overlay "${overlay}")
  done
fi
if [[ "${NEW_ENVMAP_PBR_ONLY:-0}" == "1" ]]; then
  cmd+=(--new_envmap_pbr_only)
fi
if [[ -n "${ENVMAP_ROTATION:-}" ]]; then
  cmd+=(--envmap_rotation "${ENVMAP_ROTATION}")
fi
if [[ "${ENVMAP_ROTATION_VERTICAL:-0}" == "1" ]]; then
  cmd+=(--envmap_rotation_vertical)
fi
if [[ "${CALIBRATE_ENVMAP:-0}" == "1" ]]; then
  cmd+=(--calibrate_envmap)
fi
if [[ -n "${DATASET_SOURCE:-}" ]]; then
  cmd+=(--dataset_source "${DATASET_SOURCE}")
fi
if [[ -n "${CALIBRATE_ENVMAP_SOURCE:-}" ]]; then
  cmd+=(--calibrate_envmap_source "${CALIBRATE_ENVMAP_SOURCE}")
fi
if [[ -n "${START:-}" ]]; then
  cmd+=(--eval_start_timestep "${START}")
fi
if [[ -n "${NUM_FRAMES:-}" ]]; then
  cmd+=(--eval_num_timesteps "${NUM_FRAMES}")
fi
if [[ "${NO_PBR:-0}" == "1" ]]; then
  cmd+=(--no_pbr)
fi

if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
