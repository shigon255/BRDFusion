#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

: "${CKPT:?Set CKPT to the reconstruction checkpoint path.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the training output root.}"
: "${RUN_NAME:?Set RUN_NAME to the PBR run name.}"

CONFIG="${CONFIG:-configs/omnire.yaml}"
PROJECT="${PROJECT:-brdfusion}"
DATASET="${DATASET:-self/3cams}"

cmd=(
  python tools/train.py
  --config_file "${CONFIG}"
  --output_root "${OUTPUT_ROOT}"
  --project "${PROJECT}"
  --run_name "${RUN_NAME}"
  --resume_from "${CKPT}"
)

CONFIG_OVERLAYS="${CONFIG_OVERLAYS:-configs/method/pbr.yaml}"
# shellcheck disable=SC2206
overlays=( ${CONFIG_OVERLAYS} )
for overlay in "${overlays[@]}"; do
  cmd+=(--config_overlay "${overlay}")
done

if [[ -n "${NEW_ENVMAP:-}" ]]; then
  cmd+=(--new_envmap_path "${NEW_ENVMAP}")
fi
if [[ "${NO_CALIBRATE_ENVMAP:-0}" == "1" ]]; then
  cmd+=(--no_calibrate_envmap)
fi
if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  cmd+=(--enable_wandb)
fi

cmd+=(
  dataset="${DATASET}"
  trainer.tracer.use_pbr=true
)

if [[ -n "${NUM_ITERS:-}" ]]; then
  cmd+=(trainer.optim.num_iters="${NUM_ITERS}")
fi
if [[ -n "${DATA_ROOT:-}" ]]; then
  cmd+=(data.data_root="${DATA_ROOT}")
fi
if [[ -n "${SCENE_IDX:-}" ]]; then
  cmd+=(data.scene_idx="${SCENE_IDX}")
fi
if [[ -n "${START:-}" ]]; then
  cmd+=(data.start_timestep="${START}")
fi
if [[ -n "${END:-}" ]]; then
  cmd+=(data.end_timestep="${END}")
fi
if [[ -n "${ENVMAP_PRIOR:-}" ]]; then
  cmd+=(model.Sky.params.prior_path="${ENVMAP_PRIOR}")
fi

if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

exec "${cmd[@]}"
