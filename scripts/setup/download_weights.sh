#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${1:-all}"
COSMOS_ENV="${COSMOS_ENV:-cosmos-predict1}"
COSMOS_ROOT="${COSMOS_ROOT:-${REPO_ROOT}/third_party/cosmos1-diffusion-renderer}"
COSMOS_CHECKPOINT_DIR="${COSMOS_CHECKPOINT_DIR:-${REPO_ROOT}/assets/checkpoints/cosmos}"
DIFFUSIONLIGHT_SOURCE="${DIFFUSIONLIGHT_SOURCE:-}"
DIFFUSIONLIGHT_TARGET="${DIFFUSIONLIGHT_TARGET:-${REPO_ROOT}/assets/checkpoints/diffusionlight/models}"
BRDFUSION_SOURCE="${BRDFUSION_SOURCE:-}"
BRDFUSION_TARGET="${BRDFUSION_TARGET:-${REPO_ROOT}/assets/checkpoints/brdfusion}"

download_cosmos() {
  mkdir -p "${COSMOS_CHECKPOINT_DIR}"
  if [[ ! -f "${COSMOS_ROOT}/scripts/download_diffusion_renderer_checkpoints.py" ]]; then
    echo "Missing vendored Cosmos download script: ${COSMOS_ROOT}/scripts/download_diffusion_renderer_checkpoints.py" >&2
    exit 1
  fi
  (
    cd "${COSMOS_ROOT}"
    conda run -n "${COSMOS_ENV}" python scripts/download_diffusion_renderer_checkpoints.py \
      --checkpoint_dir "${COSMOS_CHECKPOINT_DIR}"
  )
}

copy_diffusionlight() {
  mkdir -p "${DIFFUSIONLIGHT_TARGET}"
  if [[ -z "${DIFFUSIONLIGHT_SOURCE}" ]]; then
    cat >&2 <<EOF
Set DIFFUSIONLIGHT_SOURCE to a prepared DiffusionLight-Turbo models directory.
Expected local target:
  ${DIFFUSIONLIGHT_TARGET}
Required subdirectories:
  ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500
  rev3/Flickr2K/Flickr2kPlus_extended/checkpoint-230000
EOF
    exit 1
  fi
  rsync -a "${DIFFUSIONLIGHT_SOURCE%/}/" "${DIFFUSIONLIGHT_TARGET}/"
}

copy_brdfusion() {
  mkdir -p "${BRDFUSION_TARGET}"
  if [[ -z "${BRDFUSION_SOURCE}" ]]; then
    echo "No BRDFUSION_SOURCE set; skipping optional BRDFusion checkpoints." >&2
    return 0
  fi
  rsync -a "${BRDFUSION_SOURCE%/}/" "${BRDFUSION_TARGET}/"
}

case "${TARGET}" in
  cosmos) download_cosmos ;;
  diffusionlight) copy_diffusionlight ;;
  brdfusion) copy_brdfusion ;;
  all)
    download_cosmos
    copy_diffusionlight
    copy_brdfusion
    ;;
  *)
    echo "Usage: $0 {cosmos|diffusionlight|brdfusion|all}" >&2
    exit 2
    ;;
esac
