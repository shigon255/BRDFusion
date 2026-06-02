#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  SEGFORMER_PATH=/path/to/SegFormer scripts/data/extract_self_sky_masks.sh [path_name] <scene1> [scene2 ...]

Environment alternatives:
  SEGFORMER_PATH=/path/to/SegFormer PATH_ID=1 SCENES="scene1 scene2" scripts/data/extract_self_sky_masks.sh
  SEGFORMER_PATH=/path/to/SegFormer DATA_ROOT=data/self/path1_fixed_tree_gamma_full SCENES="scene1" scripts/data/extract_self_sky_masks.sh

Defaults:
  PATH_ID=1
  PATH_NAME=path${PATH_ID}_fixed_tree_gamma_full
  DATA_ROOT=${BRDFUSION_ROOT}/data/self/${PATH_NAME}, where BRDFUSION_ROOT is this checkout

The script writes Cityscapes SegFormer sky_masks/ and road_masks/ under each self scene.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_ENV="$(python3 "${REPO_ROOT}/tools/env_config.py" --operation sky_mask_extract --print_env 2>/dev/null || printf 'segformer')"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" && "${CONDA_DEFAULT_ENV:-}" != "${TARGET_ENV}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation sky_mask_extract --exec "${SCRIPT_PATH}" "$@"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

arg_path_name=""
if [[ $# -gt 0 && -z "${DATA_ROOT:-}" && -z "${PATH_NAME:-}" ]]; then
  if [[ "${1}" == path* || -d "${REPO_ROOT}/data/self/${1}" ]]; then
    arg_path_name="${1}"
    shift
  fi
fi

arg_scenes=()
if [[ $# -gt 0 ]]; then
  arg_scenes=("$@")
fi

PATH_ID="${PATH_ID:-1}"
PATH_NAME="${PATH_NAME:-${arg_path_name:-path${PATH_ID}_fixed_tree_gamma_full}}"
DATA_ROOT="$(resolve_path "${DATA_ROOT:-data/self/${PATH_NAME}}")"

if [[ ${#arg_scenes[@]} -gt 0 ]]; then
  scenes=("${arg_scenes[@]}")
elif [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  scenes=( ${SCENES} )
elif [[ -n "${SCENE:-}" ]]; then
  scenes=("${SCENE}")
else
  usage >&2
  exit 2
fi

: "${SEGFORMER_PATH:?Set SEGFORMER_PATH to the SegFormer checkout.}"
SEGFORMER_PATH="$(resolve_path "${SEGFORMER_PATH}")"
CHECKPOINT="${CHECKPOINT:-${SEGFORMER_PATH}/pretrained/segformer.b5.1024x1024.city.160k.pth}"
CHECKPOINT="$(resolve_path "${CHECKPOINT}")"
DEVICE="${DEVICE:-cuda:0}"

[[ -d "${DATA_ROOT}" ]] || { echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2; exit 1; }
[[ -d "${SEGFORMER_PATH}" ]] || { echo "SEGFORMER_PATH does not exist: ${SEGFORMER_PATH}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "CHECKPOINT does not exist: ${CHECKPOINT}" >&2; exit 1; }

for scene in "${scenes[@]}"; do
  [[ -d "${DATA_ROOT}/${scene}/images" ]] || {
    echo "Missing self image directory: ${DATA_ROOT}/${scene}/images" >&2
    exit 1
  }

  tmp_split="$(mktemp /tmp/brdfusion_self_split.XXXXXX.csv)"
  trap 'rm -f "${tmp_split:-}"' EXIT
  printf "scene_id\n%s\n" "${scene}" > "${tmp_split}"

  cmd=(
    python datasets/tools/extract_masks.py
    --data_root "${DATA_ROOT}"
    --split_file "${tmp_split}"
    --segformer_path "${SEGFORMER_PATH}"
    --checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    --rgb_dirname images
    --allow_string_scene_ids
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
  trap - EXIT
done
