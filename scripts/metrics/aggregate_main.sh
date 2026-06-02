#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation metric_compute --exec "${SCRIPT_PATH}" "$@"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${ROOT:-outputs}"

cmd=("${PYTHON_BIN}" tools/metrics/aggregate_metric.py --root "${ROOT}")

if [[ -n "${SCENE_IDX:-}" ]]; then
  cmd+=(--scene-idx "${SCENE_IDX}")
fi
if [[ -n "${WAYMO_STATIC_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ids=( ${WAYMO_STATIC_IDS} )
  cmd+=(--waymo-static-ids "${ids[@]}")
fi
if [[ -n "${WAYMO_DYNAMIC_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ids=( ${WAYMO_DYNAMIC_IDS} )
  cmd+=(--waymo-dynamic-ids "${ids[@]}")
fi
if [[ -n "${PATH_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ids=( ${PATH_IDS} )
  cmd+=(--path-ids "${ids[@]}")
fi
if [[ -n "${RELIGHT_SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  scenes=( ${RELIGHT_SCENES} )
  cmd+=(--relight-scenes "${scenes[@]}")
fi
if [[ "${MULTISEED:-0}" == "1" ]]; then
  cmd+=(--multiseed)
fi
if [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  seeds=( ${SEEDS} )
  cmd+=(--seeds "${seeds[@]}")
fi
if [[ "${INCLUDE_MAIN:-0}" == "1" ]]; then
  cmd+=(--include-main)
fi

exec "${cmd[@]}"
