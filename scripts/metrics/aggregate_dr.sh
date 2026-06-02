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

SELF_ROOT="${SELF_ROOT:-dr_eval_output}"
SHIFTED_ROOT="${SHIFTED_ROOT:-dr_eval_output_shifted_3}"
UNIRELIGHT_ROOT="${UNIRELIGHT_ROOT:-unirelight_eval_output}"
UNIRELIGHT_SHIFTED_ROOT="${UNIRELIGHT_SHIFTED_ROOT:-unirelight_eval_output_shifted_3}"

cmd=(
  "${PYTHON_BIN}" tools/metrics/aggregate_dr_metric.py
  --self-root "${SELF_ROOT}"
  --shifted-root "${SHIFTED_ROOT}"
  --unirelight-root "${UNIRELIGHT_ROOT}"
  --unirelight-shifted-root "${UNIRELIGHT_SHIFTED_ROOT}"
)

for pair in \
  "SCENE_IDX:--scene-idx" \
  "UNIRELIGHT_FOLDER:--unirelight-folder"; do
  env_name="${pair%%:*}"
  flag="${pair##*:}"
  value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    cmd+=("${flag}" "${value}")
  fi
done

if [[ -n "${PATH_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ids=( ${PATH_IDS} )
  cmd+=(--path-ids "${ids[@]}")
fi
if [[ -n "${WAYMO_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  ids=( ${WAYMO_IDS} )
  cmd+=(--waymo-ids "${ids[@]}")
fi
if [[ -n "${SELF_RELIGHT_FOLDERS:-}" ]]; then
  # shellcheck disable=SC2206
  folders=( ${SELF_RELIGHT_FOLDERS} )
  cmd+=(--self-relight-folders "${folders[@]}")
fi
if [[ -n "${SHIFTED_RELIGHT_FOLDERS:-}" ]]; then
  # shellcheck disable=SC2206
  folders=( ${SHIFTED_RELIGHT_FOLDERS} )
  cmd+=(--shifted-relight-folders "${folders[@]}")
fi
if [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  seeds=( ${SEEDS} )
  cmd+=(--seeds "${seeds[@]}")
fi
if [[ "${MULTISEED:-0}" == "1" ]]; then
  cmd+=(--multiseed)
fi
if [[ "${INCLUDE_UNIRELIGHT:-0}" == "1" ]]; then
  cmd+=(--include-unirelight)
fi

exec "${cmd[@]}"
