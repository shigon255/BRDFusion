#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
