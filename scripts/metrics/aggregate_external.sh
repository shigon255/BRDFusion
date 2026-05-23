#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${ROOT:-urbanir_eval_result/self}"
LAYOUT="${LAYOUT:-auto}"

cmd=("${PYTHON_BIN}" tools/metrics/aggregate_external_metric.py --root "${ROOT}" --layout "${LAYOUT}")

if [[ -n "${SCENE_IDX:-}" ]]; then
  cmd+=(--scene-idx "${SCENE_IDX}")
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
if [[ -n "${INTRINSIC_TYPES:-}" ]]; then
  # shellcheck disable=SC2206
  types=( ${INTRINSIC_TYPES} )
  cmd+=(--intrinsic-types "${types[@]}")
fi

exec "${cmd[@]}"
