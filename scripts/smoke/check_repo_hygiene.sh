#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

bad=$(find . \( -name __pycache__ -o -name "*.pyc" \) -print)
if [[ -n "${bad}" ]]; then
  echo "${bad}" >&2
  exit 1
fi

for path in outputs work_dirs wandb checkpoints dr_eval_output alltrain_dr_eval_output rgbx_eval_output unirelight_eval_output urbanir_eval_result r3dg_results irgs_results ablation_results gen3c_vid night_gt_vids blurry_test pcd_render; do
  if [[ -e "${path}" ]]; then
    echo "Generated/local artifact path is present: ${path}" >&2
    exit 1
  fi
done

echo "Repo hygiene OK"
