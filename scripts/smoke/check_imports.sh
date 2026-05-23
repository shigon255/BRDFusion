#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -B - <<'PY'
from pathlib import Path

paths = [
    "tools/train.py",
    "tools/run_pipeline.py",
    "tools/eval.py",
    "tools/compute_video_metrics.py",
    "tools/compute_video_metrics_1cam.py",
    "tools/export_dynamic_assets.py",
    "tools/merge_dynamic_assets.py",
    "tools/merge_dynamic_assets_into_ckpt.py",
    "tools/export_rigid_canonical.py",
    "tools/render_pointcloud_views.py",
    "tools/export_colmap.py",
    "tools/metrics/aggregate_metric.py",
    "tools/metrics/aggregate_external_metric.py",
    "tools/metrics/aggregate_dr_metric.py",
    "tools/data/link_datasets.py",
    "tools/data/inspect_layout.py",
    "datasets/waymo/waymo_download.py",
    "scripts/setup/check_assets.py",
]

for path in paths:
    compile(Path(path).read_text(), path, "exec")
    print(f"syntax ok: {path}")
PY
