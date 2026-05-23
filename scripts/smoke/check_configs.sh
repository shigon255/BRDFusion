#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY'
from omegaconf import OmegaConf

paths = [
    "configs/omnire.yaml",
    "configs/omnire_extended_cam.yaml",
    "configs/method/pbr.yaml",
    "configs/method/relighting.yaml",
    "configs/method/local_lights.yaml",
    "configs/stage/stage1_raster.yaml",
    "configs/stage/stage2_light.yaml",
    "configs/stage/stage3_finetune.yaml",
    "configs/data/brdfusion_layout.yaml",
    "configs/datasets/self/brdfusion_1cam.yaml",
    "configs/datasets/self/brdfusion_3cams.yaml",
    "configs/datasets/waymo/brdfusion_3cams.yaml",
    "configs/pipeline/self_repro.yaml",
]

for path in paths:
    OmegaConf.load(path)
    print(path)
PY
