#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

required=(
  third_party/3dgrut/setup.py
  third_party/3dgrut/threedgrt_tracer
  third_party/3dgrut/threedgrut
  third_party/cosmos1-diffusion-renderer/cosmos_predict1
  third_party/cosmos1-diffusion-renderer/sdedit_forward_renderer.py
  third_party/cosmos1-diffusion-renderer/sdedit_inverse_renderer.py
  third_party/DiffusionLight-Turbo/inpaint.py
  third_party/DiffusionLight-Turbo/relighting
)

for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing vendored source path: ${path}" >&2
    exit 1
  fi
done

bad=$(find third_party -maxdepth 3 \( -name checkpoints -o -name runs -o -name outputs -o -name output -o -name drivestudio_exp -o -name asset -o -name __pycache__ \) -print)
if [[ -n "${bad}" ]]; then
  echo "Forbidden vendored artifact/cache path found:" >&2
  echo "${bad}" >&2
  exit 1
fi

echo "Third-party layout OK"
