#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

bash -n scripts/assets/*.sh
bash -n scripts/data/*.sh
bash -n scripts/eval/*.sh
bash -n scripts/export/*.sh
bash -n scripts/metrics/*.sh
bash -n scripts/priors/*.sh
bash -n scripts/render/*.sh
bash -n scripts/setup/*.sh
bash -n scripts/smoke/*.sh
bash -n scripts/train/*.sh
