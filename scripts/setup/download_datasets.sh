#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${1:-}"

case "${TARGET}" in
  waymo)
    cat <<EOF
Waymo data requires account access and preprocessing.

Place raw/downloaded data under:
  ${REPO_ROOT}/data/waymo/

Then run the documented preprocessing wrappers:
  RAW_ROOT=${REPO_ROOT}/data/waymo/raw \\
  TARGET_ROOT=${REPO_ROOT}/data/waymo/processed \\
  scripts/data/prepare_waymo.sh

Expose processed scenes to BRDFusion configs with:
  BRDFUSION_WAYMO_SOURCE_ROOT=${REPO_ROOT}/data/waymo/processed/training \\
  python tools/data/link_datasets.py --dataset waymo --require_source
EOF
    ;;
  self)
    cat <<EOF
Self dataset can be placed under:
  ${REPO_ROOT}/data/sources/self/

Expected examples:
  ${REPO_ROOT}/data/sources/self/path1_fixed_tree_gamma_full/<scene>/
  ${REPO_ROOT}/data/sources/self/path1_fixed_tree_gamma_full_exr_intrinsic/<scene>/
  ${REPO_ROOT}/data/sources/self/path1-3-calib_fixed_tree_gamma_full/<scene>/

Expose scenes to BRDFusion configs with:
  BRDFUSION_SELF_SOURCE_ROOT=${REPO_ROOT}/data/sources/self \\
  python tools/data/link_datasets.py --dataset self --require_source

After placement, verify with:
  DATASET=self DATA_ROOT=data/brdfusion/self/scenes SCENE=<canonical_scene_id> CAM_IDS="0 1 2" scripts/data/check_dataset_layout.sh
EOF
    ;;
  all)
    "$0" waymo
    echo
    "$0" self
    ;;
  *)
    echo "Usage: $0 {waymo|self|all}" >&2
    exit 2
    ;;
esac
