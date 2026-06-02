#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${1:-}"

case "${TARGET}" in
  waymo)
    cat <<EOF
Waymo data requires account access and DriveStudio preprocessing.

Place raw/downloaded data under:
  ${REPO_ROOT}/data/waymo/raw/

Then run the documented preprocessing wrappers:
  RAW_ROOT=${REPO_ROOT}/data/waymo/raw \
  TARGET_ROOT=${REPO_ROOT}/data/waymo/processed \
  scripts/data/prepare_waymo.sh

BRDFusion uses the DriveStudio processed layout directly:
  ${REPO_ROOT}/data/waymo/processed/training/<scene_id>/

After preprocessing, verify with:
  DATASET=waymo DATA_ROOT=data/waymo/processed/training SCENE=<scene_id> CAM_IDS="0" CHECK_PRIORS=0 scripts/data/check_dataset_layout.sh
EOF
    ;;
  self)
    cat <<EOF
Self dataset should keep the DriveStudio-style path layout under:
  ${REPO_ROOT}/data/self/<path_name>/<scene_id>/

Expected examples:
  ${REPO_ROOT}/data/self/path1_fixed_tree_gamma_full/qwantani_moon_noon_puresky_4k/
  ${REPO_ROOT}/data/self/path1_fixed_tree_gamma_full/qwantani_moon_noon_puresky_4k_rot90/
  ${REPO_ROOT}/data/self/path2_fixed_tree_gamma_full/qwantani_moon_noon_puresky_4k/

After placement, verify with:
  DATASET=self DATA_ROOT=data/self/<path_name> SCENE=<scene_id> CAM_IDS="0" CHECK_PRIORS=0 scripts/data/check_dataset_layout.sh
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
