#!/usr/bin/env bash
set -euo pipefail

# Generic dynamic-asset insertion wrapper for eval-only rendering.
# Usage:
#   scripts/assets/insert_dynamic_assets.sh [target_ckpt] [asset_spec] [postfix]
#
# `asset_spec` may be:
#   - a per-object or combined asset `.pth`
#   - a manifest `.json` from export_dynamic_assets.py with selector env vars
#   - a multi-asset insertion specs `.json` (top-level list or {"items": [...]})
#
# Selector env vars for manifest JSON:
#   ASSET_ID="RigidNodes_0003"
#   SOURCE_INSTANCE_ID=3
#   OBJECT_INDEX=0
#   CLASS_NAME="RigidNodes"
#
# Placement env vars:
#   CLASSES=""  # auto-resolved when empty for single-asset paths
#   ANCHOR_TIMESTEP=0
#   ANCHOR_CAM_ID=0
#   SOURCE_ANCHOR_TIMESTEP=""
#   FORWARD_M=8.0
#   RIGHT_M=0.0
#   UP_M=0.0
#   YAW_DEG=0.0
#   PITCH_UP_DEG=0.0
#   ROLL_LEFT_DEG=0.0
#   SCALE_MODE=ego
#   MANUAL_SCALE=1.0
#   DISABLE_PLACE_SEARCH=0
#   PLACEMENT_MODE=visible
#   SEARCH_RADIUS_M=8.0
#   SEARCH_STEP_M=2.0
#   TARGET_WORLD_XYZ=""        # optional "x,y,z" absolute world target
#   TARGET_WORLD_TIMESTEP=""   # optional target timestep for TARGET_WORLD_XYZ
#   TARGET_WORLD_YAW_DEG=""    # optional absolute world-up yaw in degrees
#
# Render env vars:
#   PYTHON_BIN=python
#   CALIBRATE_ENVMAP=1
#   REMOVE_EXISTING_TARGET_NODES=1
#   REMOVE_TARGET_CLASSES="RigidNodes,SMPLNodes,DeformableNodes"
#   RENDER_INSERTED_ONLY=0
#   EVAL_START_TIMESTEP=0
#   EVAL_NUM_TIMESTEPS=1
#   EVAL_OPTS="render.render_test=false render.render_novel=null data.pixel_source.load_images_only=true"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGET_CKPT="${TARGET_CKPT:-${1:-}}"
ASSET_SPEC="${ASSET_SPEC:-${2:-}}"
POSTFIX="${POSTFIX:-${3:-}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ID="${ASSET_ID:-}"
SOURCE_INSTANCE_ID="${SOURCE_INSTANCE_ID:-}"
OBJECT_INDEX="${OBJECT_INDEX:-}"
CLASS_NAME="${CLASS_NAME:-}"
CLASSES="${CLASSES:-}"

ANCHOR_TIMESTEP="${ANCHOR_TIMESTEP:-0}"
ANCHOR_CAM_ID="${ANCHOR_CAM_ID:-0}"
SOURCE_ANCHOR_TIMESTEP="${SOURCE_ANCHOR_TIMESTEP:-}"
FORWARD_M="${FORWARD_M:-8.0}"
RIGHT_M="${RIGHT_M:-0.0}"
UP_M="${UP_M:-0.0}"
YAW_DEG="${YAW_DEG:-0.0}"
PITCH_UP_DEG="${PITCH_UP_DEG:-0.0}"
ROLL_LEFT_DEG="${ROLL_LEFT_DEG:-0.0}"
SCALE_MODE="${SCALE_MODE:-ego}"
MANUAL_SCALE="${MANUAL_SCALE:-1.0}"
DISABLE_PLACE_SEARCH="${DISABLE_PLACE_SEARCH:-0}"
PLACEMENT_MODE="${PLACEMENT_MODE:-visible}"
SEARCH_RADIUS_M="${SEARCH_RADIUS_M:-8.0}"
SEARCH_STEP_M="${SEARCH_STEP_M:-2.0}"
TARGET_WORLD_XYZ="${TARGET_WORLD_XYZ:-}"
TARGET_WORLD_TIMESTEP="${TARGET_WORLD_TIMESTEP:-}"
TARGET_WORLD_YAW_DEG="${TARGET_WORLD_YAW_DEG:-}"

CALIBRATE_ENVMAP="${CALIBRATE_ENVMAP:-1}"
REMOVE_EXISTING_TARGET_NODES="${REMOVE_EXISTING_TARGET_NODES:-1}"
REMOVE_TARGET_CLASSES="${REMOVE_TARGET_CLASSES:-RigidNodes,SMPLNodes,DeformableNodes}"
RENDER_INSERTED_ONLY="${RENDER_INSERTED_ONLY:-0}"
EVAL_START_TIMESTEP="${EVAL_START_TIMESTEP:-0}"
EVAL_NUM_TIMESTEPS="${EVAL_NUM_TIMESTEPS:-1}"
EVAL_OPTS="${EVAL_OPTS:-render.render_test=false render.render_novel=null data.pixel_source.load_images_only=true}"

DELETE_RIGID_IDS=""
DELETE_SMPL_IDS=""
DELETE_DEFORMABLE_IDS=""
MODE=""
RESOLVED_ASSET_PATH=""
RESOLVED_CLASSES=""
RESOLVED_ASSET_ID=""
RESOLVED_MANIFEST_PATH=""
SPECS_JSON_PATH=""

if [[ -z "${TARGET_CKPT}" || -z "${ASSET_SPEC}" ]]; then
  echo "[ERR] Usage: scripts/assets/insert_dynamic_assets.sh [target_ckpt] [asset_spec] [postfix]" >&2
  exit 1
fi
if [[ ! -f "${TARGET_CKPT}" ]]; then
  echo "[ERR] Target checkpoint does not exist: ${TARGET_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${ASSET_SPEC}" ]]; then
  echo "[ERR] Asset spec does not exist: ${ASSET_SPEC}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"

while IFS=$'\t' read -r key value; do
  case "${key}" in
    mode)
      MODE="${value}"
      ;;
    asset_path)
      RESOLVED_ASSET_PATH="${value}"
      ;;
    classes)
      RESOLVED_CLASSES="${value}"
      ;;
    asset_id)
      RESOLVED_ASSET_ID="${value}"
      ;;
    manifest_path)
      RESOLVED_MANIFEST_PATH="${value}"
      ;;
    specs_json)
      SPECS_JSON_PATH="${value}"
      ;;
  esac
done < <(
  "${PYTHON_BIN}" - "${ASSET_SPEC}" "${ASSET_ID}" "${SOURCE_INSTANCE_ID}" "${OBJECT_INDEX}" "${CLASS_NAME}" <<'PY'
import json
import os
import sys

import torch


def emit(key, value):
    print(f"{key}\t{value}")


asset_spec, asset_id, source_instance_id_raw, object_index_raw, class_name = sys.argv[1:6]
abs_asset_spec = os.path.abspath(asset_spec)
source_instance_id = None if source_instance_id_raw == "" else int(source_instance_id_raw)
object_index = None if object_index_raw == "" else int(object_index_raw)

if abs_asset_spec.endswith(".json"):
    with open(abs_asset_spec, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        emit("mode", "specs_json")
        emit("specs_json", abs_asset_spec)
    elif (
        isinstance(payload, dict)
        and payload.get("export_mode") == "per_object"
        and isinstance(payload.get("items"), list)
    ):
        items = list(payload.get("items", []))
        if not items:
            raise ValueError(f"Manifest contains no items: {abs_asset_spec}")

        filtered = items
        if asset_id:
            filtered = [item for item in filtered if item.get("asset_id") == asset_id]
        if class_name:
            filtered = [item for item in filtered if item.get("class_name") == class_name]
        if source_instance_id is not None:
            filtered = [item for item in filtered if int(item.get("source_instance_id", -1)) == source_instance_id]

        if object_index is not None:
            if object_index < 0 or object_index >= len(filtered):
                raise IndexError(f"OBJECT_INDEX={object_index} out of range for {len(filtered)} matching manifest item(s).")
            item = filtered[object_index]
        else:
            if len(filtered) != 1:
                raise ValueError(
                    "Manifest selector must resolve to exactly one item. "
                    f"Matches={len(filtered)}. Set ASSET_ID, SOURCE_INSTANCE_ID, CLASS_NAME, or OBJECT_INDEX."
                )
            item = filtered[0]

        asset_path = item["asset_path"]
        if not os.path.isabs(asset_path):
            asset_path = os.path.abspath(os.path.join(os.path.dirname(abs_asset_spec), asset_path))
        classes = item.get("class_name", "")
        if not classes:
            metadata = item.get("metadata", {})
            classes = metadata.get("class_name", "")
        emit("mode", "single_asset")
        emit("asset_path", asset_path)
        emit("classes", classes)
        emit("asset_id", item.get("asset_id", os.path.splitext(os.path.basename(asset_path))[0]))
        emit("manifest_path", abs_asset_spec)
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        emit("mode", "specs_json")
        emit("specs_json", abs_asset_spec)
    else:
        raise ValueError(f"Unsupported JSON format for asset spec: {abs_asset_spec}")
else:
    package = torch.load(abs_asset_spec, map_location="cpu")
    package_classes = package.get("classes", {})
    if not package_classes:
        raise KeyError(f"Asset package missing classes: {abs_asset_spec}")
    emit("mode", "single_asset")
    emit("asset_path", abs_asset_spec)
    emit("classes", ",".join(package_classes.keys()))
    emit("asset_id", os.path.splitext(os.path.basename(abs_asset_spec))[0])
    emit("manifest_path", "")
PY
)

if [[ -z "${MODE}" ]]; then
  echo "[ERR] Failed to resolve asset spec: ${ASSET_SPEC}" >&2
  exit 1
fi

if [[ -z "${POSTFIX}" ]]; then
  if [[ "${MODE}" == "specs_json" ]]; then
    spec_basename="$(basename "${SPECS_JSON_PATH}")"
    POSTFIX="_eval_insert_${spec_basename%.json}"
  else
    POSTFIX="_eval_insert_${RESOLVED_ASSET_ID}"
  fi
fi

if [[ "${REMOVE_EXISTING_TARGET_NODES}" == "1" ]]; then
  while IFS=$'\t' read -r class_name ids_csv; do
    case "${class_name}" in
      RigidNodes)
        DELETE_RIGID_IDS="${ids_csv}"
        ;;
      SMPLNodes)
        DELETE_SMPL_IDS="${ids_csv}"
        ;;
      DeformableNodes)
        DELETE_DEFORMABLE_IDS="${ids_csv}"
        ;;
    esac
  done < <(
    "${PYTHON_BIN}" - "${TARGET_CKPT}" "${REMOVE_TARGET_CLASSES}" <<'PY'
import sys
import torch

target_ckpt = sys.argv[1]
classes = [x.strip() for x in sys.argv[2].split(',') if x.strip()]
ckpt = torch.load(target_ckpt, map_location='cpu')
models = ckpt.get('models', {})

def get_num_instances(state):
    if 'instances_fv' in state:
        return int(state['instances_fv'].shape[1])
    if 'instances_size' in state:
        return int(state['instances_size'].shape[0])
    for key in ('points_ids', 'point_ids'):
        if key in state:
            pids = state[key].reshape(-1).long()
            return int(pids.max().item()) + 1 if pids.numel() > 0 else 0
    return 0

for class_name in classes:
    state = models.get(class_name, None)
    if state is None:
        print(f"{class_name}\t")
        continue
    n = get_num_instances(state)
    ids = ','.join(str(i) for i in range(n))
    print(f"{class_name}\t{ids}")
PY
  )
fi

CMD=(
  "${PYTHON_BIN}" -u -m tools.eval
  --resume_from "${TARGET_CKPT}"
  --postfix "${POSTFIX}"
  --eval_start_timestep "${EVAL_START_TIMESTEP}"
  --eval_num_timesteps "${EVAL_NUM_TIMESTEPS}"
)

if [[ "${MODE}" == "specs_json" ]]; then
  CMD+=(--insert_dynamic_specs_json "${SPECS_JSON_PATH}")
else
  if [[ -z "${CLASSES}" ]]; then
    CLASSES="${RESOLVED_CLASSES}"
  fi
  if [[ -z "${CLASSES}" ]]; then
    echo "[ERR] Failed to resolve classes for asset: ${RESOLVED_ASSET_PATH}" >&2
    exit 1
  fi

  CMD+=(
    --insert_dynamic_asset "${RESOLVED_ASSET_PATH}"
    --insert_dynamic_classes "${CLASSES}"
    --insert_anchor_timestep "${ANCHOR_TIMESTEP}"
    --insert_cam_id "${ANCHOR_CAM_ID}"
    --insert_forward_m "${FORWARD_M}"
    --insert_right_m "${RIGHT_M}"
    --insert_up_m "${UP_M}"
    --insert_yaw_deg "${YAW_DEG}"
    --insert_pitch_up_deg "${PITCH_UP_DEG}"
    --insert_roll_left_deg "${ROLL_LEFT_DEG}"
    --insert_scale_mode "${SCALE_MODE}"
    --insert_manual_scale "${MANUAL_SCALE}"
    --insert_dynamic_placement_mode "${PLACEMENT_MODE}"
    --insert_dynamic_search_radius_m "${SEARCH_RADIUS_M}"
    --insert_dynamic_search_step_m "${SEARCH_STEP_M}"
  )

  if [[ -n "${SOURCE_ANCHOR_TIMESTEP}" ]]; then
    CMD+=(--insert_dynamic_source_anchor_timestep "${SOURCE_ANCHOR_TIMESTEP}")
  fi
  if [[ "${DISABLE_PLACE_SEARCH}" == "1" ]]; then
    CMD+=(--insert_dynamic_disable_place_search)
  fi
fi
if [[ -n "${TARGET_WORLD_XYZ}" ]]; then
  CMD+=(--insert_target_world_xyz "${TARGET_WORLD_XYZ}")
fi
if [[ -n "${TARGET_WORLD_TIMESTEP}" ]]; then
  CMD+=(--insert_target_world_timestep "${TARGET_WORLD_TIMESTEP}")
fi
if [[ -n "${TARGET_WORLD_YAW_DEG}" ]]; then
  CMD+=(--insert_target_world_yaw_deg "${TARGET_WORLD_YAW_DEG}")
fi

if [[ "${RENDER_INSERTED_ONLY}" == "1" ]]; then
  CMD+=(--render_inserted_only)
fi
if [[ -n "${DELETE_RIGID_IDS}" ]]; then
  CMD+=(--delete_rigid_ids "${DELETE_RIGID_IDS}")
fi
if [[ -n "${DELETE_SMPL_IDS}" ]]; then
  CMD+=(--delete_smpl_ids "${DELETE_SMPL_IDS}")
fi
if [[ -n "${DELETE_DEFORMABLE_IDS}" ]]; then
  CMD+=(--delete_deformable_ids "${DELETE_DEFORMABLE_IDS}")
fi
if [[ "${CALIBRATE_ENVMAP}" == "1" ]]; then
  CMD+=(--calibrate_envmap)
fi
if [[ -n "${EVAL_OPTS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_OPTS=( ${EVAL_OPTS} )
  CMD+=("${EXTRA_OPTS[@]}")
fi

echo "[INFO] Target checkpoint: ${TARGET_CKPT}"
echo "[INFO] Asset spec: ${ASSET_SPEC}"
echo "[INFO] Eval postfix: ${POSTFIX}"
echo "[INFO] Insert mode: ${MODE}"
echo "[INFO] Render inserted only: ${RENDER_INSERTED_ONLY}"
echo "[INFO] Remove target classes: ${REMOVE_TARGET_CLASSES}"
if [[ "${MODE}" == "specs_json" ]]; then
  echo "[INFO] Specs JSON: ${SPECS_JSON_PATH}"
else
  [[ -n "${RESOLVED_MANIFEST_PATH}" ]] && echo "[INFO] Manifest path: ${RESOLVED_MANIFEST_PATH}"
  echo "[INFO] Resolved asset: ${RESOLVED_ASSET_PATH}"
  echo "[INFO] Asset id: ${RESOLVED_ASSET_ID}"
  echo "[INFO] Classes: ${CLASSES}"
  echo "[INFO] Placement mode: ${PLACEMENT_MODE}"
  if [[ -n "${ASSET_ID}" ]]; then
    echo "[INFO] Selector ASSET_ID=${ASSET_ID}"
  fi
  if [[ -n "${SOURCE_INSTANCE_ID}" ]]; then
    echo "[INFO] Selector SOURCE_INSTANCE_ID=${SOURCE_INSTANCE_ID}"
  fi
  if [[ -n "${CLASS_NAME}" ]]; then
    echo "[INFO] Selector CLASS_NAME=${CLASS_NAME}"
  fi
  if [[ -n "${OBJECT_INDEX}" ]]; then
    echo "[INFO] Selector OBJECT_INDEX=${OBJECT_INDEX}"
  fi
fi
if [[ "${REMOVE_EXISTING_TARGET_NODES}" == "1" ]]; then
  echo "[INFO] Removing existing target nodes for rendering only."
fi

"${CMD[@]}"

echo "[OK] Dynamic insertion render complete."
