#!/usr/bin/env bash
set -euo pipefail

# Unified eval-time application renderer.
# Configure one or more effects with environment variables and run:
#   CKPT=/path/to/checkpoint_final.pth scripts/applications/render.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"
if [[ "${BRDFUSION_SKIP_CONDA_RUN:-0}" != "1" && -z "${BRDFUSION_IN_CONDA_RUN:-}" && -z "${PYTHON_BIN+x}" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  exec python3 "${REPO_ROOT}/tools/env_config.py" --operation render --exec "${SCRIPT_PATH}" "$@"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT="${CKPT:-${TARGET_CKPT:-${1:-}}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START="${START:-${EVAL_START_TIMESTEP:-}}"
NUM_FRAMES="${NUM_FRAMES:-${EVAL_NUM_TIMESTEPS:-}}"
CAM_IDS="${APP_CAM_IDS:-${EVAL_CAM_IDS:-${CAM_IDS:-}}}"

if [[ -z "${CKPT}" ]]; then
  echo "[ERR] Usage: CKPT=/path/to/checkpoint_final.pth scripts/applications/render.sh" >&2
  exit 1
fi
[[ -f "${CKPT}" ]] || { echo "[ERR] CKPT does not exist: ${CKPT}" >&2; exit 1; }

sanitize_component() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "${value}"
}

safe_tag() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  printf '%s' "${value:-item}"
}

append_tag() {
  TAGS+=("$1")
}

append_overlay() {
  CONFIG_OVERLAY_ARGS+=(--config_overlay "$1")
}

append_csv() {
  local current="$1"
  local extra="$2"
  if [[ -z "${extra}" ]]; then
    printf '%s' "${current}"
  elif [[ -z "${current}" ]]; then
    printf '%s' "${extra}"
  else
    printf '%s,%s' "${current}" "${extra}"
  fi
}

TAGS=()
CONFIG_OVERLAY_ARGS=()
EXTRA_EVAL_ARGS=()
EXTRA_CONFIG_OPTS=()
DELETE_RIGID_IDS="${DELETE_RIGID_IDS:-}"
DELETE_SMPL_IDS="${DELETE_SMPL_IDS:-}"
DELETE_DEFORMABLE_IDS="${DELETE_DEFORMABLE_IDS:-}"

if [[ -n "${CONFIG_OVERLAYS:-}" ]]; then
  # shellcheck disable=SC2206
  overlays=( ${CONFIG_OVERLAYS} )
  for overlay in "${overlays[@]}"; do
    append_overlay "${overlay}"
  done
fi

CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
RUN_ROOT="$(dirname "${CKPT_DIR}")"

# Time range / camera path. These settings select render poses only; they must
# not rewrite the dataset window because checkpoint shapes depend on it.
if [[ -n "${START:-}" || -n "${END:-}" ]]; then
  : "${START:?Set START when END is provided.}"
  if [[ -n "${END:-}" ]]; then
    if (( END < START )); then
      echo "[ERR] END must be >= START. Got START=${START}, END=${END}" >&2
      exit 1
    fi
    NUM_FRAMES="$((END - START + 1))"
    append_tag "start${START}_end${END}"
  elif [[ -n "${NUM_FRAMES:-}" ]]; then
    if (( NUM_FRAMES < 1 )); then
      echo "[ERR] NUM_FRAMES must be >= 1. Got NUM_FRAMES=${NUM_FRAMES}" >&2
      exit 1
    fi
    END="$((START + NUM_FRAMES - 1))"
    append_tag "start${START}_num${NUM_FRAMES}"
  else
    append_tag "start${START}"
  fi
elif [[ -n "${NUM_FRAMES:-}" ]]; then
  if (( NUM_FRAMES < 1 )); then
    echo "[ERR] NUM_FRAMES must be >= 1. Got NUM_FRAMES=${NUM_FRAMES}" >&2
    exit 1
  fi
  append_tag "num${NUM_FRAMES}"
fi
if [[ -n "${CAM_IDS}" ]]; then
  append_tag "cams$(safe_tag "${CAM_IDS}")"
fi

CAMERA_INTERP_STEPS="${CAMERA_INTERP_STEPS:-0}"
TIMESTEP="${TIMESTEP:-${SPIRAL_TIMESTEP:-}}"
CAM_ID="${CAM_ID:-${SPIRAL_CAM_ID:-0}}"
SPIRAL_FRAMES="${SPIRAL_FRAMES:-120}"
SPIRAL_LOOPS="${SPIRAL_LOOPS:-1}"
SPIRAL_RADIUS_M="${SPIRAL_RADIUS_M:-1.0}"
SPIRAL_VERTICAL_AMPLITUDE_M="${SPIRAL_VERTICAL_AMPLITUDE_M:-0.3}"
SPIRAL_TARGET_DISTANCE_M="${SPIRAL_TARGET_DISTANCE_M:-10.0}"

if [[ -n "${TIMESTEP}" && "${CAMERA_INTERP_STEPS}" != "0" ]]; then
  echo "[ERR] Spiral mode cannot be combined with CAMERA_INTERP_STEPS." >&2
  exit 1
fi
if (( CAMERA_INTERP_STEPS < 0 )); then
  echo "[ERR] CAMERA_INTERP_STEPS must be >= 0." >&2
  exit 1
fi
if (( CAMERA_INTERP_STEPS > 0 )); then
  append_tag "interp${CAMERA_INTERP_STEPS}"
  EXTRA_EVAL_ARGS+=(--camera_interp_steps "${CAMERA_INTERP_STEPS}")
fi
if [[ -n "${TIMESTEP}" ]]; then
  if (( TIMESTEP < 0 )); then
    echo "[ERR] TIMESTEP must be >= 0." >&2
    exit 1
  fi
  if (( CAM_ID < 0 )); then
    echo "[ERR] CAM_ID must be >= 0." >&2
    exit 1
  fi
  if (( SPIRAL_FRAMES < 2 )); then
    echo "[ERR] SPIRAL_FRAMES must be >= 2." >&2
    exit 1
  fi
  loops_tag="$(sanitize_component "${SPIRAL_LOOPS}")"
  radius_tag="$(sanitize_component "${SPIRAL_RADIUS_M}")"
  vertical_tag="$(sanitize_component "${SPIRAL_VERTICAL_AMPLITUDE_M}")"
  target_tag="$(sanitize_component "${SPIRAL_TARGET_DISTANCE_M}")"
  append_tag "spiral_timestep${TIMESTEP}_cam${CAM_ID}_frames${SPIRAL_FRAMES}_loops${loops_tag}_r${radius_tag}_z${vertical_tag}_target${target_tag}"
  EXTRA_EVAL_ARGS+=(
    --spiral_timestep "${TIMESTEP}"
    --spiral_cam_id "${CAM_ID}"
    --spiral_frames "${SPIRAL_FRAMES}"
    --spiral_loops "${SPIRAL_LOOPS}"
    --spiral_radius_m "${SPIRAL_RADIUS_M}"
    --spiral_vertical_amplitude_m "${SPIRAL_VERTICAL_AMPLITUDE_M}"
    --spiral_target_distance_m "${SPIRAL_TARGET_DISTANCE_M}"
  )
  if [[ -n "${SPIRAL_FPS:-}" ]]; then
    EXTRA_EVAL_ARGS+=(--spiral_fps "${SPIRAL_FPS}")
  fi
fi

# Relighting.
if [[ -n "${NEW_ENVMAP:-}" ]]; then
  [[ -f "${NEW_ENVMAP}" ]] || { echo "[ERR] NEW_ENVMAP does not exist: ${NEW_ENVMAP}" >&2; exit 1; }
  relight_stem="$(basename "${NEW_ENVMAP}")"
  relight_stem="${relight_stem%.*}"
  RELIGHT_ENVMAP_TAG="${RELIGHT_ENVMAP_TAG:-$(safe_tag "${relight_stem}")}"
  append_tag "relight_${RELIGHT_ENVMAP_TAG}"
  EXTRA_EVAL_ARGS+=(--new_envmap_path "${NEW_ENVMAP}" --new_envmap_path_res "${NEW_ENVMAP_RES:-1024}")
  if [[ "${NEW_ENVMAP_PBR_ONLY:-0}" == "1" ]]; then
    append_tag "pbr_only"
    EXTRA_EVAL_ARGS+=(--new_envmap_pbr_only)
  fi
  if [[ -n "${ENVMAP_ROTATION:-}" ]]; then
    append_tag "rotated"
    EXTRA_EVAL_ARGS+=(--envmap_rotation "${ENVMAP_ROTATION}")
  fi
  if [[ "${ENVMAP_ROTATION_VERTICAL:-0}" == "1" ]]; then
    append_tag "vertical"
    EXTRA_EVAL_ARGS+=(--envmap_rotation_vertical)
  fi
fi

# Local lights.
LOCAL_LIGHT_OVERLAY="${LOCAL_LIGHT_OVERLAY:-configs/method/local_lights.yaml}"
LOCAL_LIGHTS_ENABLED=0
if [[ "${LOCAL_LIGHTS:-0}" == "1" || -n "${LOCAL_LIGHT_CONFIG:-}" || -n "${POINT_LIGHTS:-}" || -n "${POINT_LIGHTS_FILE:-}" ]]; then
  LOCAL_LIGHTS_ENABLED=1
fi
if [[ "${LOCAL_LIGHTS_ENABLED}" == "1" ]]; then
  [[ -f "${LOCAL_LIGHT_OVERLAY}" ]] || { echo "[ERR] Local-light overlay missing: ${LOCAL_LIGHT_OVERLAY}" >&2; exit 1; }
  if [[ -n "${LOCAL_LIGHT_CONFIG:-}" && ! -f "${LOCAL_LIGHT_CONFIG}" ]]; then
    echo "[ERR] LOCAL_LIGHT_CONFIG does not exist: ${LOCAL_LIGHT_CONFIG}" >&2
    exit 1
  fi
  if [[ -n "${POINT_LIGHTS_FILE:-}" && ! -f "${POINT_LIGHTS_FILE}" ]]; then
    echo "[ERR] POINT_LIGHTS_FILE does not exist: ${POINT_LIGHTS_FILE}" >&2
    exit 1
  fi
  if [[ -n "${POINT_LIGHTS_FILE_REANCHOR_POSE_PATH:-}" && ! -f "${POINT_LIGHTS_FILE_REANCHOR_POSE_PATH}" ]]; then
    echo "[ERR] POINT_LIGHTS_FILE_REANCHOR_POSE_PATH does not exist: ${POINT_LIGHTS_FILE_REANCHOR_POSE_PATH}" >&2
    exit 1
  fi
  append_tag "local_lights"
  append_overlay "${LOCAL_LIGHT_OVERLAY}"
  if [[ -n "${LOCAL_LIGHT_CONFIG:-}" ]]; then
    append_overlay "${LOCAL_LIGHT_CONFIG}"
  fi
  EXTRA_CONFIG_OPTS+=(trainer.tracer.use_pbr=true trainer.tracer.direct_light_mode=point_lights)
  [[ -n "${POINT_LIGHTS:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights=${POINT_LIGHTS}")
  [[ -n "${POINT_LIGHTS_FILE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file=${POINT_LIGHTS_FILE}")
  [[ -n "${POINT_LIGHTS_FILE_FORMAT:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_format=${POINT_LIGHTS_FILE_FORMAT}")
  [[ -n "${POINT_LIGHTS_FILE_ENERGY_SCALE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_energy_scale=${POINT_LIGHTS_FILE_ENERGY_SCALE}")
  [[ -n "${POINT_LIGHTS_FILE_SPREAD_ANGLE_DEG:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_spread_angle_deg=${POINT_LIGHTS_FILE_SPREAD_ANGLE_DEG}")
  [[ -n "${POINT_LIGHTS_FILE_INNER_RATIO:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_inner_ratio=${POINT_LIGHTS_FILE_INNER_RATIO}")
  [[ -n "${POINT_LIGHTS_FILE_DIRECTION_AXIS:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_direction_axis=${POINT_LIGHTS_FILE_DIRECTION_AXIS}")
  [[ -n "${POINT_LIGHTS_FILE_DIRECTION_SIGN:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_direction_sign=${POINT_LIGHTS_FILE_DIRECTION_SIGN}")
  [[ -n "${POINT_LIGHTS_FILE_REANCHOR:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_reanchor=${POINT_LIGHTS_FILE_REANCHOR}")
  [[ -n "${POINT_LIGHTS_FILE_REANCHOR_POSE_PATH:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_file_reanchor_pose_path=${POINT_LIGHTS_FILE_REANCHOR_POSE_PATH}")
  [[ -n "${POINT_LIGHTS_USE_ENVMAP:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_use_envmap=${POINT_LIGHTS_USE_ENVMAP}")
  [[ -n "${POINT_LIGHT_INTENSITY_MODE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_intensity_mode=${POINT_LIGHT_INTENSITY_MODE}")
  [[ -n "${POINT_LIGHT_FALLOFF_EXPONENT:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_falloff_exponent=${POINT_LIGHT_FALLOFF_EXPONENT}")
  [[ -n "${POINT_LIGHT_MIN_DISTANCE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_min_distance=${POINT_LIGHT_MIN_DISTANCE}")
  RENDER_POINT_LIGHT_EMITTERS_VALUE="${RENDER_POINT_LIGHT_EMITTERS:-${RENDER_LIGHT_EMITTERS:-}}"
  [[ -n "${RENDER_POINT_LIGHT_EMITTERS_VALUE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.render_point_light_emitters=${RENDER_POINT_LIGHT_EMITTERS_VALUE}")
  [[ -n "${POINT_LIGHT_EMITTER_COLOR_SCALE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_emitter_color_scale=${POINT_LIGHT_EMITTER_COLOR_SCALE}")
  [[ -n "${POINT_LIGHT_EMITTER_RADIUS:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_emitter_radius=${POINT_LIGHT_EMITTER_RADIUS}")
  if [[ "${RENDER_POINT_LIGHT_EMITTERS_VALUE:-0}" == "1" ]]; then
    append_tag "light_emitters"
  fi
fi

# Headlights.
HEADLIGHTS_ENABLED=0
if [[ "${HEADLIGHTS:-0}" == "1" || "${EVAL_HEADLIGHTS:-0}" == "1"   || -n "${HEADLIGHT_INTENSITY:-}" || -n "${HEADLIGHT_LATERAL_OFFSET:-}"   || -n "${HEADLIGHT_DOWN_OFFSET:-}" || -n "${HEADLIGHT_FORWARD_OFFSET:-}"   || -n "${HEADLIGHT_AS_SPOTLIGHT:-}" || -n "${HEADLIGHT_SPOT_DIRECTION_CAM:-}"   || -n "${HEADLIGHT_INNER_ANGLE_DEG:-}" || -n "${HEADLIGHT_OUTER_ANGLE_DEG:-}"   || -n "${EMIT_HEADLIGHTS:-}" || -n "${POINT_LIGHT_MIN_DISTANCE:-}" || -n "${HEADLIGHT_MIN_DISTANCE:-}" ]]; then
  HEADLIGHTS_ENABLED=1
fi
if [[ "${HEADLIGHTS_ENABLED}" == "1" ]]; then
  append_tag "headlights"
  EXTRA_EVAL_ARGS+=(--eval_headlight)
  EXTRA_CONFIG_OPTS+=(trainer.tracer.use_pbr=true trainer.tracer.direct_light_mode=point_lights)
  [[ -n "${POINT_LIGHT_MIN_DISTANCE:-}" || -n "${HEADLIGHT_MIN_DISTANCE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_min_distance=${POINT_LIGHT_MIN_DISTANCE:-${HEADLIGHT_MIN_DISTANCE}}")
  [[ -n "${POINT_LIGHT_INTENSITY_MODE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_intensity_mode=${POINT_LIGHT_INTENSITY_MODE}")
  [[ -n "${POINT_LIGHT_FALLOFF_EXPONENT:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_falloff_exponent=${POINT_LIGHT_FALLOFF_EXPONENT}")
  [[ -n "${HEADLIGHT_INTENSITY:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_intensity=${HEADLIGHT_INTENSITY}")
  [[ -n "${HEADLIGHT_LATERAL_OFFSET:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_lateral_offset=${HEADLIGHT_LATERAL_OFFSET}")
  [[ -n "${HEADLIGHT_DOWN_OFFSET:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_down_offset=${HEADLIGHT_DOWN_OFFSET}")
  [[ -n "${HEADLIGHT_FORWARD_OFFSET:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_forward_offset=${HEADLIGHT_FORWARD_OFFSET}")
  [[ -n "${HEADLIGHT_AS_SPOTLIGHT:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_as_spotlight=${HEADLIGHT_AS_SPOTLIGHT}")
  [[ -n "${HEADLIGHT_SPOT_DIRECTION_CAM:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_spot_direction_cam=${HEADLIGHT_SPOT_DIRECTION_CAM}")
  [[ -n "${HEADLIGHT_INNER_ANGLE_DEG:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_inner_angle_deg=${HEADLIGHT_INNER_ANGLE_DEG}")
  [[ -n "${HEADLIGHT_OUTER_ANGLE_DEG:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.eval_headlight_outer_angle_deg=${HEADLIGHT_OUTER_ANGLE_DEG}")
  [[ -n "${POINT_LIGHTS_USE_ENVMAP:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_lights_use_envmap=${POINT_LIGHTS_USE_ENVMAP}")
  EMIT_HEADLIGHTS_VALUE="${EMIT_HEADLIGHTS:-${RENDER_POINT_LIGHT_EMITTERS:-${RENDER_LIGHT_EMITTERS:-}}}"
  [[ -n "${EMIT_HEADLIGHTS_VALUE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.render_point_light_emitters=${EMIT_HEADLIGHTS_VALUE}" "trainer.tracer.emit_for_eval_headlight=${EMIT_HEADLIGHTS_VALUE}")
  [[ -n "${POINT_LIGHT_EMITTER_COLOR_SCALE:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_emitter_color_scale=${POINT_LIGHT_EMITTER_COLOR_SCALE}")
  [[ -n "${POINT_LIGHT_EMITTER_RADIUS:-}" ]] && EXTRA_CONFIG_OPTS+=("trainer.tracer.point_light_emitter_radius=${POINT_LIGHT_EMITTER_RADIUS}")
  if [[ "${EMIT_HEADLIGHTS_VALUE:-0}" == "1" ]]; then
    append_tag "light_emitters"
  fi
fi

# Dynamic insertion.
INSERT_ASSET="${INSERT_ASSET:-}"
INSERT_SPEC="${INSERT_SPEC:-}"
ASSET_SPEC="${ASSET_SPEC:-}"
RENDER_INSERTED_ONLY="${RENDER_INSERTED_ONLY:-0}"
INSERT_REPLACE_DYNAMIC="${INSERT_REPLACE_DYNAMIC:-0}"
REMOVE_TARGET_CLASSES="${REMOVE_TARGET_CLASSES:-RigidNodes,SMPLNodes,DeformableNodes}"
SHOULD_DELETE_TARGET_NODES=0
DELETE_RIGID_IDS_INTERNAL=""
DELETE_SMPL_IDS_INTERNAL=""
DELETE_DEFORMABLE_IDS_INTERNAL=""

if [[ -n "${ASSET_SPEC}" && ( -n "${INSERT_ASSET}" || -n "${INSERT_SPEC}" ) ]]; then
  echo "[ERR] Use INSERT_ASSET/INSERT_SPEC or legacy ASSET_SPEC, not both." >&2
  exit 1
fi
if [[ -n "${INSERT_ASSET}" && -n "${INSERT_SPEC}" ]]; then
  echo "[ERR] Use either INSERT_ASSET for one object or INSERT_SPEC for multiple objects, not both." >&2
  exit 1
fi

if [[ -n "${INSERT_ASSET}" || -n "${INSERT_SPEC}" ]]; then
  if [[ -n "${INSERT_SPEC}" ]]; then
    [[ -f "${INSERT_SPEC}" ]] || { echo "[ERR] INSERT_SPEC does not exist: ${INSERT_SPEC}" >&2; exit 1; }
    spec_basename="$(basename "${INSERT_SPEC}")"
    append_tag "insert_$(safe_tag "${spec_basename%.json}")"
    EXTRA_EVAL_ARGS+=(--insert_dynamic_specs_json "${INSERT_SPEC}" --insert_specs_simple)
  else
    [[ -f "${INSERT_ASSET}" ]] || { echo "[ERR] INSERT_ASSET does not exist: ${INSERT_ASSET}" >&2; exit 1; }
    INSERT_CLASSES="${INSERT_CLASSES:-}"
    if [[ -z "${INSERT_CLASSES}" ]]; then
      INSERT_CLASSES="$(${PYTHON_BIN} - "${INSERT_ASSET}" <<'INNERPY'
import sys
import torch
asset_path = sys.argv[1]
package = torch.load(asset_path, map_location="cpu")
classes = package.get("classes", {})
if not classes:
    raise KeyError(f"Asset package missing classes: {asset_path}")
print(",".join(classes.keys()))
INNERPY
)"
    fi
    INSERT_ANCHOR_TIMESTEP="${INSERT_ANCHOR_TIMESTEP:-0}"
    INSERT_ANCHOR_CAM_ID="${INSERT_ANCHOR_CAM_ID:-0}"
    INSERT_FORWARD_M="${INSERT_FORWARD_M:-8.0}"
    INSERT_RIGHT_M="${INSERT_RIGHT_M:-0.0}"
    INSERT_UP_M="${INSERT_UP_M:-0.0}"
    INSERT_YAW_DEG="${INSERT_YAW_DEG:-0.0}"
    INSERT_SCALE="${INSERT_SCALE:-1.0}"
    INSERT_LEGACY_UP_M="$(${PYTHON_BIN} -c 'import sys; print(-float(sys.argv[1]))' "${INSERT_UP_M}")"
    asset_stem="$(basename "${INSERT_ASSET}")"
    asset_stem="${asset_stem%.*}"
    append_tag "insert_$(safe_tag "${asset_stem}")"
    EXTRA_EVAL_ARGS+=(
      --insert_dynamic_asset "${INSERT_ASSET}"
      --insert_dynamic_classes "${INSERT_CLASSES}"
      --insert_anchor_timestep "${INSERT_ANCHOR_TIMESTEP}"
      --insert_cam_id "${INSERT_ANCHOR_CAM_ID}"
      --insert_forward_m "${INSERT_FORWARD_M}"
      --insert_right_m "${INSERT_RIGHT_M}"
      --insert_up_m "${INSERT_LEGACY_UP_M}"
      --insert_yaw_deg "${INSERT_YAW_DEG}"
      --insert_scale_mode manual
      --insert_manual_scale "${INSERT_SCALE}"
    )
    [[ -n "${INSERT_SOURCE_ANCHOR_TIMESTEP:-}" ]] && EXTRA_EVAL_ARGS+=(--insert_dynamic_source_anchor_timestep "${INSERT_SOURCE_ANCHOR_TIMESTEP}")
    if [[ "${INSERT_AUTO_PLACE:-0}" == "1" ]]; then
      EXTRA_EVAL_ARGS+=(--insert_dynamic_placement_mode "${INSERT_PLACEMENT_MODE:-visible}")
    else
      EXTRA_EVAL_ARGS+=(--insert_dynamic_disable_place_search)
    fi
  fi

  if [[ "${INSERT_REPLACE_DYNAMIC}" == "1" ]]; then
    SHOULD_DELETE_TARGET_NODES=1
    append_tag "replace_dynamic"
  fi
  if [[ "${RENDER_INSERTED_ONLY}" == "1" ]]; then
    EXTRA_EVAL_ARGS+=(--render_inserted_only)
    append_tag "inserted_only"
  fi
fi

# Legacy dynamic insertion interface. Prefer INSERT_ASSET/INSERT_SPEC above.
if [[ -n "${ASSET_SPEC}" ]]; then
  [[ -f "${ASSET_SPEC}" ]] || { echo "[ERR] ASSET_SPEC does not exist: ${ASSET_SPEC}" >&2; exit 1; }
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
  REMOVE_EXISTING_TARGET_NODES="${REMOVE_EXISTING_TARGET_NODES:-1}"

  MODE=""
  RESOLVED_ASSET_PATH=""
  RESOLVED_CLASSES=""
  RESOLVED_ASSET_ID=""
  SPECS_JSON_PATH=""
  while IFS=$'\t' read -r key value; do
    case "${key}" in
      mode) MODE="${value}" ;;
      asset_path) RESOLVED_ASSET_PATH="${value}" ;;
      classes) RESOLVED_CLASSES="${value}" ;;
      asset_id) RESOLVED_ASSET_ID="${value}" ;;
      specs_json) SPECS_JSON_PATH="${value}" ;;
    esac
  done < <(
    "${PYTHON_BIN}" - "${ASSET_SPEC}" "${ASSET_ID}" "${SOURCE_INSTANCE_ID}" "${OBJECT_INDEX}" "${CLASS_NAME}" <<'INNERPY'
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
    elif isinstance(payload, dict) and payload.get("export_mode") == "per_object" and isinstance(payload.get("items"), list):
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
            classes = item.get("metadata", {}).get("class_name", "")
        emit("mode", "single_asset")
        emit("asset_path", asset_path)
        emit("classes", classes)
        emit("asset_id", item.get("asset_id", os.path.splitext(os.path.basename(asset_path))[0]))
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
INNERPY
  )

  if [[ -z "${MODE}" ]]; then
    echo "[ERR] Failed to resolve ASSET_SPEC: ${ASSET_SPEC}" >&2
    exit 1
  fi
  if [[ "${MODE}" == "specs_json" ]]; then
    spec_basename="$(basename "${SPECS_JSON_PATH}")"
    append_tag "insert_${spec_basename%.json}"
    EXTRA_EVAL_ARGS+=(--insert_dynamic_specs_json "${SPECS_JSON_PATH}")
  else
    [[ -z "${CLASSES}" ]] && CLASSES="${RESOLVED_CLASSES}"
    if [[ -z "${CLASSES}" ]]; then
      echo "[ERR] Failed to resolve classes for asset: ${RESOLVED_ASSET_PATH}" >&2
      exit 1
    fi
    append_tag "insert_$(safe_tag "${RESOLVED_ASSET_ID}")"
    EXTRA_EVAL_ARGS+=(
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
    [[ -n "${SOURCE_ANCHOR_TIMESTEP}" ]] && EXTRA_EVAL_ARGS+=(--insert_dynamic_source_anchor_timestep "${SOURCE_ANCHOR_TIMESTEP}")
    [[ "${DISABLE_PLACE_SEARCH}" == "1" ]] && EXTRA_EVAL_ARGS+=(--insert_dynamic_disable_place_search)
  fi
  if [[ -n "${TARGET_WORLD_XYZ}" ]]; then
    EXTRA_EVAL_ARGS+=(--insert_target_world_xyz "${TARGET_WORLD_XYZ}")
    append_tag "target_world"
  fi
  [[ -n "${TARGET_WORLD_TIMESTEP}" ]] && EXTRA_EVAL_ARGS+=(--insert_target_world_timestep "${TARGET_WORLD_TIMESTEP}")
  [[ -n "${TARGET_WORLD_YAW_DEG}" ]] && EXTRA_EVAL_ARGS+=(--insert_target_world_yaw_deg "${TARGET_WORLD_YAW_DEG}")
  if [[ "${RENDER_INSERTED_ONLY}" == "1" ]]; then
    EXTRA_EVAL_ARGS+=(--render_inserted_only)
    append_tag "inserted_only"
  fi
  if [[ "${REMOVE_EXISTING_TARGET_NODES}" == "1" ]]; then
    SHOULD_DELETE_TARGET_NODES=1
  fi
fi

if [[ "${SHOULD_DELETE_TARGET_NODES}" == "1" ]]; then
  while IFS=$'\t' read -r class_name ids_csv; do
    case "${class_name}" in
      RigidNodes) DELETE_RIGID_IDS_INTERNAL="$(append_csv "${DELETE_RIGID_IDS_INTERNAL}" "${ids_csv}")" ;;
      SMPLNodes) DELETE_SMPL_IDS_INTERNAL="$(append_csv "${DELETE_SMPL_IDS_INTERNAL}" "${ids_csv}")" ;;
      DeformableNodes) DELETE_DEFORMABLE_IDS_INTERNAL="$(append_csv "${DELETE_DEFORMABLE_IDS_INTERNAL}" "${ids_csv}")" ;;
    esac
  done < <(
    "${PYTHON_BIN}" - "${CKPT}" "${REMOVE_TARGET_CLASSES}" <<'INNERPY'
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
INNERPY
  )
fi

if [[ -n "${DELETE_RIGID_IDS_INTERNAL}" ]]; then
  EXTRA_EVAL_ARGS+=(--delete_rigid_ids "${DELETE_RIGID_IDS_INTERNAL}")
fi
if [[ -n "${DELETE_SMPL_IDS_INTERNAL}" ]]; then
  EXTRA_EVAL_ARGS+=(--delete_smpl_ids "${DELETE_SMPL_IDS_INTERNAL}")
fi
if [[ -n "${DELETE_DEFORMABLE_IDS_INTERNAL}" ]]; then
  EXTRA_EVAL_ARGS+=(--delete_deformable_ids "${DELETE_DEFORMABLE_IDS_INTERNAL}")
fi

# Object and material edits.
MOVE_SPEC="${MOVE_SPEC:-}"
MOVE_SINGLE_RELATIVE=0
if [[ -n "${MOVE_RIGID_IDS:-}" && ( -n "${MOVE_ANCHOR_TIMESTEP:-}" || -n "${MOVE_ANCHOR_CAM_ID:-}" || -n "${MOVE_FORWARD_M:-}" || -n "${MOVE_RIGHT_M:-}" || -n "${MOVE_UP_M:-}" || -n "${MOVE_YAW_DEG:-}" || -n "${MOVE_SCALE:-}" ) ]]; then
  MOVE_SINGLE_RELATIVE=1
fi
if [[ -n "${MOVE_SPEC}" ]]; then
  if [[ -n "${MOVE_RIGID_IDS:-}" || -n "${MOVE_ANCHOR_TIMESTEP:-}" || -n "${MOVE_ANCHOR_CAM_ID:-}" || -n "${MOVE_FORWARD_M:-}" || -n "${MOVE_RIGHT_M:-}" || -n "${MOVE_UP_M:-}" || -n "${MOVE_YAW_DEG:-}" || -n "${MOVE_SCALE:-}" || -n "${MOVE_RIGID_X_M:-}" || -n "${MOVE_RIGID_Y_M:-}" || -n "${MOVE_RIGID_Z_M:-}" ]]; then
    echo "[ERR] Use MOVE_SPEC or single-object MOVE_RIGID_IDS/MOVE_* settings, not both." >&2
    exit 1
  fi
  [[ -f "${MOVE_SPEC}" ]] || { echo "[ERR] MOVE_SPEC does not exist: ${MOVE_SPEC}" >&2; exit 1; }
  move_spec_basename="$(basename "${MOVE_SPEC}")"
  append_tag "move_$(safe_tag "${move_spec_basename%.json}")"
  EXTRA_EVAL_ARGS+=(--move_rigid_specs_json "${MOVE_SPEC}")
elif [[ "${MOVE_SINGLE_RELATIVE}" == "1" ]]; then
  append_tag "move_rigid"
  EXTRA_EVAL_ARGS+=(
    --move_rigid_ids "${MOVE_RIGID_IDS}"
    --move_anchor_timestep "${MOVE_ANCHOR_TIMESTEP:-0}"
    --move_anchor_cam_id "${MOVE_ANCHOR_CAM_ID:-0}"
    --move_forward_m "${MOVE_FORWARD_M:-0.0}"
    --move_right_m "${MOVE_RIGHT_M:-0.0}"
    --move_up_m "${MOVE_UP_M:-0.0}"
    --move_yaw_deg "${MOVE_YAW_DEG:-0.0}"
    --move_scale "${MOVE_SCALE:-1.0}"
  )
elif [[ -n "${MOVE_RIGID_IDS:-}" ]]; then
  append_tag "move_rigid"
  EXTRA_EVAL_ARGS+=(
    --move_rigid_ids "${MOVE_RIGID_IDS}"
    --move_rigid_x_m "${MOVE_RIGID_X_M:-0.0}"
    --move_rigid_y_m "${MOVE_RIGID_Y_M:-0.0}"
    --move_rigid_z_m "${MOVE_RIGID_Z_M:-0.0}"
  )
fi
if [[ -n "${DELETE_RIGID_IDS}" ]]; then
  append_tag "delete_rigid"
  EXTRA_EVAL_ARGS+=(--delete_rigid_ids "${DELETE_RIGID_IDS}")
fi
if [[ -n "${DELETE_SMPL_IDS}" ]]; then
  append_tag "delete_smpl"
  EXTRA_EVAL_ARGS+=(--delete_smpl_ids "${DELETE_SMPL_IDS}")
fi
if [[ -n "${DELETE_DEFORMABLE_IDS}" ]]; then
  append_tag "delete_deformable"
  EXTRA_EVAL_ARGS+=(--delete_deformable_ids "${DELETE_DEFORMABLE_IDS}")
fi
if [[ -n "${OVERRIDE_GAUSSIAN_METALLIC:-}" ]]; then
  append_tag "metallic"
  EXTRA_EVAL_ARGS+=(--override_gaussian_metallic "${OVERRIDE_GAUSSIAN_METALLIC}")
fi
if [[ -n "${OVERRIDE_GAUSSIAN_ROUGHNESS:-}" ]]; then
  append_tag "roughness"
  EXTRA_EVAL_ARGS+=(--override_gaussian_roughness "${OVERRIDE_GAUSSIAN_ROUGHNESS}")
fi
if [[ "${NO_PBR:-0}" == "1" ]]; then
  append_tag "no_pbr"
fi

if [[ "${#TAGS[@]}" -eq 0 ]]; then
  APP_RENDER_TASK="${APP_RENDER_NAME:-application}"
else
  APP_RENDER_TASK="${APP_RENDER_NAME:-$(IFS=_; printf '%s' "${TAGS[*]}")}"
fi
POSTFIX="${POSTFIX:-_eval_${APP_RENDER_TASK}}"
VIDEO_OUTPUT_DIR="${VIDEO_OUTPUT_DIR:-${RUN_ROOT}/render/${APP_RENDER_TASK}/videos}"
CALIBRATE_ENVMAP="${CALIBRATE_ENVMAP:-1}"
EVAL_START_TIMESTEP="${START:-}"
CALIBRATE_ENVMAP_TIMESTEP="${CALIBRATE_ENVMAP_TIMESTEP:-${TIMESTEP:-${START:-0}}}"
CALIBRATE_ENVMAP_CAM_ID="${CALIBRATE_ENVMAP_CAM_ID:-${CAM_ID:-0}}"

cmd=(
  "${PYTHON_BIN}" tools/eval.py
  --resume_from "${CKPT}"
  --postfix "${POSTFIX}"
  --video_output_dir "${VIDEO_OUTPUT_DIR}"
  --calibrate_envmap_timestep "${CALIBRATE_ENVMAP_TIMESTEP}"
  --calibrate_envmap_cam_id "${CALIBRATE_ENVMAP_CAM_ID}"
)

cmd+=("${CONFIG_OVERLAY_ARGS[@]}")
cmd+=("${EXTRA_EVAL_ARGS[@]}")

if [[ -n "${EVAL_START_TIMESTEP}" ]]; then
  cmd+=(--eval_start_timestep "${EVAL_START_TIMESTEP}")
fi
if [[ -n "${NUM_FRAMES:-}" ]]; then
  cmd+=(--eval_num_timesteps "${NUM_FRAMES}")
fi
if [[ -n "${CAM_IDS}" ]]; then
  cmd+=(--eval_cam_ids "${CAM_IDS}")
fi
if [[ -n "${DATASET_SOURCE:-}" ]]; then
  cmd+=(--dataset_source "${DATASET_SOURCE}")
fi
if [[ -n "${CALIBRATE_ENVMAP_SOURCE:-}" ]]; then
  cmd+=(--calibrate_envmap_source "${CALIBRATE_ENVMAP_SOURCE}")
fi
if [[ "${CALIBRATE_ENVMAP}" == "1" ]]; then
  cmd+=(--calibrate_envmap)
fi
if [[ "${NO_PBR:-0}" == "1" ]]; then
  cmd+=(--no_pbr)
fi
if [[ -n "${RENDER_VIDEO_POSTFIX:-}" ]]; then
  cmd+=(--render_video_postfix "${RENDER_VIDEO_POSTFIX}")
fi
if [[ -n "${METRICS_OUTPUT_DIR:-}" ]]; then
  cmd+=(--metrics_output_dir "${METRICS_OUTPUT_DIR}")
fi
if [[ "${LOG_METRICS:-0}" == "1" ]]; then
  cmd+=(--log_metrics)
fi

for arg in "${EXTRA_EVAL_ARGS[@]}"; do
  if [[ "${arg}" != --* && "${arg}" == *=* ]]; then
    echo "[ERR] Internal argument-order bug: config override '${arg}' was added before eval flags. Use EXTRA_CONFIG_OPTS instead." >&2
    exit 1
  fi
done
CONFIG_OPTS_START=${#cmd[@]}
cmd+=("${EXTRA_CONFIG_OPTS[@]}")
if [[ -n "${EXTERNAL_DATA_ROOT:-}" || -n "${EXTERNAL_SCENE_IDX:-}" || -n "${EXTERNAL_RELIGHT_SCENE_IDX:-}" ]]; then
  cmd+=(data.pixel_source.external_source.enable=true)
  [[ -n "${EXTERNAL_DATA_ROOT:-}" ]] && cmd+=(data.pixel_source.external_source.data_root="${EXTERNAL_DATA_ROOT}")
  [[ -n "${EXTERNAL_SCENE_IDX:-}" ]] && cmd+=(data.pixel_source.external_source.scene_idx="${EXTERNAL_SCENE_IDX}")
  [[ -n "${EXTERNAL_RELIGHT_SCENE_IDX:-}" ]] && cmd+=(data.pixel_source.external_source.relighted_scene_idx="${EXTERNAL_RELIGHT_SCENE_IDX}")
fi
cmd+=(render.render_full=true render.render_test=false render.render_novel=null data.pixel_source.load_images_only=true)

if [[ -n "${EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTS} )
  cmd+=("${extra[@]}")
fi

for ((i = CONFIG_OPTS_START; i < ${#cmd[@]}; i++)); do
  if [[ "${cmd[$i]}" == --* ]]; then
    echo "[ERR] Eval flag '${cmd[$i]}' appears after config overrides; argparse would treat it as an OmegaConf opt. Move it before EXTRA_CONFIG_OPTS/EXTRA_OPTS." >&2
    exit 1
  fi
done

echo "[INFO] Application render task: ${APP_RENDER_TASK}"
echo "[INFO] Video output dir: ${VIDEO_OUTPUT_DIR}"
echo "[INFO] Requested range: START=${START:-<default>} NUM_FRAMES=${NUM_FRAMES:-<default>} END=${END:-<default>}"
echo "[INFO] Requested cameras: CAM_IDS=${CAM_IDS:-<all>}"
printf '[INFO] Eval command:'
printf ' %q' "${cmd[@]}"
printf '
'
if [[ "${PRINT_CMD:-0}" == "1" ]]; then
  exit 0
fi
exec "${cmd[@]}"
