#!/usr/bin/env bash
set -euo pipefail

: "${DATASET:?Set DATASET to waymo or self.}"
: "${DATA_ROOT:?Set DATA_ROOT to the split/path root containing the scene directory.}"
: "${SCENE:?Set SCENE to the scene index/name.}"

CAM_IDS="${CAM_IDS:-0}"
FRAME="${FRAME:-0}"
CHECK_PRIORS="${CHECK_PRIORS:-0}"

if [[ "${DATASET}" == "waymo" ]]; then
  if [[ "${SCENE}" =~ ^[0-9]+$ ]]; then
    scene_dir="${DATA_ROOT}/$(printf "%03d" "${SCENE}")"
  else
    scene_dir="${DATA_ROOT}/${SCENE}"
  fi
  image_ext="jpg"
  required_dirs=(images sky_masks lidar ego_pose extrinsics intrinsics)
elif [[ "${DATASET}" == "self" ]]; then
  scene_dir="${DATA_ROOT}/${SCENE}"
  image_ext="png"
  required_dirs=(images sky_masks)
else
  echo "Unsupported DATASET=${DATASET}; expected waymo or self." >&2
  exit 1
fi

[[ -d "${scene_dir}" ]] || { echo "Scene directory missing: ${scene_dir}" >&2; exit 1; }

for dir_name in "${required_dirs[@]}"; do
  [[ -d "${scene_dir}/${dir_name}" ]] || {
    echo "Required directory missing: ${scene_dir}/${dir_name}" >&2
    exit 1
  }
done

# shellcheck disable=SC2206
cams=( ${CAM_IDS} )
frame_id="$(printf "%03d" "${FRAME}")"

for cam in "${cams[@]}"; do
  image_path="${scene_dir}/images/${frame_id}_${cam}.${image_ext}"
  sky_path="${scene_dir}/sky_masks/${frame_id}_${cam}.png"
  [[ -f "${image_path}" ]] || { echo "Expected image missing: ${image_path}" >&2; exit 1; }
  [[ -f "${sky_path}" ]] || { echo "Expected sky mask missing: ${sky_path}" >&2; exit 1; }
done

if [[ "${CHECK_PRIORS}" == "1" ]]; then
  prior_dirs=(
    diffusion_renderer_normal
    diffusion_renderer_depth
    diffusion_renderer_albedo
    diffusion_renderer_roughness
    diffusion_renderer_metallic
  )
  for dir_name in "${prior_dirs[@]}"; do
    [[ -d "${scene_dir}/${dir_name}" ]] || {
      echo "Prior directory missing: ${scene_dir}/${dir_name}" >&2
      exit 1
    }
    for cam in "${cams[@]}"; do
      prior_path="${scene_dir}/${dir_name}/${frame_id}_${cam}.jpg"
      [[ -f "${prior_path}" ]] || { echo "Expected prior missing: ${prior_path}" >&2; exit 1; }
    done
  done
fi

echo "Layout OK: ${scene_dir}"
