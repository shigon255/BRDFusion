#!/usr/bin/env bash
set -euo pipefail

: "${DATASET:?Set DATASET to waymo or self.}"
: "${DATA_ROOT:?Set DATA_ROOT to the split/path root containing the scene directory.}"
: "${SCENE:?Set SCENE to the scene index/name.}"

CAM_IDS="${CAM_IDS:-0}"
FRAME="${FRAME:-0}"
CHECK_PRIORS="${CHECK_PRIORS:-0}"
CHECK_GT="${CHECK_GT:-0}"
CHECK_GT_SKY="${CHECK_GT_SKY:-0}"
CHECK_ENVMAP="${CHECK_ENVMAP:-0}"

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
  required_dirs=(images sky_masks intrinsics)
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


if [[ "${DATASET}" == "self" ]]; then
  for cam in ${CAM_IDS}; do
    case "${cam}" in
      0) cam_name="Camera_Center" ;;
      1) cam_name="Cam_Left" ;;
      2) cam_name="Cam_Right" ;;
      *) echo "Unsupported self camera id ${cam}; expected 0, 1, or 2." >&2; exit 1 ;;
    esac
    pose_path="${scene_dir}/${cam_name}_poses.txt"
    if [[ "${cam}" == "0" && ! -f "${pose_path}" && -f "${scene_dir}/Cam_Center_poses.txt" ]]; then
      pose_path="${scene_dir}/Cam_Center_poses.txt"
    fi
    intrinsic_path="${scene_dir}/intrinsics/${cam_name}.txt"
    if [[ "${cam}" == "0" && ! -f "${intrinsic_path}" && -f "${scene_dir}/intrinsics/Cam_Center.txt" ]]; then
      intrinsic_path="${scene_dir}/intrinsics/Cam_Center.txt"
    fi
    [[ -f "${pose_path}" ]] || { echo "Expected camera pose file missing: ${pose_path}" >&2; exit 1; }
    [[ -f "${intrinsic_path}" ]] || { echo "Expected intrinsic file missing: ${intrinsic_path}" >&2; exit 1; }
  done
fi

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


if [[ "${DATASET}" == "self" && "${CHECK_GT}" == "1" ]]; then
  gt_dirs=(depth normal albedo metallic roughness)
  for dir_name in "${gt_dirs[@]}"; do
    [[ -d "${scene_dir}/${dir_name}" ]] || {
      echo "GT directory missing: ${scene_dir}/${dir_name}" >&2
      exit 1
    }
  done
  for cam in "${cams[@]}"; do
    [[ -f "${scene_dir}/depth/${frame_id}_${cam}.exr" ]] || { echo "Expected GT depth missing: ${scene_dir}/depth/${frame_id}_${cam}.exr" >&2; exit 1; }
    [[ -f "${scene_dir}/normal/${frame_id}_${cam}.exr" ]] || { echo "Expected GT normal missing: ${scene_dir}/normal/${frame_id}_${cam}.exr" >&2; exit 1; }
    [[ -f "${scene_dir}/albedo/${frame_id}_${cam}.exr" ]] || { echo "Expected GT albedo missing: ${scene_dir}/albedo/${frame_id}_${cam}.exr" >&2; exit 1; }
    [[ -f "${scene_dir}/metallic/${frame_id}_${cam}.exr" ]] || { echo "Expected GT metallic missing: ${scene_dir}/metallic/${frame_id}_${cam}.exr" >&2; exit 1; }
    [[ -f "${scene_dir}/roughness/${frame_id}_${cam}.exr" ]] || { echo "Expected GT roughness missing: ${scene_dir}/roughness/${frame_id}_${cam}.exr" >&2; exit 1; }
  done
fi

if [[ "${DATASET}" == "self" && "${CHECK_GT_SKY}" == "1" ]]; then
  [[ -d "${scene_dir}/gt_sky_mask" ]] || { echo "GT sky-mask directory missing: ${scene_dir}/gt_sky_mask" >&2; exit 1; }
  for cam in "${cams[@]}"; do
    [[ -f "${scene_dir}/gt_sky_mask/${frame_id}_${cam}.png" ]] || { echo "Expected GT sky mask missing: ${scene_dir}/gt_sky_mask/${frame_id}_${cam}.png" >&2; exit 1; }
  done
fi

if [[ "${DATASET}" == "self" && "${CHECK_ENVMAP}" == "1" ]]; then
  envmap_path="${scene_dir}/${SCENE}.exr"
  [[ -f "${envmap_path}" ]] || { echo "Expected scene envmap missing: ${envmap_path}" >&2; exit 1; }
fi


echo "Layout OK: ${scene_dir}"
