#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_self_dataset.sh [scene_name] [num_timesteps] [path_name] [interval]
# Images are read from {scene}/images/XXX_Y.png (0-based 3-digit frame, camera index)

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRDFUSION_ROOT="${BRDFUSION_ROOT:-$(cd "$SCRIPT_ROOT/../.." && pwd)}"
dataset_root="${DATASET_ROOT:-${BRDFUSION_ROOT}/data/self}"
path_name=${3:-path1_tree_gamma_full}
interval=${INTERVAL:-${4:-2}}
if (( interval <= 0 )); then
    interval=1
fi

sampled_items() {
    local num_timesteps=${1:-0}
    local interval=${2:-1}

    if (( interval <= 0 )); then
        interval=1
    fi

    sampled=()
    for ((t=0; t<num_timesteps; t+=interval)); do
        sampled+=("$(printf "%03d" "$t")")
    done

    echo "${sampled[@]}"
}

scene_name=${1:-kiara_8_sunset_4k}
num_timesteps=${2:-20}

resolve_self_cameras() {
    local cam_ids="${CAM_IDS:-0}"
    if [ -n "${CAM_NAMES:-}" ]; then
        cam_ids="${CAM_NAMES}"
    fi
    local selected=()
    local cam_id
    # shellcheck disable=SC2206
    for cam_id in ${cam_ids}; do
        case "$cam_id" in
            0|Camera_Center) selected+=("Camera_Center:0") ;;
            1|Cam_Left) selected+=("Cam_Left:1") ;;
            2|Cam_Right) selected+=("Cam_Right:2") ;;
            *)
                echo "Error: unsupported self camera '$cam_id'. Use CAM_IDS=\"0 1 2\" or names Camera_Center, Cam_Left, Cam_Right." >&2
                return 1
                ;;
        esac
    done
    echo "${selected[@]}"
}

cams=()
cam_tags=()
cam_idxs=()
camera_list="$(resolve_self_cameras)" || exit 1
for entry in $camera_list; do
    cam_tags+=("${entry%%:*}")
    cams+=("${entry%%:*}")
    cam_idxs+=("${entry##*:}")
done
if ((${#cam_idxs[@]} == 0)); then
    echo "Error: CAM_IDS resolved to an empty camera list." >&2
    exit 1
fi
seeds=(0 37 71)

sampled=$(sampled_items "$num_timesteps" "$interval")
num_sampled=$(( (num_timesteps + interval - 1) / interval * ${#cams[@]} ))
echo "Selected interval: $interval"
echo "Selected cameras: ${cam_idxs[*]} (${cam_tags[*]})"
echo "Number of sampled (timestep, camera) pairs: $num_sampled"
num_seeds=${#seeds[@]}
echo "Number of seeds: $num_seeds"
echo "Total inpaintings to be performed: $((num_sampled * num_seeds))"
echo "Estimated time: $(( (num_sampled * num_seeds * 30) / 60 )) minutes"

test_image_dir=./inputs/${path_name}/${scene_name}
rm -rf "$test_image_dir"
mkdir -p "$test_image_dir"

output_image_dir=./outputs/${path_name}/${scene_name}
rm -rf "$output_image_dir"
mkdir -p "$output_image_dir"

tmp="input_crop_tmp_${path_name}_${scene_name}"
rm -rf "$tmp" "${tmp}_output"
mkdir -p "$tmp"

for frame_id in $sampled; do
    for i in "${!cams[@]}"; do
        cam_tag="${cam_tags[$i]}"
        cam_idx="${cam_idxs[$i]}"
        input_path="${dataset_root}/${path_name}/${scene_name}/images/${frame_id}_${cam_idx}.png"
        name="${scene_name}_${cam_tag}_${frame_id}.jpg"
        if [ ! -f "$input_path" ]; then
            echo "Missing input: $input_path" >&2
            exit 1
        fi
        INPUT_PATH="$input_path" OUTPUT_PATH="$tmp/$name" python - <<'PY'
import imageio.v2 as imageio
import os

input_path = os.environ["INPUT_PATH"]
output_path = os.environ["OUTPUT_PATH"]
img = imageio.imread(input_path)
imageio.imwrite(output_path, img)
PY
    done
done

python resize_crop.py \
    $tmp \
    ${tmp}_output

for frame_id in $sampled; do
    for i in "${!cams[@]}"; do
        cam_tag="${cam_tags[$i]}"
        name="${scene_name}_${cam_tag}_${frame_id}"
        cp "${tmp}_output/${name}.jpg" "$test_image_dir/${name}_crop.jpg"
    done
done

rm -r "$tmp"
rm -r "${tmp}_output"

seed_str=$(IFS=','; echo "${seeds[*]}")
echo "Using seeds: $seed_str"

python inpaint.py \
    --dataset $test_image_dir --output_dir $output_image_dir \
    --seed "$seed_str" \
    --exposure_lora_path "${EXPOSURE_LORA_PATH:-DiffusionLight/ExposureLoRA}" \
    --turbo_lora_path "${TURBO_LORA_PATH:-DiffusionLight/TurboLoRA}"
python ball2envmap.py \
    --ball_dir $output_image_dir/square --envmap_dir $output_image_dir/envmap

for seed in "${seeds[@]}"; do
    python exposure2hdr.py \
        --input_dir $output_image_dir/envmap --output_dir $output_image_dir/hdr_seed${seed} \
        --endwith "_seed${seed}.png" --preview_output 

done
