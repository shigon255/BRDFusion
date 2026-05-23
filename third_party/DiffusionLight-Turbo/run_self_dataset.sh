#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_self_dataset.sh [scene_name] [num_timesteps] [path_name]
# Images are read from {scene}/images/XXX_Y.png (0-based 3-digit frame, camera index)

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRDFUSION_ROOT="${BRDFUSION_ROOT:-$(cd "$SCRIPT_ROOT/../.." && pwd)}"
dataset_root="${DATASET_ROOT:-${BRDFUSION_ROOT}/data/self}"
path_name=${3:-path1_tree_gamma_full}

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

cams=("Camera_Center" "Cam_Left" "Cam_Right")
cam_tags=("Camera_Center" "Cam_Left" "Cam_Right")
cam_idxs=(0 1 2)
interval=2
seeds=(0 37 71)

sampled=$(sampled_items "$num_timesteps" "$interval")
num_sampled=$(( (num_timesteps + interval - 1) / interval * ${#cams[@]} ))
echo "Number of sampled (timestep, camera) pairs: $num_sampled"
num_seeds=${#seeds[@]}
echo "Number of seeds: $num_seeds"
echo "Total inpaintings to be performed: $((num_sampled * num_seeds))"
echo "Estimated time: $(( (num_sampled * num_seeds * 30) / 60 )) minutes"

test_image_dir=./inputs/${path_name}/${scene_name}
mkdir -p "$test_image_dir"

output_image_dir=./outputs/${path_name}/${scene_name}
mkdir -p "$output_image_dir"

tmp="input_crop_tmp"
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
        cp ${tmp}_output/${name}.jpg $test_image_dir/${name}_crop.jpg
    done
done

rm -r $tmp
rm -r ${tmp}_output

seed_str=$(IFS=','; echo "${seeds[*]}")
echo "Using seeds: $seed_str"

python inpaint.py \
    --dataset $test_image_dir --output_dir $output_image_dir \
    --seed "$seed_str" \
    --exposure_lora_path "${EXPOSURE_LORA_PATH:-${BRDFUSION_ROOT}/assets/checkpoints/diffusionlight/models/ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500}" \
    --turbo_lora_path "${TURBO_LORA_PATH:-${BRDFUSION_ROOT}/assets/checkpoints/diffusionlight/models/rev3/Flickr2K/Flickr2kPlus_extended/checkpoint-230000}"
python ball2envmap.py \
    --ball_dir $output_image_dir/square --envmap_dir $output_image_dir/envmap

for seed in "${seeds[@]}"; do
    python exposure2hdr.py \
        --input_dir $output_image_dir/envmap --output_dir $output_image_dir/hdr_seed${seed} \
        --endwith "_seed${seed}.png" --preview_output 

done
