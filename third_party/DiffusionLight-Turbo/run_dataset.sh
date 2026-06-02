set -e
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRDFUSION_ROOT="${BRDFUSION_ROOT:-$(cd "$SCRIPT_ROOT/../.." && pwd)}"
dataset_root="${DATASET_ROOT:-${BRDFUSION_ROOT}/data/waymo/processed/training}"

sampled_items() {
    local num_timesteps=${1:-0}
    local interval=${2:-1}
    shift 2
    local cameras=("$@")

    # guard interval
    if (( interval <= 0 )); then
        interval=1
    fi

    sampled=()
    for ((t=0; t<num_timesteps; t+=interval)); do
        for c in "${cameras[@]}"; do
            sampled+=("$(printf "%03d_%d" "$t" "$c")")
        done
    done

    echo "${sampled[@]}"
}

parse_cam_ids() {
    local cam_ids="${CAM_IDS:-0}"
    local selected=()
    local cam_id
    # shellcheck disable=SC2206
    for cam_id in ${cam_ids}; do
        case "$cam_id" in
            0|1|2|3|4) selected+=("$cam_id") ;;
            *)
                echo "Error: unsupported Waymo camera '$cam_id'. Use CAM_IDS with ids 0 1 2 3 4." >&2
                return 1
                ;;
        esac
    done
    echo "${selected[@]}"
}

scene=$1
num_timesteps=$2
interval=${3:-2}
if (( interval <= 0 )); then
    interval=1
fi
camera_list="$(parse_cam_ids)" || exit 1
read -r -a cameras <<< "$camera_list"
if ((${#cameras[@]} == 0)); then
    echo "Error: CAM_IDS resolved to an empty camera list." >&2
    exit 1
fi

# scene=16
# num_timesteps=199

seeds=(0 37 71)

scene_idx=$(printf "%03d" $scene)

sampled=$(sampled_items "$num_timesteps" "$interval" "${cameras[@]}")
# print the number of sampled (timestep, camera) pairs
num_sampled=$(( (num_timesteps + interval - 1) / interval * ${#cameras[@]} ))
echo "Selected interval: $interval"
echo "Selected cameras: ${cameras[*]}"
echo "Number of sampled (timestep, camera) pairs: $num_sampled"
num_seeds=${#seeds[@]}
echo "Number of seeds: $num_seeds"
echo "Total inpaintings to be performed: $((num_sampled * num_seeds))"
echo "Estimated time: $(( (num_sampled * num_seeds * 30) / 60 )) minutes"

test_image_dir=./inputs/$scene_idx
rm -rf $test_image_dir
mkdir -p $test_image_dir

output_image_dir=./outputs/$scene_idx
rm -rf $output_image_dir
mkdir -p $output_image_dir

tmp="input_crop_tmp_${scene_idx}"
rm -rf $tmp ${tmp}_output
mkdir -p $tmp

for item in $sampled; do
    input_path="${dataset_root}/${scene_idx}/images/${item}.jpg"
    name=${scene_idx}_${item}.jpg
    cp "$input_path" "$tmp/${name}"
done

python resize_crop.py \
    $tmp \
    ${tmp}_output

for item in $sampled; do
    name=${scene_idx}_${item}
    cp "${tmp}_output/${name}.jpg" "$test_image_dir/${name}_crop.jpg"
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
        --endwith "_seed${seed}.png" --preview_output \

done
