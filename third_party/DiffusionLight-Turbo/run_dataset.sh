set -e
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRDFUSION_ROOT="${BRDFUSION_ROOT:-$(cd "$SCRIPT_ROOT/../.." && pwd)}"
dataset_root="${DATASET_ROOT:-${BRDFUSION_ROOT}/data/waymo/processed/training}"

sampled_items() {
    local num_timesteps=${1:-0}
    local num_cams=${2:-0}
    local interval=${3:-1}

    # guard interval
    if (( interval <= 0 )); then
        interval=1
    fi

    sampled=()
    for ((t=0; t<num_timesteps; t+=interval)); do
        for ((c=0; c<num_cams; c++)); do
            sampled+=("$(printf "%03d_%d" "$t" "$c")")
        done
    done

    num_sampled=$(( (num_timesteps + interval - 1) / interval ))

    echo "${sampled[@]}"
}

scene=$1
num_timesteps=$2
interval=${3:-2}
num_cams=3

# scene=16
# num_timesteps=199

seeds=(0 37 71)

scene_idx=$(printf "%03d" $scene)

sampled=$(sampled_items $num_timesteps $num_cams $interval)
# print the number of sampled (timestep, camera) pairs
num_sampled=$(( (num_timesteps + interval - 1) / interval * num_cams ))
echo "Number of sampled (timestep, camera) pairs: $num_sampled"
num_seeds=${#seeds[@]}
echo "Number of seeds: $num_seeds"
echo "Total inpaintings to be performed: $((num_sampled * num_seeds))"
echo "Estimated time: $(( (num_sampled * num_seeds * 30) / 60 )) minutes"

test_image_dir=./inputs/$scene_idx
mkdir -p $test_image_dir

output_image_dir=./outputs/$scene_idx
mkdir -p $output_image_dir

tmp="input_crop_tmp"
mkdir -p $tmp

for item in $sampled; do
    input_path="${dataset_root}/${scene_idx}/images/${item}.jpg"
    name=${scene_idx}_${item}.jpg
    cp $input_path $tmp/${name}
done

python resize_crop.py \
    $tmp \
    ${tmp}_output

for item in $sampled; do
    name=${scene_idx}_${item}
    cp ${tmp}_output/${name}.jpg $test_image_dir/${name}_crop.jpg
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
        --endwith "_seed${seed}.png" --preview_output \

done
