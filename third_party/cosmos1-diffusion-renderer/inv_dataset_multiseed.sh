#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./inv_dataset_multiseed.sh <dataset_id> <num_runs> [camera] [base_seed]
# Example:
#   ./inv_dataset_multiseed.sh 157 4
#   ./inv_dataset_multiseed.sh 157 4 0 1000

dataset_id_raw=${1:-}
num_runs=${2:-}
camera=${3:-0}
base_seed=${4:-1000}

if [[ -z "$dataset_id_raw" || -z "$num_runs" ]]; then
    echo "Usage: $0 <dataset_id> <num_runs> [camera] [base_seed]"
    exit 1
fi

if ! [[ "$num_runs" =~ ^[0-9]+$ ]] || (( num_runs < 1 )); then
    echo "Error: num_runs must be an integer >= 1"
    exit 1
fi

if ! [[ "$camera" =~ ^[0-9]+$ ]]; then
    echo "Error: camera must be a non-negative integer"
    exit 1
fi

if ! [[ "$base_seed" =~ ^[0-9]+$ ]]; then
    echo "Error: base_seed must be a non-negative integer"
    exit 1
fi

dataset_id=$(printf "%03d" "$dataset_id_raw")
dataset_prefix=/project2/yi-ray/BRDFusion/data/waymo/processed/training/$dataset_id

# Fixed input setup: exactly 51 real frames [000..050].
# Inference still must run with chunk_size=57; loader will repeat last frame.
num_input_frames=51
chunk_size=57
start_frame=0
end_frame=$((num_input_frames - 1))

width=1280
height=704
postfix="1cam_51steps_multiseed${num_runs}"

save_normal_dir=$dataset_prefix/diffusion_renderer_normal_$postfix
save_depth_dir=$dataset_prefix/diffusion_renderer_depth_$postfix
save_albedo_dir=$dataset_prefix/diffusion_renderer_albedo_$postfix
save_roughness_dir=$dataset_prefix/diffusion_renderer_roughness_$postfix
save_metallic_dir=$dataset_prefix/diffusion_renderer_metallic_$postfix

echo "Processing dataset $dataset_id, camera $camera, runs=$num_runs, base_seed=$base_seed"

mkdir -p "$save_normal_dir" "$save_depth_dir" "$save_albedo_dir" "$save_roughness_dir" "$save_metallic_dir"

# Validate input frames exist before launching expensive jobs.
for frame_id in $(seq -f "%03g" "$start_frame" "$end_frame"); do
    src_img="$dataset_prefix/images/${frame_id}_${camera}.jpg"
    if [[ ! -f "$src_img" ]]; then
        echo "Error: missing input image $src_img"
        exit 1
    fi
done

for ((run_idx=0; run_idx<num_runs; run_idx++)); do
    seed=$((base_seed + run_idx))
    run_tag=$(printf "%03d" "$run_idx")

    input_tmp="tmp_${dataset_id}_cam${camera}_run${run_tag}"
    output_tmp="output_tmp_${dataset_id}_cam${camera}_run${run_tag}"

    echo "Run ${run_idx}/${num_runs} (seed=$seed)"

    rm -rf "$input_tmp" "$output_tmp"
    mkdir -p "$input_tmp"

    for frame_id in $(seq -f "%03g" "$start_frame" "$end_frame"); do
        cp "$dataset_prefix/images/${frame_id}_${camera}.jpg" "$input_tmp/"
    done

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
        --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
        --dataset_path="$input_tmp" \
        --num_video_frames "$chunk_size" \
        --group_mode folder \
        --overlap_n_frames 0 \
        --chunk_mode all \
        --video_save_folder="$output_tmp" \
        --normalize_normal True \
        --resize_resolution "$height" "$width" \
        --height "$height" \
        --width "$width" \
        --seed "$seed" \
        # --offload_diffusion_transformer --offload_tokenizer

    # Single batch id is expected: 51 inputs produce one short chunk that is padded to 57 internally.
    batch_id="0000"
    for frame_id in $(seq -f "%03g" "$start_frame" "$end_frame"); do
        gbuffer_frame=$((10#$frame_id - start_frame))
        gbuffer_frame_id=$(printf "%04d" "$gbuffer_frame")

        normal_src="$output_tmp/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.normal.jpg"
        depth_src="$output_tmp/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.depth.jpg"
        albedo_src="$output_tmp/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.basecolor.jpg"
        roughness_src="$output_tmp/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.roughness.jpg"
        metallic_src="$output_tmp/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.metallic.jpg"

        [[ -f "$normal_src" ]] || { echo "Error: missing output $normal_src"; exit 1; }
        [[ -f "$depth_src" ]] || { echo "Error: missing output $depth_src"; exit 1; }
        [[ -f "$albedo_src" ]] || { echo "Error: missing output $albedo_src"; exit 1; }
        [[ -f "$roughness_src" ]] || { echo "Error: missing output $roughness_src"; exit 1; }
        [[ -f "$metallic_src" ]] || { echo "Error: missing output $metallic_src"; exit 1; }

        cp "$normal_src" "$save_normal_dir/${frame_id}_${camera}_run${run_tag}.jpg"
        cp "$depth_src" "$save_depth_dir/${frame_id}_${camera}_run${run_tag}.jpg"
        cp "$albedo_src" "$save_albedo_dir/${frame_id}_${camera}_run${run_tag}.jpg"
        cp "$roughness_src" "$save_roughness_dir/${frame_id}_${camera}_run${run_tag}.jpg"
        cp "$metallic_src" "$save_metallic_dir/${frame_id}_${camera}_run${run_tag}.jpg"
    done

    rm -rf "$input_tmp" "$output_tmp"
done

echo "Averaging outputs across $num_runs runs"
for frame_id in $(seq -f "%03g" "$start_frame" "$end_frame"); do
    python average_images.py "$save_normal_dir/${frame_id}_${camera}.jpg" "$save_normal_dir/${frame_id}_${camera}_run"*.jpg
    python average_images.py "$save_depth_dir/${frame_id}_${camera}.jpg" "$save_depth_dir/${frame_id}_${camera}_run"*.jpg
    python average_images.py "$save_albedo_dir/${frame_id}_${camera}.jpg" "$save_albedo_dir/${frame_id}_${camera}_run"*.jpg
    python average_images.py "$save_roughness_dir/${frame_id}_${camera}.jpg" "$save_roughness_dir/${frame_id}_${camera}_run"*.jpg
    python average_images.py "$save_metallic_dir/${frame_id}_${camera}.jpg" "$save_metallic_dir/${frame_id}_${camera}_run"*.jpg

    rm "$save_normal_dir/${frame_id}_${camera}_run"*.jpg
    rm "$save_depth_dir/${frame_id}_${camera}_run"*.jpg
    rm "$save_albedo_dir/${frame_id}_${camera}_run"*.jpg
    rm "$save_roughness_dir/${frame_id}_${camera}_run"*.jpg
    rm "$save_metallic_dir/${frame_id}_${camera}_run"*.jpg
done

echo "Resizing images to 1920x1280"
python resize_image.py "$save_normal_dir" 1920 1280
python resize_image.py "$save_depth_dir" 1920 1280
python resize_image.py "$save_albedo_dir" 1920 1280
python resize_image.py "$save_roughness_dir" 1920 1280
python resize_image.py "$save_metallic_dir" 1920 1280

echo "Done. Outputs saved under:"
echo "  $save_normal_dir"
echo "  $save_depth_dir"
echo "  $save_albedo_dir"
echo "  $save_roughness_dir"
echo "  $save_metallic_dir"
