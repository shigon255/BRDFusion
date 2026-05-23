#!/usr/bin/env bash
# Script for inverse rendering on self-captured dataset (gamma_full layout), with multi-seed averaging.
# Runs inference N times with different seeds and averages results per frame.
#
# Usage:
#   # Process all scenes under path1_tree_gamma_full, 4 runs, base_seed 1000
#   bash inv_dataset_self_multiseed.sh "" path1_tree_gamma_full "" 4 1000
#
#   # Process one scene
#   bash inv_dataset_self_multiseed.sh sunny_rose_garden_4k path1_tree_gamma_full "" 4 1000
#
#   # Process one scene, first 40 timesteps, 8 runs
#   bash inv_dataset_self_multiseed.sh sunny_rose_garden_4k path1_tree_gamma_full 40 8 1000

set -euo pipefail

sliding_window_starts() {
    local total_length=$1
    local window_length=$2
    local overlap_length=$3

    local step=$(( window_length - overlap_length ))
    if (( step <= 0 )); then
        echo "Error: overlap_length must be less than window_length" >&2
        return 1
    fi

    local cutoff=$(( total_length - overlap_length ))
    if (( cutoff <= 0 )); then
        return 0
    fi

    local starts=()
    local start=0
    while (( start < cutoff )); do
        starts+=("$start")
        (( start += step ))
    done

    echo "${starts[@]}"
}

dataset_base=/project2/yi-ray/BRDFusion/data/self
scene_name=${1:-}
path_name=${2:-path1_tree_gamma_full}
num_timestep_override=${3:-}
num_runs=${4:-4}
base_seed=${5:-1000}

# Inverse renderer expects 57-frame chunks (8n+1), keep this fixed.
chunk_size=57
overlap_n_frames=50
width=1280
height=704

if (( chunk_size != 57 )); then
    echo "Error: chunk_size must be 57"
    exit 1
fi

if [ -n "$num_timestep_override" ]; then
    if ! [[ "$num_timestep_override" =~ ^[0-9]+$ ]] || [ "$num_timestep_override" -le 0 ]; then
        echo "Error: <num_timesteps> must be a positive integer, got: $num_timestep_override"
        echo "Usage: $0 [scene_name] [path_name] [num_timesteps] [num_runs] [base_seed]"
        exit 1
    fi
fi

if ! [[ "$num_runs" =~ ^[0-9]+$ ]] || [ "$num_runs" -le 0 ]; then
    echo "Error: <num_runs> must be a positive integer, got: $num_runs"
    echo "Usage: $0 [scene_name] [path_name] [num_timesteps] [num_runs] [base_seed]"
    exit 1
fi

if ! [[ "$base_seed" =~ ^[0-9]+$ ]]; then
    echo "Error: <base_seed> must be a non-negative integer, got: $base_seed"
    echo "Usage: $0 [scene_name] [path_name] [num_timesteps] [num_runs] [base_seed]"
    exit 1
fi

declare -A cam_idx_map
cam_idx_map=( ["Camera_Center"]="0" ["Cam_Left"]="1" ["Cam_Right"]="2" )

if [ -n "$scene_name" ]; then
    scenes=("$scene_name")
else
    scenes=($(ls -d "$dataset_base/$path_name"/*/ | xargs -n 1 basename))
fi

# cameras=("Camera_Center" "Cam_Left" "Cam_Right")
cameras=("Camera_Center")
postfix="1cam_51steps_multiseed${num_runs}"

for scene in "${scenes[@]}"; do
    echo "=========================================="
    echo "Processing scene: $scene"
    echo "=========================================="

    scene_path=$dataset_base/$path_name/$scene

    save_normal_dir=$scene_path/diffusion_renderer_normal_$postfix
    save_depth_dir=$scene_path/diffusion_renderer_depth_$postfix
    save_albedo_dir=$scene_path/diffusion_renderer_albedo_$postfix
    save_roughness_dir=$scene_path/diffusion_renderer_roughness_$postfix
    save_metallic_dir=$scene_path/diffusion_renderer_metallic_$postfix

    mkdir -p "$save_normal_dir" "$save_depth_dir" "$save_albedo_dir" "$save_roughness_dir" "$save_metallic_dir"

    for camera in "${cameras[@]}"; do
        echo "Processing camera: $camera"

        cam_idx=${cam_idx_map[$camera]}
        images_dir=$scene_path/images

        if [ ! -d "$images_dir" ]; then
            echo "Warning: Images directory $images_dir not found, skipping..."
            continue
        fi

        num_frames=$(ls "$images_dir"/*_"$cam_idx".png 2>/dev/null | wc -l)
        if [ "$num_frames" -eq 0 ]; then
            echo "Warning: No frames found for camera $camera (index $cam_idx) in $images_dir, skipping..."
            continue
        fi

        use_num_frames=$num_frames
        if [ -n "$num_timestep_override" ]; then
            if (( num_timestep_override < num_frames )); then
                use_num_frames=$num_timestep_override
                echo "Detected $num_frames frames, using first $use_num_frames frames due to num_timesteps override"
            else
                echo "Detected $num_frames frames, num_timesteps override ($num_timestep_override) exceeds available frames; using all $num_frames"
            fi
        else
            echo "Detected $num_frames frames"
        fi

        for ((run_idx=0; run_idx<num_runs; run_idx++)); do
            seed=$((base_seed + run_idx))
            run_tag=$(printf "%03d" "$run_idx")
            run_tmp="tmp_${path_name}_${scene}_cam${cam_idx}_run${run_tag}"
            run_out="output_tmp_${path_name}_${scene}_cam${cam_idx}_run${run_tag}"

            echo "Run ${run_idx}/${num_runs} (seed=$seed)"

            rm -rf "$run_tmp" "$run_out"
            mkdir -p "$run_tmp"

            if (( use_num_frames < chunk_size )); then
                echo "Short sequence detected, padding to $chunk_size frames"

                for ((fi=0; fi<use_num_frames; fi++)); do
                    src_name=$(printf "%03d_%s.png" "$fi" "$cam_idx")
                    dst_name=$(printf "frame_%04d.png" $((fi + 1)))
                    cp "$images_dir/$src_name" "$run_tmp/$dst_name"
                done

                last_frame=$(printf "%03d_%s.png" $((use_num_frames - 1)) "$cam_idx")
                for frame_id in $(seq -f "%04g" $((use_num_frames + 1)) "$chunk_size"); do
                    cp "$images_dir/$last_frame" "$run_tmp/frame_${frame_id}.png"
                done

                CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
                    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
                    --dataset_path="$run_tmp" \
                    --num_video_frames "$chunk_size" \
                    --group_mode folder \
                    --video_save_folder="$run_out" \
                    --normalize_normal True \
                    --resize_resolution "$height" "$width" \
                    --height "$height" \
                    --width "$width" \
                    --seed "$seed" \
                    # --offload_diffusion_transformer --offload_tokenizer

                # Keep only real frames; padded tail (if any) is discarded.
                for frame_idx in $(seq 0 $((use_num_frames - 1))); do
                    gbuffer_frame_id=$(printf "%04d" "$frame_idx")
                    out_name=$(printf "%03d_%s" "$frame_idx" "$cam_idx")

                    cp "$run_out/gbuffer_frames/0000.${gbuffer_frame_id}.normal.jpg" "$save_normal_dir/${out_name}_run${run_tag}.jpg"
                    cp "$run_out/gbuffer_frames/0000.${gbuffer_frame_id}.depth.jpg" "$save_depth_dir/${out_name}_run${run_tag}.jpg"
                    cp "$run_out/gbuffer_frames/0000.${gbuffer_frame_id}.basecolor.jpg" "$save_albedo_dir/${out_name}_run${run_tag}.jpg"
                    cp "$run_out/gbuffer_frames/0000.${gbuffer_frame_id}.roughness.jpg" "$save_roughness_dir/${out_name}_run${run_tag}.jpg"
                    cp "$run_out/gbuffer_frames/0000.${gbuffer_frame_id}.metallic.jpg" "$save_metallic_dir/${out_name}_run${run_tag}.jpg"
                done

            else
                echo "Long sequence detected, using sliding window (chunk_size=$chunk_size, overlap=$overlap_n_frames)"

                for ((fi=0; fi<use_num_frames; fi++)); do
                    src_name=$(printf "%03d_%s.png" "$fi" "$cam_idx")
                    dst_name=$(printf "frame_%04d.png" $((fi + 1)))
                    cp "$images_dir/$src_name" "$run_tmp/$dst_name"
                done

                starting_frames=($(sliding_window_starts "$use_num_frames" "$chunk_size" "$overlap_n_frames"))

                CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
                    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
                    --dataset_path="$run_tmp" \
                    --num_video_frames "$chunk_size" \
                    --group_mode folder \
                    --overlap_n_frames "$overlap_n_frames" \
                    --chunk_mode all \
                    --video_save_folder="$run_out" \
                    --normalize_normal True \
                    --resize_resolution "$height" "$width" \
                    --height "$height" \
                    --width "$width" \
                    --seed "$seed" \
                    # --offload_diffusion_transformer --offload_tokenizer

                for ((i=0; i<${#starting_frames[@]}; i++)); do
                    start_frame=${starting_frames[$i]}
                    end_frame=$((start_frame + chunk_size - 1))
                    if (( end_frame >= use_num_frames )); then
                        end_frame=$((use_num_frames - 1))
                    fi
                    batch_id=$(printf "%04d" "$i")

                    # Keep only real frames inside this chunk; padded tail is ignored.
                    for frame_idx in $(seq "$start_frame" "$end_frame"); do
                        gbuffer_frame=$((frame_idx - start_frame))
                        gbuffer_frame_id=$(printf "%04d" "$gbuffer_frame")
                        out_name=$(printf "%03d_%s" "$frame_idx" "$cam_idx")

                        cp "$run_out/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.normal.jpg" "$save_normal_dir/${out_name}_run${run_tag}_${batch_id}.jpg"
                        cp "$run_out/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.depth.jpg" "$save_depth_dir/${out_name}_run${run_tag}_${batch_id}.jpg"
                        cp "$run_out/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.basecolor.jpg" "$save_albedo_dir/${out_name}_run${run_tag}_${batch_id}.jpg"
                        cp "$run_out/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.roughness.jpg" "$save_roughness_dir/${out_name}_run${run_tag}_${batch_id}.jpg"
                        cp "$run_out/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.metallic.jpg" "$save_metallic_dir/${out_name}_run${run_tag}_${batch_id}.jpg"
                    done
                done

                for frame_idx in $(seq 0 $((use_num_frames - 1))); do
                    out_name=$(printf "%03d_%s" "$frame_idx" "$cam_idx")

                    python average_images.py "$save_normal_dir/${out_name}_run${run_tag}.jpg" "$save_normal_dir/${out_name}_run${run_tag}_"*.jpg
                    python average_images.py "$save_depth_dir/${out_name}_run${run_tag}.jpg" "$save_depth_dir/${out_name}_run${run_tag}_"*.jpg
                    python average_images.py "$save_albedo_dir/${out_name}_run${run_tag}.jpg" "$save_albedo_dir/${out_name}_run${run_tag}_"*.jpg
                    python average_images.py "$save_roughness_dir/${out_name}_run${run_tag}.jpg" "$save_roughness_dir/${out_name}_run${run_tag}_"*.jpg
                    python average_images.py "$save_metallic_dir/${out_name}_run${run_tag}.jpg" "$save_metallic_dir/${out_name}_run${run_tag}_"*.jpg

                    rm "$save_normal_dir/${out_name}_run${run_tag}_"*.jpg
                    rm "$save_depth_dir/${out_name}_run${run_tag}_"*.jpg
                    rm "$save_albedo_dir/${out_name}_run${run_tag}_"*.jpg
                    rm "$save_roughness_dir/${out_name}_run${run_tag}_"*.jpg
                    rm "$save_metallic_dir/${out_name}_run${run_tag}_"*.jpg
                done
            fi

            rm -rf "$run_tmp" "$run_out"
        done

        echo "Averaging across $num_runs runs for camera: $camera"
        for frame_idx in $(seq 0 $((use_num_frames - 1))); do
            out_name=$(printf "%03d_%s" "$frame_idx" "$cam_idx")

            python average_images.py "$save_normal_dir/${out_name}.jpg" "$save_normal_dir/${out_name}_run"*.jpg
            python average_images.py "$save_depth_dir/${out_name}.jpg" "$save_depth_dir/${out_name}_run"*.jpg
            python average_images.py "$save_albedo_dir/${out_name}.jpg" "$save_albedo_dir/${out_name}_run"*.jpg
            python average_images.py "$save_roughness_dir/${out_name}.jpg" "$save_roughness_dir/${out_name}_run"*.jpg
            python average_images.py "$save_metallic_dir/${out_name}.jpg" "$save_metallic_dir/${out_name}_run"*.jpg

            rm "$save_normal_dir/${out_name}_run"*.jpg
            rm "$save_depth_dir/${out_name}_run"*.jpg
            rm "$save_albedo_dir/${out_name}_run"*.jpg
            rm "$save_roughness_dir/${out_name}_run"*.jpg
            rm "$save_metallic_dir/${out_name}_run"*.jpg
        done

        echo "Finished processing camera: $camera"
    done

    echo "Resizing images to 1920x1280"
    python resize_image.py "$save_normal_dir" 1920 1280
    python resize_image.py "$save_depth_dir" 1920 1280
    python resize_image.py "$save_albedo_dir" 1920 1280
    python resize_image.py "$save_roughness_dir" 1920 1280
    python resize_image.py "$save_metallic_dir" 1920 1280

    echo "Finished processing scene: $scene"
done

echo "=========================================="
echo "All processing completed successfully!"
echo "=========================================="
