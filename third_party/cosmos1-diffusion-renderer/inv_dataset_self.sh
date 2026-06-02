#!/bin/bash
# Script for inverse rendering on self-captured dataset (gamma_full layout)
# Dataset path: /project2/yi-ray/BRDFusion/data/self/<path>
# Each scene has 3 cameras (Camera_Center, Cam_Left, Cam_Right)
# RGB images are stored in {scene}/images/ with naming XXX_Y.png (0-based frame, camera index)
# Supports both short sequences (<57 frames, padded) and long sequences (>=57 frames, sliding window)

# Usage:
# # process all scenes under path1_tree_gamma_full
# bash /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/inv_dataset_self.sh "" path1_tree_gamma_full
# # process one scene under path1_tree_gamma_full
# bash /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/inv_dataset_self.sh sunny_rose_garden_4k path1_tree_gamma_full
# # process only first 40 timesteps for one scene
# bash /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/inv_dataset_self.sh sunny_rose_garden_4k path1_tree_gamma_full 40
# set -e

# Sliding window function for long sequences
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
    # If cutoff <= 0, Python's range(0, cutoff, step) would be empty.
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

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRDFUSION_ROOT="${BRDFUSION_ROOT:-$(cd "$SCRIPT_ROOT/../.." && pwd)}"
dataset_base="${DATASET_BASE:-${BRDFUSION_ROOT}/data/self}"
path_name=${2:-path1_tree_gamma_full}
num_timestep_override=$3
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_ROOT}/checkpoints}"
chunk_size=57  # Required by the model
overlap_n_frames=50
width=1280
height=704

if [ -n "$num_timestep_override" ]; then
    if ! [[ "$num_timestep_override" =~ ^[0-9]+$ ]] || [ "$num_timestep_override" -le 0 ]; then
        echo "Error: <num_timesteps> must be a positive integer, got: $num_timestep_override"
        echo "Usage: $0 [scene_name] [path_name] [num_timesteps]"
        exit 1
    fi
fi

# Camera name to index mapping (for images/XXX_Y.png naming)
declare -A cam_idx_map
cam_idx_map=( ["Camera_Center"]="0" ["Cam_Left"]="1" ["Cam_Right"]="2" )

# Get scene name from argument, or process all scenes in the selected path
if [ -n "$1" ]; then
    scenes=("$1")
else
    scenes=($(ls -d $dataset_base/$path_name/*/ | xargs -n 1 basename))
fi

resolve_self_cameras() {
    local selected=()
    if [ -n "${CAM_NAMES:-}" ]; then
        # shellcheck disable=SC2206
        selected=( ${CAM_NAMES} )
    else
        local cam_ids="${CAM_IDS:-0}"
        local cam_id
        # shellcheck disable=SC2206
        for cam_id in ${cam_ids}; do
            case "$cam_id" in
                0) selected+=("Camera_Center") ;;
                1) selected+=("Cam_Left") ;;
                2) selected+=("Cam_Right") ;;
                Camera_Center|Cam_Left|Cam_Right) selected+=("$cam_id") ;;
                *)
                    echo "Error: unsupported self camera '$cam_id'. Use CAM_IDS=\"0 1 2\" or CAM_NAMES=\"Camera_Center Cam_Left Cam_Right\"." >&2
                    return 1
                    ;;
            esac
        done
    fi

    local camera
    for camera in "${selected[@]}"; do
        if [ -z "${cam_idx_map[$camera]+x}" ]; then
            echo "Error: unsupported self camera name '$camera'. Use Camera_Center, Cam_Left, or Cam_Right." >&2
            return 1
        fi
    done

    echo "${selected[@]}"
}

camera_list="$(resolve_self_cameras)" || exit 1
read -r -a cameras <<< "$camera_list"
echo "Selected cameras: ${cameras[*]}"
postfix=""
for scene in "${scenes[@]}"; do
    echo "=========================================="
    echo "Processing scene: $scene"
    echo "=========================================="

    scene_path=$dataset_base/$path_name/$scene

    # Create output directories
    save_normal_dir=$scene_path/diffusion_renderer_normal$postfix
    save_depth_dir=$scene_path/diffusion_renderer_depth$postfix
    save_albedo_dir=$scene_path/diffusion_renderer_albedo$postfix
    save_roughness_dir=$scene_path/diffusion_renderer_roughness$postfix
    save_metallic_dir=$scene_path/diffusion_renderer_metallic$postfix

    mkdir -p $save_normal_dir
    mkdir -p $save_depth_dir
    mkdir -p $save_albedo_dir
    mkdir -p $save_roughness_dir
    mkdir -p $save_metallic_dir

    for camera in "${cameras[@]}"; do
        echo "Processing camera: $camera"

        cam_idx=${cam_idx_map[$camera]}
        images_dir=$scene_path/images

        if [ ! -d "$images_dir" ]; then
            echo "Warning: Images directory $images_dir not found, skipping..."
            continue
        fi

        # Detect actual number of frames for this camera (files matching *_${cam_idx}.png)
        num_frames=$(ls "$images_dir"/*_${cam_idx}.png 2>/dev/null | wc -l)
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

        # Create temp directory and copy frames
        mkdir -p ./tmp_${path_name}_${scene}

        if (( use_num_frames < chunk_size )); then
            # SHORT SEQUENCE: Pad with last frame to reach chunk_size
            echo "Short sequence detected, padding to $chunk_size frames"

            # Copy original frames (source: 0-based 3-digit XXX_Y.png -> tmp: 1-based 4-digit frame_XXXX.png)
            for ((fi=0; fi<use_num_frames; fi++)); do
                src_name=$(printf "%03d_%s.png" $fi $cam_idx)
                dst_name=$(printf "frame_%04d.png" $((fi + 1)))
                cp "$images_dir/$src_name" "./tmp_${path_name}_${scene}/$dst_name"
            done

            # Repeat the last frame to reach chunk_size
            last_frame=$(printf "%03d_%s.png" $((use_num_frames - 1)) $cam_idx)
            for frame_id in $(seq -f "%04g" $((use_num_frames + 1)) $chunk_size); do
                cp "$images_dir/$last_frame" "./tmp_${path_name}_${scene}/frame_${frame_id}.png"
            done

            echo "Prepared $chunk_size frames (original: $use_num_frames, repeated last frame: $((chunk_size - use_num_frames)))"

            # Run inverse rendering (single chunk, no overlap)
            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
                --checkpoint_dir "$CHECKPOINT_DIR" --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
                --dataset_path=tmp_${path_name}_${scene} \
                --num_video_frames $chunk_size \
                --group_mode folder \
                --video_save_folder=output_tmp_${path_name}_${scene} \
                --normalize_normal True \
                --resize_resolution $height $width \
                --height $height \
                --width $width \
                --offload_diffusion_transformer --offload_tokenizer

            # Move output frames (only keep original frames, not the repeated ones)
            for frame_idx in $(seq 0 $((use_num_frames - 1))); do
                gbuffer_frame_id=$(printf "%04d" $frame_idx)
                out_name=$(printf "%03d_%s" $frame_idx $cam_idx)

                mv "output_tmp_${path_name}_${scene}/gbuffer_frames/0000.${gbuffer_frame_id}.normal.jpg" "$save_normal_dir/${out_name}.jpg"
                mv "output_tmp_${path_name}_${scene}/gbuffer_frames/0000.${gbuffer_frame_id}.depth.jpg" "$save_depth_dir/${out_name}.jpg"
                mv "output_tmp_${path_name}_${scene}/gbuffer_frames/0000.${gbuffer_frame_id}.basecolor.jpg" "$save_albedo_dir/${out_name}.jpg"
                mv "output_tmp_${path_name}_${scene}/gbuffer_frames/0000.${gbuffer_frame_id}.roughness.jpg" "$save_roughness_dir/${out_name}.jpg"
                mv "output_tmp_${path_name}_${scene}/gbuffer_frames/0000.${gbuffer_frame_id}.metallic.jpg" "$save_metallic_dir/${out_name}.jpg"
            done

        else
            # LONG SEQUENCE: Use sliding window with overlap
            echo "Long sequence detected, using sliding window (chunk_size=$chunk_size, overlap=$overlap_n_frames)"

            # Copy all frames (source: 0-based 3-digit XXX_Y.png -> tmp: 1-based 4-digit frame_XXXX.png)
            for ((fi=0; fi<use_num_frames; fi++)); do
                src_name=$(printf "%03d_%s.png" $fi $cam_idx)
                dst_name=$(printf "frame_%04d.png" $((fi + 1)))
                cp "$images_dir/$src_name" "./tmp_${path_name}_${scene}/$dst_name"
            done

            # Calculate sliding window starts
            starting_frames=($(sliding_window_starts $use_num_frames $chunk_size $overlap_n_frames))
            echo "Sliding window starts: ${starting_frames[@]}"

            # Run inverse rendering with chunk_mode all
            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
                --checkpoint_dir "$CHECKPOINT_DIR" --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
                --dataset_path=tmp_${path_name}_${scene} \
                --num_video_frames $chunk_size \
                --group_mode folder \
                --overlap_n_frames $overlap_n_frames \
                --chunk_mode all \
                --video_save_folder=output_tmp_${path_name}_${scene} \
                --normalize_normal True \
                --resize_resolution $height $width \
                --height $height \
                --width $width \
                --offload_diffusion_transformer --offload_tokenizer

            # Move output frames from each chunk (with batch_id suffix for later averaging)
            for ((i=0; i<${#starting_frames[@]}; i++)); do
                echo "Processing chunk $i for camera $camera"
                start_frame=${starting_frames[$i]}
                end_frame=$((start_frame + chunk_size - 1))
                if (( end_frame >= use_num_frames )); then
                    end_frame=$((use_num_frames - 1))
                fi
                batch_id=$(printf "%04d" $i)

                # frame_idx is 0-indexed here (matching the file naming from inference)
                for frame_idx in $(seq $start_frame $end_frame); do
                    gbuffer_frame=$((frame_idx - start_frame))
                    gbuffer_frame_id=$(printf "%04d" $gbuffer_frame)
                    out_name=$(printf "%03d_%s" $frame_idx $cam_idx)

                    mv "output_tmp_${path_name}_${scene}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.normal.jpg" "$save_normal_dir/${out_name}_${batch_id}.jpg"
                    mv "output_tmp_${path_name}_${scene}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.depth.jpg" "$save_depth_dir/${out_name}_${batch_id}.jpg"
                    mv "output_tmp_${path_name}_${scene}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.basecolor.jpg" "$save_albedo_dir/${out_name}_${batch_id}.jpg"
                    mv "output_tmp_${path_name}_${scene}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.roughness.jpg" "$save_roughness_dir/${out_name}_${batch_id}.jpg"
                    mv "output_tmp_${path_name}_${scene}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.metallic.jpg" "$save_metallic_dir/${out_name}_${batch_id}.jpg"
                done
            done

            # Average overlapping frames
            echo "Averaging overlapping frames for camera $camera"
            for frame_idx in $(seq 0 $((use_num_frames - 1))); do
                out_name=$(printf "%03d_%s" $frame_idx $cam_idx)

                python average_images.py "$save_normal_dir/${out_name}.jpg" $save_normal_dir/${out_name}_*.jpg
                python average_images.py "$save_depth_dir/${out_name}.jpg" $save_depth_dir/${out_name}_*.jpg
                python average_images.py "$save_albedo_dir/${out_name}.jpg" $save_albedo_dir/${out_name}_*.jpg
                python average_images.py "$save_roughness_dir/${out_name}.jpg" $save_roughness_dir/${out_name}_*.jpg
                python average_images.py "$save_metallic_dir/${out_name}.jpg" $save_metallic_dir/${out_name}_*.jpg

                # Remove the batch-specific files
                rm $save_normal_dir/${out_name}_*.jpg
                rm $save_depth_dir/${out_name}_*.jpg
                rm $save_albedo_dir/${out_name}_*.jpg
                rm $save_roughness_dir/${out_name}_*.jpg
                rm $save_metallic_dir/${out_name}_*.jpg
            done
        fi

        # Cleanup
        rm -r ./tmp_${path_name}_${scene}
        rm -r ./output_tmp_${path_name}_${scene}

        echo "Finished processing camera: $camera"
    done

    # Resize images to original resolution (1920x1280)
    echo "Resizing images to 1920x1280"
    python resize_image.py $save_normal_dir 1920 1280
    python resize_image.py $save_depth_dir 1920 1280
    python resize_image.py $save_albedo_dir 1920 1280
    python resize_image.py $save_roughness_dir 1920 1280
    python resize_image.py $save_metallic_dir 1920 1280

    echo "Finished processing scene: $scene"
done

echo "=========================================="
echo "All processing completed successfully!"
echo "=========================================="
