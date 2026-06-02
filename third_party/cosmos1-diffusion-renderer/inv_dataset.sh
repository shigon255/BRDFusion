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



# dataset_id=157
# num_timestep=196

# Read command-line arguments
dataset_id=$1          # First argument
num_timestep=$2        # Second argument

if [ -z "$dataset_id" ] || [ -z "$num_timestep" ]; then
    echo "Usage: $0 <dataset_id> <num_timestep>"
    exit 1
fi

# NOTE: it only works when num_timesteps > overlap_n_frames

dataset_id=$(printf "%03d" $dataset_id)  # Ensure dataset_id is zero-padded to 3 digits
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRDFUSION_ROOT="${BRDFUSION_ROOT:-$(cd "$SCRIPT_ROOT/../.." && pwd)}"
dataset_prefix="${DATASET_PREFIX:-${BRDFUSION_ROOT}/data/waymo/processed/training/$dataset_id/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_ROOT}/checkpoints}"
chunk_size=57
overlap_n_frames=50
width=1280
height=704

set -e

starting_frames=($(sliding_window_starts $num_timestep $chunk_size $overlap_n_frames))

# postfix="_overlap${overlap_n_frames}"
postfix=""

save_normal_dir=$dataset_prefix/diffusion_renderer_normal$postfix
save_depth_dir=$dataset_prefix/diffusion_renderer_depth$postfix
save_albedo_dir=$dataset_prefix/diffusion_renderer_albedo$postfix
save_roughness_dir=$dataset_prefix/diffusion_renderer_roughness$postfix
save_metallic_dir=$dataset_prefix/diffusion_renderer_metallic$postfix

echo "Processing dataset with ID: $dataset_id"

mkdir -p $save_normal_dir
mkdir -p $save_depth_dir
mkdir -p $save_albedo_dir
mkdir -p $save_roughness_dir
mkdir -p $save_metallic_dir


# Select cameras with CAM_IDS, for example:
#   CAM_IDS="0 1 2" bash inv_dataset.sh 003 198  # override default single-camera use
# Defaults to the front camera used by the original BRDFusion scripts.
CAM_IDS="${CAM_IDS:-0}"
# shellcheck disable=SC2206
selected_cameras=( ${CAM_IDS} )
main_cameras=()
side_cameras=()
for camera in "${selected_cameras[@]}"; do
    case "$camera" in
        0|1|2) main_cameras+=("$camera") ;;
        3|4) side_cameras+=("$camera") ;;
        *)
            echo "Error: unsupported Waymo camera '$camera'. Use CAM_IDS with ids 0 1 2 3 4." >&2
            exit 1
            ;;
    esac
done

echo "Selected cameras: ${selected_cameras[*]}"
if ((${#main_cameras[@]} > 0)); then
    echo "Main/front cameras: ${main_cameras[*]}"
fi
if ((${#side_cameras[@]} > 0)); then
    echo "Side cameras: ${side_cameras[*]}"
fi

for camera in "${main_cameras[@]}"; do
    echo "Processing camera $camera"
    
    mkdir -p ./tmp_${dataset_id}
    for frame_id in $(seq -f "%03g" 0 $((num_timestep-1))); do
        cp "$dataset_prefix/images/${frame_id}_$camera.jpg" ./tmp_${dataset_id}/
    done

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
        --checkpoint_dir "$CHECKPOINT_DIR" --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
        --dataset_path=tmp_${dataset_id} \
        --num_video_frames $chunk_size \
        --group_mode folder \
        --overlap_n_frames $overlap_n_frames \
        --chunk_mode all \
        --video_save_folder=output_tmp_${dataset_id} \
        --normalize_normal True \
        --resize_resolution $height $width \
        --height $height \
        --width $width \
        --offload_diffusion_transformer --offload_tokenizer \


    for ((i=0; i<${#starting_frames[@]}; i++)); do
        echo "Processing chunk $i for camera $camera"
        start_frame=${starting_frames[$i]}
        end_frame=$((start_frame + chunk_size - 1))
        if (( end_frame >= num_timestep )); then
            end_frame=$((num_timestep - 1))
        fi
        batch_id=$(printf "%04d" $i)

        for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
            gbuffer_frame=$((10#$frame_id - $start_frame))

            gbuffer_frame_id=$(printf "%04d\n" $gbuffer_frame)
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.normal.jpg $save_normal_dir/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.depth.jpg $save_depth_dir/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.basecolor.jpg $save_albedo_dir/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.roughness.jpg $save_roughness_dir/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.metallic.jpg $save_metallic_dir/${frame_id}_${camera}_${batch_id}.jpg
        done

    done

    rm -r ./tmp_${dataset_id}
    rm -r ./output_tmp_${dataset_id}

    for frame_id in $(seq -f "%03g" 0 $((num_timestep-1))); do
        python average_images.py $save_normal_dir/${frame_id}_${camera}.jpg $save_normal_dir/${frame_id}_${camera}_*.jpg
        python average_images.py $save_depth_dir/${frame_id}_${camera}.jpg $save_depth_dir/${frame_id}_${camera}_*.jpg
        python average_images.py $save_albedo_dir/${frame_id}_${camera}.jpg $save_albedo_dir/${frame_id}_${camera}_*.jpg
        python average_images.py $save_roughness_dir/${frame_id}_${camera}.jpg $save_roughness_dir/${frame_id}_${camera}_*.jpg
        python average_images.py $save_metallic_dir/${frame_id}_${camera}.jpg $save_metallic_dir/${frame_id}_${camera}_*.jpg

        rm $save_normal_dir/${frame_id}_${camera}_*.jpg
        rm $save_depth_dir/${frame_id}_${camera}_*.jpg
        rm $save_albedo_dir/${frame_id}_${camera}_*.jpg
        rm $save_roughness_dir/${frame_id}_${camera}_*.jpg
        rm $save_metallic_dir/${frame_id}_${camera}_*.jpg
    done

done


if ((${#main_cameras[@]} > 0)); then
    echo "Resizing selected main/front camera images to 1920x1280"
    for camera in "${main_cameras[@]}"; do
        python resize_image.py $save_normal_dir 1920 1280 --pattern "*_${camera}.jpg"
        python resize_image.py $save_depth_dir 1920 1280 --pattern "*_${camera}.jpg"
        python resize_image.py $save_albedo_dir 1920 1280 --pattern "*_${camera}.jpg"
        python resize_image.py $save_roughness_dir 1920 1280 --pattern "*_${camera}.jpg"
        python resize_image.py $save_metallic_dir 1920 1280 --pattern "*_${camera}.jpg"
    done
fi


if ((${#side_cameras[@]} > 0)); then
    echo "Processing side cameras with 1920x886 resize: ${side_cameras[*]}"
    mkdir -p ./tmp_diffusion_renderer_normal_${dataset_id}
    mkdir -p ./tmp_diffusion_renderer_depth_${dataset_id}
    mkdir -p ./tmp_diffusion_renderer_albedo_${dataset_id}
    mkdir -p ./tmp_diffusion_renderer_roughness_${dataset_id}
    mkdir -p ./tmp_diffusion_renderer_metallic_${dataset_id}

cameras=("${side_cameras[@]}")
for camera in "${cameras[@]}"; do
    echo "Processing camera $camera"
    
    mkdir -p ./tmp_${dataset_id}
    for frame_id in $(seq -f "%03g" 0 $((num_timestep-1))); do
        cp "$dataset_prefix/images/${frame_id}_$camera.jpg" ./tmp_${dataset_id}/
    done

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
        --checkpoint_dir "$CHECKPOINT_DIR" --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
        --dataset_path=tmp_${dataset_id} \
        --num_video_frames $chunk_size \
        --group_mode folder \
        --overlap_n_frames $overlap_n_frames \
        --chunk_mode all \
        --video_save_folder=output_tmp_${dataset_id} \
        --normalize_normal True \
        --resize_resolution $height $width \
        --height $height \
        --width $width \
        --offload_diffusion_transformer --offload_tokenizer \

    for ((i=0; i<${#starting_frames[@]}; i++)); do
        start_frame=${starting_frames[$i]}
        end_frame=$((start_frame + chunk_size - 1))
        if (( end_frame >= num_timestep )); then
            end_frame=$((num_timestep - 1))
        fi
        batch_id=$(printf "%04d" $i)

        for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
            gbuffer_frame=$((10#$frame_id - $start_frame))

            gbuffer_frame_id=$(printf "%04d\n" $gbuffer_frame)
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.normal.jpg ./tmp_diffusion_renderer_normal_${dataset_id}/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.depth.jpg ./tmp_diffusion_renderer_depth_${dataset_id}/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.basecolor.jpg ./tmp_diffusion_renderer_albedo_${dataset_id}/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.roughness.jpg ./tmp_diffusion_renderer_roughness_${dataset_id}/${frame_id}_${camera}_${batch_id}.jpg
            mv output_tmp_${dataset_id}/gbuffer_frames/${batch_id}.${gbuffer_frame_id}.metallic.jpg ./tmp_diffusion_renderer_metallic_${dataset_id}/${frame_id}_${camera}_${batch_id}.jpg
        done
    done

    rm -r ./tmp_${dataset_id}
    rm -r ./output_tmp_${dataset_id}

    for frame_id in $(seq -f "%03g" 0 $((num_timestep-1))); do
        python average_images.py ./tmp_diffusion_renderer_normal_${dataset_id}/${frame_id}_${camera}.jpg ./tmp_diffusion_renderer_normal_${dataset_id}/${frame_id}_${camera}_*.jpg
        python average_images.py ./tmp_diffusion_renderer_depth_${dataset_id}/${frame_id}_${camera}.jpg ./tmp_diffusion_renderer_depth_${dataset_id}/${frame_id}_${camera}_*.jpg
        python average_images.py ./tmp_diffusion_renderer_albedo_${dataset_id}/${frame_id}_${camera}.jpg ./tmp_diffusion_renderer_albedo_${dataset_id}/${frame_id}_${camera}_*.jpg
        python average_images.py ./tmp_diffusion_renderer_roughness_${dataset_id}/${frame_id}_${camera}.jpg ./tmp_diffusion_renderer_roughness_${dataset_id}/${frame_id}_${camera}_*.jpg
        python average_images.py ./tmp_diffusion_renderer_metallic_${dataset_id}/${frame_id}_${camera}.jpg ./tmp_diffusion_renderer_metallic_${dataset_id}/${frame_id}_${camera}_*.jpg

        rm ./tmp_diffusion_renderer_normal_${dataset_id}/${frame_id}_${camera}_*.jpg
        rm ./tmp_diffusion_renderer_depth_${dataset_id}/${frame_id}_${camera}_*.jpg
        rm ./tmp_diffusion_renderer_albedo_${dataset_id}/${frame_id}_${camera}_*.jpg
        rm ./tmp_diffusion_renderer_roughness_${dataset_id}/${frame_id}_${camera}_*.jpg
        rm ./tmp_diffusion_renderer_metallic_${dataset_id}/${frame_id}_${camera}_*.jpg
    done
done

echo "Resizing images to 1920x886"
python resize_image.py ./tmp_diffusion_renderer_normal_${dataset_id} 1920 886
python resize_image.py ./tmp_diffusion_renderer_depth_${dataset_id} 1920 886
python resize_image.py ./tmp_diffusion_renderer_albedo_${dataset_id} 1920 886
python resize_image.py ./tmp_diffusion_renderer_roughness_${dataset_id} 1920 886
python resize_image.py ./tmp_diffusion_renderer_metallic_${dataset_id} 1920 886

# Move the resized images to the final directories
mv ./tmp_diffusion_renderer_normal_${dataset_id}/* $save_normal_dir
mv ./tmp_diffusion_renderer_depth_${dataset_id}/* $save_depth_dir
mv ./tmp_diffusion_renderer_albedo_${dataset_id}/* $save_albedo_dir
mv ./tmp_diffusion_renderer_roughness_${dataset_id}/* $save_roughness_dir
mv ./tmp_diffusion_renderer_metallic_${dataset_id}/* $save_metallic_dir

# Clean up temporary directories
rm -r ./tmp_diffusion_renderer_normal_${dataset_id}
rm -r ./tmp_diffusion_renderer_depth_${dataset_id}
rm -r ./tmp_diffusion_renderer_albedo_${dataset_id}
rm -r ./tmp_diffusion_renderer_roughness_${dataset_id}
rm -r ./tmp_diffusion_renderer_metallic_${dataset_id}
fi


echo "All processing completed successfully for dataset ID: $dataset_id"
