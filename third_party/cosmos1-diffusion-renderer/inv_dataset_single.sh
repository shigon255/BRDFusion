dataset_id=36
dataset_id=$(printf "%03d" $dataset_id)  # Ensure dataset_id is zero-padded to 3 digits
dataset_prefix=/project2/yi-ray/BRDFusion/data/waymo/processed/training/$dataset_id/
chunk_size=57
num_timestep=198

# # Calculate the largest multiple of chunk_size greater than num_timestep
# max_frame=$(( ( (num_timestep + chunk_size - 1) / chunk_size ) * chunk_size ))
# starting_frames=()
# for ((i=chunk_size; i<max_frame; i+=chunk_size)); do
#     starting_frames+=($i)
# done
# starting_frames+=($((num_timestep - chunk_size)))

set -e

echo "Processing dataset with ID: $dataset_id"

mkdir -p $dataset_prefix/diffusion_renderer_normal
mkdir -p $dataset_prefix/diffusion_renderer_depth
mkdir -p $dataset_prefix/diffusion_renderer_albedo
mkdir -p $dataset_prefix/diffusion_renderer_roughness
mkdir -p $dataset_prefix/diffusion_renderer_metallic


cameras=(0 1 2)
for camera in "${cameras[@]}"; do
    echo "Processing camera $camera"
    
    start_frame=0
    end_frame=$((num_timestep - 1))
    echo "Handling frames from $start_frame to $end_frame"

    mkdir -p ./tmp

    for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
        cp "$dataset_prefix/images/${frame_id}_$camera.jpg" ./tmp/
    done

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
        --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
        --dataset_path=tmp --num_video_frames 1 --group_mode webdataset \
        --video_save_folder=output_tmp --save_video=False \
        --normalize_normal True \

    # CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    #     --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    #     --dataset_path=tmp \
    #     --num_video_frames $chunk_size \
    #     --group_mode folder \
    #     --video_save_folder=output_tmp

    for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
        gbuffer_frame=$((10#$frame_id - $start_frame))
        gbuffer_frame_id=$(printf "%03d\n" $gbuffer_frame)
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.normal.jpg $dataset_prefix/diffusion_renderer_normal/${frame_id}_$camera.jpg
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.depth.jpg $dataset_prefix/diffusion_renderer_depth/${frame_id}_$camera.jpg 
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.basecolor.jpg $dataset_prefix/diffusion_renderer_albedo/${frame_id}_$camera.jpg
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.roughness.jpg $dataset_prefix/diffusion_renderer_roughness/${frame_id}_$camera.jpg
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.metallic.jpg $dataset_prefix/diffusion_renderer_metallic/${frame_id}_$camera.jpg
    done

    rm -r ./tmp
    rm -r ./output_tmp

done

echo "Resizing images to 1920x1280"
python resize_image.py $dataset_prefix/diffusion_renderer_normal 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_depth 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_albedo 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_roughness 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_metallic 1920 1280


# for camera 3 and 4
mkdir ./tmp_diffusion_renderer_normal
mkdir ./tmp_diffusion_renderer_depth
mkdir ./tmp_diffusion_renderer_albedo
mkdir ./tmp_diffusion_renderer_roughness
mkdir ./tmp_diffusion_renderer_metallic

cameras=(3 4)
for camera in "${cameras[@]}"; do
    echo "Processing camera $camera"
    
    start_frame=0
    end_frame=$((num_timestep - 1))
    echo "Handling frames from $start_frame to $end_frame"

    mkdir -p ./tmp

    for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
        cp "$dataset_prefix/images/${frame_id}_$camera.jpg" ./tmp/
    done

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=tmp --num_video_frames 1 --group_mode webdataset \
    --video_save_folder=output_tmp --save_video=False \
    --normalize_normal True \

    # CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    #     --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    #     --dataset_path=tmp \
    #     --num_video_frames $chunk_size \
    #     --group_mode folder \
    #     --video_save_folder=output_tmp

    for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
        gbuffer_frame=$((10#$frame_id - $start_frame))
        gbuffer_frame_id=$(printf "%03d\n" $((10#$frame_id - $start_frame)))
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.normal.jpg ./tmp_diffusion_renderer_normal/${frame_id}_$camera.jpg
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.depth.jpg ./tmp_diffusion_renderer_depth/${frame_id}_$camera.jpg 
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.basecolor.jpg ./tmp_diffusion_renderer_albedo/${frame_id}_$camera.jpg
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.roughness.jpg ./tmp_diffusion_renderer_roughness/${frame_id}_$camera.jpg
        mv output_tmp/gbuffer_frames/${gbuffer_frame_id}_${camera}/0000.0000.metallic.jpg ./tmp_diffusion_renderer_metallic/${frame_id}_$camera.jpg
    done

    rm -r ./tmp
    rm -r ./output_tmp

done

echo "Resizing images to 1920x886"
python resize_image.py ./tmp_diffusion_renderer_normal 1920 886
python resize_image.py ./tmp_diffusion_renderer_depth 1920 886
python resize_image.py ./tmp_diffusion_renderer_albedo 1920 886
python resize_image.py ./tmp_diffusion_renderer_roughness 1920 886
python resize_image.py ./tmp_diffusion_renderer_metallic 1920 886

# Move the resized images to the final directories
mv ./tmp_diffusion_renderer_normal/* $dataset_prefix/diffusion_renderer_normal/
mv ./tmp_diffusion_renderer_depth/* $dataset_prefix/diffusion_renderer_depth/
mv ./tmp_diffusion_renderer_albedo/* $dataset_prefix/diffusion_renderer_albedo/
mv ./tmp_diffusion_renderer_roughness/* $dataset_prefix/diffusion_renderer_roughness/
mv ./tmp_diffusion_renderer_metallic/* $dataset_prefix/diffusion_renderer_metallic/

# Clean up temporary directories
rm -r ./tmp_diffusion_renderer_normal
rm -r ./tmp_diffusion_renderer_depth
rm -r ./tmp_diffusion_renderer_albedo
rm -r ./tmp_diffusion_renderer_roughness
rm -r ./tmp_diffusion_renderer_metallic

echo "All processing completed successfully for dataset ID: $dataset_id"