dataset_id=19
dataset_id=$(printf "%03d" $dataset_id)  # Ensure dataset_id is zero-padded to 3 digits
dataset_prefix=/project2/yi-ray/BRDFusion/data/waymo/processed/training/$dataset_id/
chunk_size=57
num_timestep=198

# Calculate the largest multiple of chunk_size greater than num_timestep
max_frame=$(( ( (num_timestep + chunk_size - 1) / chunk_size ) * chunk_size ))
starting_frames=()
for ((i=0; i+chunk_size<max_frame; i+=chunk_size)); do
    starting_frames+=($i)
done
starting_frames+=($((num_timestep - chunk_size)))

set -e


echo "Processing dataset with ID: $dataset_id"

mkdir -p $dataset_prefix/diffusion_renderer_normal
mkdir -p $dataset_prefix/diffusion_renderer_depth
mkdir -p $dataset_prefix/diffusion_renderer_albedo
mkdir -p $dataset_prefix/diffusion_renderer_roughness
mkdir -p $dataset_prefix/diffusion_renderer_metallic

python /project2/yi-ray/BRDFusion/tools/video/stack_drive_dataset_img.py \
    --input_folder $dataset_prefix/images \
    --output_folder stack_tmp \
    --camera_order 1 0 2 \
    --scale 1.0

mkdir -p tmp_diffusion_renderer_normal
mkdir -p tmp_diffusion_renderer_depth
mkdir -p tmp_diffusion_renderer_albedo
mkdir -p tmp_diffusion_renderer_roughness
mkdir -p tmp_diffusion_renderer_metallic

for start_frame in "${starting_frames[@]}"; do
    
    end_frame=$((start_frame + chunk_size - 1))
    echo "Handling frames from $start_frame to $end_frame"

    mkdir -p ./tmp

    for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
        cp "stack_tmp/${frame_id}.jpg" ./tmp/
    done

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
        --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
        --dataset_path=tmp \
        --num_video_frames $chunk_size \
        --group_mode folder \
        --video_save_folder=output_tmp \
        --normalize_normal True \

    for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
        gbuffer_frame_id=$(printf "%04d\n" $((10#$frame_id - $start_frame)))
        mv output_tmp/gbuffer_frames/0000.$gbuffer_frame_id.normal.jpg tmp_diffusion_renderer_normal/$frame_id.jpg
        mv output_tmp/gbuffer_frames/0000.$gbuffer_frame_id.depth.jpg tmp_diffusion_renderer_depth/$frame_id.jpg
        mv output_tmp/gbuffer_frames/0000.$gbuffer_frame_id.basecolor.jpg tmp_diffusion_renderer_albedo/$frame_id.jpg
        mv output_tmp/gbuffer_frames/0000.$gbuffer_frame_id.roughness.jpg tmp_diffusion_renderer_roughness/$frame_id.jpg
        mv output_tmp/gbuffer_frames/0000.$gbuffer_frame_id.metallic.jpg tmp_diffusion_renderer_metallic/$frame_id.jpg
    done

    # Clean up temporary directories
    rm -r ./tmp
    rm -r ./output_tmp

done

python /project2/yi-ray/BRDFusion/tools/video/split_drive_dataset_img.py \
    --input_folder tmp_diffusion_renderer_normal \
    --output_folder $dataset_prefix/diffusion_renderer_normal \
    --camera_order 1 0 2 \

python /project2/yi-ray/BRDFusion/tools/video/split_drive_dataset_img.py \
    --input_folder tmp_diffusion_renderer_depth \
    --output_folder $dataset_prefix/diffusion_renderer_depth \
    --camera_order 1 0 2 \

python /project2/yi-ray/BRDFusion/tools/video/split_drive_dataset_img.py \
    --input_folder tmp_diffusion_renderer_albedo \
    --output_folder $dataset_prefix/diffusion_renderer_albedo \
    --camera_order 1 0 2 

python /project2/yi-ray/BRDFusion/tools/video/split_drive_dataset_img.py \
    --input_folder tmp_diffusion_renderer_roughness \
    --output_folder $dataset_prefix/diffusion_renderer_roughness \
    --camera_order 1 0 2

python /project2/yi-ray/BRDFusion/tools/video/split_drive_dataset_img.py \
    --input_folder tmp_diffusion_renderer_metallic \
    --output_folder $dataset_prefix/diffusion_renderer_metallic \
    --camera_order 1 0 2    

echo "Resizing images to 1920x1280"
python resize_image.py $dataset_prefix/diffusion_renderer_normal 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_depth 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_albedo 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_roughness 1920 1280
python resize_image.py $dataset_prefix/diffusion_renderer_metallic 1920 1280


rm -r ./tmp_diffusion_renderer_normal
rm -r ./tmp_diffusion_renderer_depth
rm -r ./tmp_diffusion_renderer_albedo
rm -r ./tmp_diffusion_renderer_roughness
rm -r ./tmp_diffusion_renderer_metallic

rm -r stack_tmp

echo "All processing completed successfully for dataset ID: $dataset_id"