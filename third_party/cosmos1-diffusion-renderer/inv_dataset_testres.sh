dataset_id=36
dataset_id=$(printf "%03d" $dataset_id)  # Ensure dataset_id is zero-padded to 3 digits
dataset_prefix=/project2/yi-ray/BRDFusion/data/waymo/processed/training/$dataset_id/
chunk_size=57
num_timestep=198


# Calculate the largest multiple of chunk_size greater than num_timestep
max_frame=$(( ( (num_timestep + chunk_size - 1) / chunk_size ) * chunk_size ))
starting_frames=()
for ((i=0; i<max_frame-chunk_size; i+=chunk_size)); do
    starting_frames+=($i)
done
starting_frames+=($((num_timestep - chunk_size)))

set -e

# heights=(704 854 640)
# widths=(1280 1280 960)
heights=(704)
widths=(1056)

# (832, 1248) will make height over 704

for i in "${!heights[@]}"; do
    height=${heights[$i]}
    width=${widths[$i]}
    
    echo "Processing dataset with width: $width and height: $height"

    postfix="_${width}x${height}"

    save_normal_dir=$dataset_prefix/diffusion_renderer_normal$postfix
    save_depth_dir=$dataset_prefix/diffusion_renderer_depth$postfix
    save_albedo_dir=$dataset_prefix/diffusion_renderer_albedo$postfix
    save_roughness_dir=$dataset_prefix/diffusion_renderer_roughness$postfix
    save_metallic_dir=$dataset_prefix/diffusion_renderer_metallic$postfix

    tmpdir=tmp$postfix
    output_tmpdir=output_tmp$postfix
    tmp_diffusion_renderer_normal_dir=tmp_diffusion_renderer_normal$postfix
    tmp_diffusion_renderer_depth_dir=tmp_diffusion_renderer_depth$postfix
    tmp_diffusion_renderer_albedo_dir=tmp_diffusion_renderer_albedo$postfix
    tmp_diffusion_renderer_roughness_dir=tmp_diffusion_renderer_roughness$postfix
    tmp_diffusion_renderer_metallic_dir=tmp_diffusion_renderer_metallic$postfix

    echo "Processing dataset with ID: $dataset_id"

    mkdir -p $save_normal_dir
    mkdir -p $save_depth_dir
    mkdir -p $save_albedo_dir
    mkdir -p $save_roughness_dir
    mkdir -p $save_metallic_dir


    cameras=(0 1 2)
    for camera in "${cameras[@]}"; do
        echo "Processing camera $camera"
        
        for start_frame in "${starting_frames[@]}"; do
            end_frame=$((start_frame + chunk_size - 1))
            echo "Handling frames from $start_frame to $end_frame"

            mkdir -p $tmpdir

            for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
                cp "$dataset_prefix/images/${frame_id}_$camera.jpg" $tmpdir/
            done

            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
                --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
                --dataset_path=$tmpdir \
                --num_video_frames $chunk_size \
                --group_mode folder \
                --video_save_folder=$output_tmpdir \
                --normalize_normal True \
                --offload_diffusion_transformer --offload_tokenizer \
                --resize_resolution $height $width \
                --height $height \
                --width $width \

            for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
                gbuffer_frame=$((10#$frame_id - $start_frame))

                gbuffer_frame_id=$(printf "%04d\n" $gbuffer_frame)
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.normal.jpg $save_normal_dir/${frame_id}_$camera.jpg
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.depth.jpg $save_depth_dir/${frame_id}_$camera.jpg 
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.basecolor.jpg $save_albedo_dir/${frame_id}_$camera.jpg
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.roughness.jpg $save_roughness_dir/${frame_id}_$camera.jpg
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.metallic.jpg $save_metallic_dir/${frame_id}_$camera.jpg
            done

            rm -r $tmpdir
            rm -r $output_tmpdir
        done
    done

    echo "Resizing images to 1920x1280"
    python resize_image.py $save_normal_dir 1920 1280
    python resize_image.py $save_depth_dir 1920 1280
    python resize_image.py $save_albedo_dir 1920 1280
    python resize_image.py $save_roughness_dir 1920 1280
    python resize_image.py $save_metallic_dir 1920 1280


    # for camera 3 and 4
    mkdir $tmp_diffusion_renderer_normal_dir
    mkdir $tmp_diffusion_renderer_depth_dir
    mkdir $tmp_diffusion_renderer_albedo_dir
    mkdir $tmp_diffusion_renderer_roughness_dir
    mkdir $tmp_diffusion_renderer_metallic_dir

    cameras=(3 4)
    for camera in "${cameras[@]}"; do
        echo "Processing camera $camera"
        
        for start_frame in "${starting_frames[@]}"; do
            end_frame=$((start_frame + chunk_size - 1))
            echo "Handling frames from $start_frame to $end_frame"

            mkdir -p $tmpdir

            for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
                cp "$dataset_prefix/images/${frame_id}_$camera.jpg" $tmpdir/
            done

            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
                --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
                --dataset_path=$tmpdir \
                --num_video_frames $chunk_size \
                --group_mode folder \
                --video_save_folder=$output_tmpdir \
                --normalize_normal True \
                --offload_diffusion_transformer --offload_tokenizer \
                --resize_resolution $height $width \
                --height $height \
                --width $width \

            for frame_id in $(seq -f "%03g" $start_frame $end_frame); do
                gbuffer_frame=$((10#$frame_id - $start_frame))

                gbuffer_frame_id=$(printf "%04d\n" $gbuffer_frame)
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.normal.jpg ./$tmp_diffusion_renderer_normal_dir/${frame_id}_$camera.jpg
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.depth.jpg ./$tmp_diffusion_renderer_depth_dir/${frame_id}_$camera.jpg 
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.basecolor.jpg ./$tmp_diffusion_renderer_albedo_dir/${frame_id}_$camera.jpg
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.roughness.jpg ./$tmp_diffusion_renderer_roughness_dir/${frame_id}_$camera.jpg
                mv $output_tmpdir/gbuffer_frames/0000.${gbuffer_frame_id}.metallic.jpg ./$tmp_diffusion_renderer_metallic_dir/${frame_id}_$camera.jpg
            done

            rm -r $tmpdir
            rm -r $output_tmpdir
        done
    done

    echo "Resizing images to 1920x886"
    python resize_image.py ./$tmp_diffusion_renderer_normal_dir 1920 886
    python resize_image.py ./$tmp_diffusion_renderer_depth_dir 1920 886
    python resize_image.py ./$tmp_diffusion_renderer_albedo_dir 1920 886
    python resize_image.py ./$tmp_diffusion_renderer_roughness_dir 1920 886
    python resize_image.py ./$tmp_diffusion_renderer_metallic_dir 1920 886

    # Move the resized images to the final directories
    mv $tmp_diffusion_renderer_normal_dir/* $save_normal_dir
    mv $tmp_diffusion_renderer_depth_dir/* $save_depth_dir
    mv $tmp_diffusion_renderer_albedo_dir/* $save_albedo_dir
    mv $tmp_diffusion_renderer_roughness_dir/* $save_roughness_dir
    mv $tmp_diffusion_renderer_metallic_dir/* $save_metallic_dir

    # Clean up temporary directories
    rm -r $tmp_diffusion_renderer_normal_dir
    rm -r $tmp_diffusion_renderer_depth_dir
    rm -r $tmp_diffusion_renderer_albedo_dir
    rm -r $tmp_diffusion_renderer_roughness_dir
    rm -r $tmp_diffusion_renderer_metallic_dir
done




echo "All processing completed successfully for dataset ID: $dataset_id"
