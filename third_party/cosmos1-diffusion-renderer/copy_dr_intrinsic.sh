scene_idx=3
scene_idx3d=$(printf "%03d" $scene_idx)
dataset_root=/project2/yi-ray/BRDFusion/data/waymo/processed/training/$scene_idx3d/
output_root=./drivestudio_exp/$scene_idx
max_frames=198

intrinsics=("albedo" "roughness" "metallic" "depth" "normal")

for intrinsic in "${intrinsics[@]}"; do
    for cam_id in 0 1 2; do
        mkdir -p ./tmp
        # copy all images $dataset  _root/diffusion_renderer_${intrinsic}_{frame:03d}_0.jpg ./tmp
        # only collect images up to max_frames
        for ((i=0; i<max_frames; i++)); do
            frame_idx=$(printf "%03d" $i)
            cp $dataset_root/diffusion_renderer_${intrinsic}/${frame_idx}_${cam_id}.jpg ./tmp/
        done

        python /project2/yi-ray/BRDFusion/tools/video/imgs2vid.py --image_dir ./tmp --output_path $output_root/fulldr${intrinsic}_${cam_id}.mp4

        rm -r ./tmp
    done

done