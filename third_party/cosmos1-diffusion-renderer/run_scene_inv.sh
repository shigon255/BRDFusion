scene_idx=527

seed=1000

case=refine_intrinsics_gtrgb
prefix=""
# start_frame=10
mkdir -p ./drivestudio_exp/$scene_idx/${case}

for start_frame in 0 10 20; do
    for id in 0 1 2; do
        for strength in 0 1 2 3 4 5 6 7 8 9 10; do
            sdedit_strength=$(echo "scale=1; $strength/10" | bc)
            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python sdedit_inverse_renderer.py \
                --input_rgb_video ./drivestudio_exp/$scene_idx/gt_rgb_${id}.mp4 \
                --basecolor_video ./drivestudio_exp/$scene_idx/albedo_${id}.mp4 \
                --normal_video ./drivestudio_exp/$scene_idx/normal_${id}.mp4 \
                --depth_video ./drivestudio_exp/$scene_idx/normalized_depth_${id}.mp4 \
                --roughness_video ./drivestudio_exp/$scene_idx/roughness_${id}.mp4 \
                --metallic_video ./drivestudio_exp/$scene_idx/metallic_${id}.mp4 \
                --output_dir ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_cam${id}_seed${seed}_start${start_frame} \
                --save_image True \
                --sdedit_strength $sdedit_strength \
                --start_frame $start_frame \
                --guidance 0 \
                --num_steps 15 \
                --height 704 \
                --width 1280 \
                --max_frames 57 \
                --seed $seed \
                --normalize_normal True \
                --offload_diffusion_transformer \
                --offload_tokenizer
        done

    done
done



