scene_idx=114

seed=1000

# GS intrinsic x DL / optimized envmap x all SDEdit

# cases=("dlenvmap" "optenvmap")
# prefixs=("dl" "")
cases=("strongrotate2")
prefixs=("strongrotate2")

for case_id in 0; do
    case=${cases[$case_id]}
    prefix=${prefixs[$case_id]}

    mkdir -p ./drivestudio_exp/$scene_idx/${case}

    for id in 0 1 2; do
        for strength in 0 1 2 3 4 5 6 7 8 9 10; do
            sdedit_strength=$(echo "scale=1; $strength/10" | bc)
            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python sdedit_forward_renderer.py \
                --input_rgb_video ./drivestudio_exp/$scene_idx/${prefix}pbr_rgb_${id}.mp4 \
                --basecolor_video ./drivestudio_exp/$scene_idx/${prefix}albedo_${id}.mp4 \
                --normal_video ./drivestudio_exp/$scene_idx/${prefix}normal_${id}.mp4 \
                --depth_video ./drivestudio_exp/$scene_idx/${prefix}normalized_depth_${id}.mp4 \
                --roughness_video ./drivestudio_exp/$scene_idx/${prefix}roughness_${id}.mp4 \
                --metallic_video ./drivestudio_exp/$scene_idx/${prefix}metallic_${id}.mp4 \
                --env_map ./drivestudio_exp/$scene_idx/${prefix}envmap_${id}.hdr \
                --output_video ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_relit_cam${id}_seed${seed}.mp4 \
                --sdedit_strength $sdedit_strength \
                --rotate_light \
                --guidance 0 \
                --num_steps 15 \
                --height 704 \
                --width 1280 \
                --max_frames 57 \
                --seed $seed \
                --offload_diffusion_transformer \
                --offload_tokenizer

            # resize to 960x640
            python resize_vid.py \
                --input ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_relit_cam${id}_seed${seed}.mp4 \
                --output ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_relit_cam${id}_seed${seed}_960x640.mp4 \
                --width 960 \
                --height 640
        done

        # combine different strength videos and GT video into one video
        python /project2/yi-ray/BRDFusion/tools/video/stack_video_cv.py \
            --videos \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w0_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.1_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.2_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.3_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.4_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.5_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.6_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.7_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.8_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w.9_relit_cam${id}_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${prefix}pbr_rgb_${id}.mp4 \
            ./drivestudio_exp/$scene_idx/gt_rgb_${id}.mp4 \
            --titles \
            "SDEdit w=0.0" \
            "SDEdit w=0.1" \
            "SDEdit w=0.2" \
            "SDEdit w=0.3" \
            "SDEdit w=0.4" \
            "SDEdit w=0.5" \
            "SDEdit w=0.6" \
            "SDEdit w=0.7" \
            "SDEdit w=0.8" \
            "SDEdit w=0.9" \
            "PBR" \
            "GT" \
            --output \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_all_strengths_cam${id}_seed${seed}.mp4 \
            --columns 4 \
            --max_frames 57

    done

    # merge all to make it [1 0 2] video for each weight
    for strength in 0 1 2 3 4 5 6 7 8 9 10; do
        sdedit_strength=$(echo "scale=1; $strength/10" | bc)
        python merge.py \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_relit_cam1_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_relit_cam0_seed${seed}_960x640.mp4 \
            ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_relit_cam2_seed${seed}_960x640.mp4 \
            --output ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_relit_seed${seed}_960x640.mp4
    done

    python /project2/yi-ray/BRDFusion/tools/video/stack_video_cv.py \
        --videos \
        ./drivestudio_exp/$scene_idx/${case}/sdedit_w0_relit_seed${seed}_960x640.mp4 \
        ./drivestudio_exp/$scene_idx/${case}/sdedit_w.4_relit_seed${seed}_960x640.mp4 \
        ./drivestudio_exp/$scene_idx/${case}/sdedit_w.7_relit_seed${seed}_960x640.mp4 \
        ./drivestudio_exp/$scene_idx/${case}/sdedit_w1.0_relit_seed${seed}_960x640.mp4 \
        ./drivestudio_exp/$scene_idx/gt_rgb.mp4 \
        --titles \
        "SDEdit w=0.0" \
        "SDEdit w=0.4" \
        "SDEdit w=0.7" \
        "SDEdit w=1.0" \
        "GT" \
        --output ./drivestudio_exp/$scene_idx/${case}/sdedit_all_strengths_merged_seed${seed}_960x640.mp4 \
        --columns 1 \
        --max_frames 57
done

