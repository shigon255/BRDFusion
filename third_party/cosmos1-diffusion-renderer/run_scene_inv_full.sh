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

scene_idx=114
num_timesteps=198

seed=1000
chunk_size=57
overlap_n_frames=50

starting_frames=($(sliding_window_starts $num_timesteps $chunk_size $overlap_n_frames))


case=refine_intrinsics_gtrgb_full_refine_overlap${overlap_n_frames}
prefix="fullrefine"
mkdir -p ./drivestudio_exp/$scene_idx/${case}

for strength in 1 2 3 4 5 6 2 3 7 8 9; do
    for id in 0 1 2; do    
        for start_frame in ${starting_frames[@]}; do
            sdedit_strength=$(echo "scale=1; $strength/10" | bc)
            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python sdedit_inverse_renderer.py \
                --input_rgb_video ./drivestudio_exp/$scene_idx/${prefix}gt_rgb_${id}.mp4 \
                --basecolor_video ./drivestudio_exp/$scene_idx/${prefix}albedo_${id}.mp4 \
                --normal_video ./drivestudio_exp/$scene_idx/${prefix}normal_${id}.mp4 \
                --depth_video ./drivestudio_exp/$scene_idx/${prefix}normalized_depth_${id}.mp4 \
                --roughness_video ./drivestudio_exp/$scene_idx/${prefix}roughness_${id}.mp4 \
                --metallic_video ./drivestudio_exp/$scene_idx/${prefix}metallic_${id}.mp4 \
                --output_dir ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_cam${id}_seed${seed}_start${start_frame} \
                --save_image False \
                --sdedit_strength $sdedit_strength \
                --start_frame $start_frame \
                --guidance 0 \
                --num_steps 15 \
                --height 704 \
                --width 1280 \
                --max_frames $chunk_size \
                --seed $seed \
                --normalize_normal True \
                --offload_diffusion_transformer \
                --offload_tokenizer
        done

    done
done



