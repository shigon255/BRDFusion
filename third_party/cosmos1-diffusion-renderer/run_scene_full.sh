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

scene_idx=3
num_timesteps=198

seed=1000
chunk_size=57
overlap_n_frames=50
step=$(( chunk_size - overlap_n_frames ))

starting_frames=($(sliding_window_starts $num_timesteps $chunk_size $overlap_n_frames))

strengthes=(2)

case=forward_full_overlap${overlap_n_frames}
prefix="fullrefine5"
mkdir -p ./drivestudio_exp/$scene_idx/${case}

# Phase 1: Chunked forward SDEdit
for strength in ${strengthes[@]}; do
    # for id in 0 1 2; do
    for id in 1 2; do
        for start_frame in ${starting_frames[@]}; do
            sdedit_strength=$(echo "scale=1; $strength/10" | bc)
            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python sdedit_forward_renderer.py \
                --input_rgb_video ./drivestudio_exp/$scene_idx/${prefix}pbr_rgb_${id}.mp4 \
                --basecolor_video ./drivestudio_exp/$scene_idx/${prefix}albedo_${id}.mp4 \
                --normal_video ./drivestudio_exp/$scene_idx/${prefix}normal_${id}.mp4 \
                --depth_video ./drivestudio_exp/$scene_idx/${prefix}normalized_depth_${id}.mp4 \
                --roughness_video ./drivestudio_exp/$scene_idx/${prefix}roughness_${id}.mp4 \
                --metallic_video ./drivestudio_exp/$scene_idx/${prefix}metallic_${id}.mp4 \
                --env_map ./drivestudio_exp/$scene_idx/${prefix}envmap_${id}.hdr \
                --output_video ./drivestudio_exp/$scene_idx/${case}/sdedit_w${sdedit_strength}_cam${id}_seed${seed}_start${start_frame}/video.rgb.mp4 \
                --start_frame $start_frame \
                --max_frames $chunk_size \
                --sdedit_strength $sdedit_strength \
                --guidance 0 \
                --num_steps 15 \
                --height 704 \
                --width 1280 \
                --seed $seed \
                --offload_diffusion_transformer \
                --offload_tokenizer
        done
    done
done

# Phase 2: Merge overlapping chunks
for strength in ${strengthes[@]}; do
    sdedit_strength=$(echo "scale=1; $strength/10" | bc)
    for id in 0 1 2; do
        python merge_overlap_videos.py \
            ./drivestudio_exp/$scene_idx/${case} \
            ./drivestudio_exp/$scene_idx/${case} \
            $sdedit_strength $id $seed rgb \
            --chunk_size $chunk_size --step $step
    done
done
