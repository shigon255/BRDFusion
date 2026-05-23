#!/bin/bash

scene_idx=3
root=./drivestudio_exp/$scene_idx

# Set paths
INPUT_RGB="${root}/pbr_rgb_0.mp4"
BASECOLOR_VIDEO="${root}/albedo_0.mp4"
ROUGHNESS_VIDEO="${root}/roughness_0.mp4"
METALLIC_VIDEO="${root}/metallic_0.mp4"
DEPTH_VIDEO="${root}/normalized_depth_0.mp4"
NORMAL_VIDEO="${root}/normal_0.mp4"

# WARNING: if the video is not center-frame, envmap need calibration
ENV_MAP="${root}/envmap.hdr"

sdedit_strength=0.0

OUTPUT_VIDEO="${root}/sdedit_w${sdedit_strength}_relit_seed500.mp4"

# Run SDEdit refinement
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python sdedit_forward_renderer.py \
    --input_rgb_video $INPUT_RGB \
    --basecolor_video $BASECOLOR_VIDEO \
    --normal_video $NORMAL_VIDEO \
    --depth_video $DEPTH_VIDEO \
    --roughness_video $ROUGHNESS_VIDEO \
    --metallic_video $METALLIC_VIDEO \
    --env_map $ENV_MAP \
    --output_video $OUTPUT_VIDEO \
    --sdedit_strength $sdedit_strength \
    --guidance 0 \
    --num_steps 15 \
    --height 704 \
    --width 1280 \
    --max_frames 57 \
    --seed 500 \
    --offload_diffusion_transformer \
    --offload_tokenizer