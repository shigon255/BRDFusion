#!/bin/bash
# Convert videos to gbuffer_frames format for forward renderer

# Default configuration
scene_idx=114
INPUT_DIR="/project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/asset/waymo_3_invw0/video_delighting/"
OUTPUT_DIR="/project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/asset/waymo_3_invw0/video_delighting/gbuffer_frames"
ALBEDO_VIDEO="albedo.mp4"
NORMAL_VIDEO="normal.mp4"
DEPTH_VIDEO="normalized_depth.mp4"
ROUGHNESS_VIDEO="roughness.mp4"
METALLIC_VIDEO="metallic.mp4"

# Run the conversion script
python convert_videos_to_gbuffer.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --albedo_video "$ALBEDO_VIDEO" \
    --normal_video "$NORMAL_VIDEO" \
    --depth_video "$DEPTH_VIDEO" \
    --roughness_video "$ROUGHNESS_VIDEO" \
    --metallic_video "$METALLIC_VIDEO"

echo ""
echo "Conversion complete! Frames saved to: $OUTPUT_DIR"
