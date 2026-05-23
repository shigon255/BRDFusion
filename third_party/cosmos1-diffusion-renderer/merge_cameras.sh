#!/bin/bash
#
# Merge all 3 cameras (order: 1, 0, 2) horizontally for weights with complete camera sets.
# This should be run AFTER merge_overlap_all.sh completes.
#

set -e

# ============================================================================
# CONFIGURATION: Edit this array to select which weights to merge cameras for
# Only include weights that have all 3 cameras (0, 1, 2) fully processed
# ============================================================================
WEIGHTS=(
    ".2"
)

# Input/output directory (same as merge_overlap_all.sh)
INPUT_DIR="drivestudio_exp/3/refine_intrinsics_gtrgb_full_refine_overlap50"
OUTPUT_DIR="$INPUT_DIR"

# Seed (usually fixed)
SEED="1000"

# Buffer types to process
BUFFER_TYPES=("basecolor" "normal" "depth" "roughness" "metallic")

# ============================================================================
# Processing logic (no need to edit below)
# ============================================================================

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Merging cameras for ${#WEIGHTS[@]} weight(s): ${WEIGHTS[*]}"
echo ""

for weight in "${WEIGHTS[@]}"; do
    echo "=========================================="
    echo "Processing: weight=$weight"
    echo "=========================================="

    # Check if all 3 camera folders exist
    missing_cams=()
    for cam in 0 1 2; do
        cam_folder="$INPUT_DIR/sdedit_w${weight}_cam${cam}_seed${SEED}"
        if [ ! -d "$cam_folder" ]; then
            missing_cams+=("$cam")
        fi
    done

    if [ ${#missing_cams[@]} -gt 0 ]; then
        echo "WARNING: Missing camera folders for weight=$weight: ${missing_cams[*]}"
        echo "Skipping... (run merge_overlap_all.sh first)"
        continue
    fi

    # Create output directory
    output_folder="$OUTPUT_DIR/sdedit_w${weight}_seed${SEED}"
    mkdir -p "$output_folder"

    # Process each buffer type
    for buffer in "${BUFFER_TYPES[@]}"; do
        echo "  Merging cameras for buffer: $buffer"

        # Video filenames include camera ID: fullrefinegt_rgb_{cam}.{buffer}.mp4
        cam0_folder="$INPUT_DIR/sdedit_w${weight}_cam0_seed${SEED}"
        cam1_folder="$INPUT_DIR/sdedit_w${weight}_cam1_seed${SEED}"
        cam2_folder="$INPUT_DIR/sdedit_w${weight}_cam2_seed${SEED}"

        # Find video files for each camera
        cam0_video=$(ls "$cam0_folder"/*_0."$buffer".mp4 2>/dev/null | head -1)
        cam1_video=$(ls "$cam1_folder"/*_1."$buffer".mp4 2>/dev/null | head -1)
        cam2_video=$(ls "$cam2_folder"/*_2."$buffer".mp4 2>/dev/null | head -1)

        # Check all videos exist
        if [ -z "$cam0_video" ] || [ -z "$cam1_video" ] || [ -z "$cam2_video" ]; then
            echo "    WARNING: Missing video files for $buffer, skipping..."
            echo "      cam0: $cam0_video"
            echo "      cam1: $cam1_video"
            echo "      cam2: $cam2_video"
            continue
        fi

        # Output filename (use generic name without camera ID)
        # Extract prefix from cam0 video: fullrefinegt_rgb_0.basecolor.mp4 -> fullrefinegt_rgb.basecolor.mp4
        video_basename=$(basename "$cam0_video" | sed 's/_0\./\./')
        output_video="$output_folder/$video_basename"

        # Check if output already exists
        if [ -f "$output_video" ]; then
            echo "    Output already exists, skipping: $output_video"
            continue
        fi

        # Merge cameras (order: left=1, middle=0, right=2)
        python merge.py \
            "$cam1_video" \
            "$cam0_video" \
            "$cam2_video" \
            --output "$output_video"
    done

    echo "Completed: weight=$weight -> $output_folder"
    echo ""
done

echo "All camera merging completed!"
