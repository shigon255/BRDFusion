#!/bin/bash
#
# Merge all overlapping video chunks by averaging frames.
#
# Configure the COMBINATIONS array below to select which (weight, cam) pairs to process.
#

set -e

# ============================================================================
# CONFIGURATION: Edit this array to select which (weight, cam) pairs to merge
# Format: "weight,cam"
# ============================================================================
COMBINATIONS=(                                                                                                                                                                                                           
    # ".2,0"   
    ".2,1"
    ".2,2"
    ".3,0"
    ".3,1"                                                                                                                                                                                                            
    # ".4,0"                                                                                                                                                                                                               
    # ".4,1"                                                                                                                                                                                                               
    # ".4,2"                                                                                                                                                                                                               
    # ".5,0"                                                                                                                                                                                                               
    # ".5,1"                                                                                                                                                                                                               
    # ".5,2"                                                                                                                                                                                                               
    # ".6,0"                                                                                                                                                                                                               
    # ".6,1"                                                                                                                                                                                                               
    # ".7,0"                                                                                                                                                                                                               
    # ".8,0"      
    # ".6,2"                                                                                                                                                                                                         
)  

# Input/output directory
INPUT_DIR="drivestudio_exp/3/refine_intrinsics_gtrgb_full_refine_overlap50"
OUTPUT_DIR="$INPUT_DIR"

# Seed (usually fixed)
SEED="1000"

# Buffer types to process
BUFFER_TYPES=("basecolor" "normal" "depth" "roughness" "metallic")

# Video parameters
CHUNK_SIZE=57
STEP=7
FPS=24

# ============================================================================
# Processing logic (no need to edit below)
# ============================================================================

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Processing ${#COMBINATIONS[@]} combination(s)"
echo ""

for combo in "${COMBINATIONS[@]}"; do
    # Parse weight and camera from combo
    IFS=',' read -r weight camera <<< "$combo"

    echo "=========================================="
    echo "Processing: weight=$weight, camera=$camera, seed=$SEED"
    echo "=========================================="

    # Check if source folders exist
    pattern="sdedit_w${weight}_cam${camera}_seed${SEED}_start*"
    matching_folders=$(ls -d "$INPUT_DIR"/$pattern 2>/dev/null | wc -l)
    if [ "$matching_folders" -eq 0 ]; then
        echo "WARNING: No folders found matching $pattern, skipping..."
        continue
    fi
    echo "Found $matching_folders source chunks"

    # Check if output already exists
    output_folder="$OUTPUT_DIR/sdedit_w${weight}_cam${camera}_seed${SEED}"
    if [ -d "$output_folder" ]; then
        # Check if all buffer types exist
        all_exist=true
        for buffer in "${BUFFER_TYPES[@]}"; do
            if ! ls "$output_folder"/*"$buffer".mp4 >/dev/null 2>&1; then
                all_exist=false
                break
            fi
        done
        if [ "$all_exist" = true ]; then
            echo "Output already exists, skipping: $output_folder"
            continue
        fi
    fi

    # Process each buffer type
    for buffer in "${BUFFER_TYPES[@]}"; do
        echo "  Processing buffer: $buffer"
        python merge_overlap_videos.py \
            "$INPUT_DIR" \
            "$OUTPUT_DIR" \
            "$weight" \
            "$camera" \
            "$SEED" \
            "$buffer" \
            --chunk_size $CHUNK_SIZE \
            --step $STEP \
            --fps $FPS
    done

    echo "Completed: weight=$weight, camera=$camera"
    echo ""
done

echo "All merging completed!"
