# Assume that ckpt lives in project_root/ckpt
# Comment out the ckpt you haven't downloaded yet

waymo_ckpts_to_stage=(
    "003"
    "019"
    "114"
    "172"
    "703"
)

self_ckpts_to_stage=(
    "1"
    "2"
    "3"
    "4"
    "5"
    "6"
)

for waymo_ckpt in "${waymo_ckpts_to_stage[@]}"; do
    # All waymo ckpts are trained under 1cam and 51 frames, with test_image_stride = 10.
    conda run -n brdfusion --no-capture-output python -u tools/stage_render_checkpoint.py \
        --checkpoint ./ckpt/waymo/${waymo_ckpt}/checkpoint_final.pth \
        --dataset waymo \
        --cams 1 \
        --scene_idx ${waymo_ckpt} \
        --num_frames 51 \
        --data_root ./data/waymo/processed/training \
        --output_root ./work_dirs \
        --symlink
done

for self_ckpt in "${self_ckpts_to_stage[@]}"; do
    # All self ckpts are trained under 1cam and 51 frames, with test_image_stride = 10.
    conda run -n brdfusion --no-capture-output python -u tools/stage_render_checkpoint.py \
        --checkpoint ./ckpt/self/path${self_ckpt}_fixed_tree_gamma_full/qwantani_moon_noon_puresky_4k/checkpoint_final.pth \
        --dataset self \
        --cams 1 \
        --path_id ${self_ckpt} \
        --scene_idx qwantani_moon_noon_puresky_4k \
        --num_frames 51 \
        --symlink
done