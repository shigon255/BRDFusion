exp_name="omnire_327"
data_root="/project2/yi-ray/BRDFusion/outputs/${exp_name}"

# camera.json and gaussian.ply need to be under the data_root

python render_ply.py \
    --config-name apps/colmap_3dgrt.yaml \
    path="${data_root}" \
    out_dir="runs/eval/${exp_name}" \
    experiment_name="${exp_name}" \
    dataset.downsample_factor=2 \
    import_ply.path="${data_root}/gaussian.ply" \
    dataset.type="camtraj"

# python render_ply.py \
#     --config-name apps/colmap_3dgrt.yaml \
#     path=/project2/yi-ray/BRDFusion/data/examples/360_v2/garden \
#     out_dir="runs/eval/${exp_name}" \
#     experiment_name="${exp_name}" \
#     dataset.downsample_factor=2 \
#     import_ply.path="/project2/yi-ray/BRDFusion/data/examples/gaussian-splatting/output/9cb19488-b/point_cloud/iteration_30000/point_cloud.ply" \