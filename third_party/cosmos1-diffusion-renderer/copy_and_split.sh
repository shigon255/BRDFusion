scene_idx=path1_tree_full_qwantani_moon_noon_puresky_4k

mkdir -p ./drivestudio_exp/$scene_idx

vid_root=/project2/yi-ray/BRDFusion/outputs/drivestudio/omnire_self_path1_tree_full_qwantani_moon_noon_puresky_4k_pbr_shorterseqhead/videos_eval_50k_relightrot90
step=50000

prefix="relightrot90"

# special case
pbr_rgb_path=${vid_root}/full_set_${step}_pbr_colors.mp4
gt_rgb_path=${vid_root}/full_set_${step}_gt_rgbs.mp4
normal_path=${vid_root}/full_set_${step}_normals.mp4
depth_path=${vid_root}/full_set_${step}_normalized_depths.mp4
albedo_path=${vid_root}/full_set_${step}_albedos.mp4
roughness_path=${vid_root}/full_set_${step}_roughnesses.mp4
metallic_path=${vid_root}/full_set_${step}_metallics.mp4
envmap_path=${vid_root}/envmap.hdr

cp $pbr_rgb_path ./drivestudio_exp/$scene_idx/${prefix}pbr_rgb.mp4
cp $gt_rgb_path ./drivestudio_exp/$scene_idx/${prefix}gt_rgb.mp4
cp $normal_path ./drivestudio_exp/$scene_idx/${prefix}normal.mp4
cp $depth_path ./drivestudio_exp/$scene_idx/${prefix}normalized_depth.mp4
cp $albedo_path ./drivestudio_exp/$scene_idx/${prefix}albedo.mp4
cp $roughness_path ./drivestudio_exp/$scene_idx/${prefix}roughness.mp4
cp $metallic_path ./drivestudio_exp/$scene_idx/${prefix}metallic.mp4
cp $envmap_path ./drivestudio_exp/$scene_idx/${prefix}envmap_0.hdr

python split.py ./drivestudio_exp/$scene_idx/${prefix}pbr_rgb.mp4
python split.py ./drivestudio_exp/$scene_idx/${prefix}gt_rgb.mp4
python split.py ./drivestudio_exp/$scene_idx/${prefix}normal.mp4
python split.py ./drivestudio_exp/$scene_idx/${prefix}normalized_depth.mp4
python split.py ./drivestudio_exp/$scene_idx/${prefix}albedo.mp4
python split.py ./drivestudio_exp/$scene_idx/${prefix}roughness.mp4
python split.py ./drivestudio_exp/$scene_idx/${prefix}metallic.mp4

