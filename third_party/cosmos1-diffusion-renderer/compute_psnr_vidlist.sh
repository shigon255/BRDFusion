
scene_idx=3
cam_id=1

python compute_psnr_vidlist.py \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/gt_rgb_${cam_id}.mp4 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w0_relit_cam${cam_id}_seed1000_960x640.mp4=0.0 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.1_relit_cam${cam_id}_seed1000_960x640.mp4=0.1 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.2_relit_cam${cam_id}_seed1000_960x640.mp4=0.2 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.3_relit_cam${cam_id}_seed1000_960x640.mp4=0.3 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.4_relit_cam${cam_id}_seed1000_960x640.mp4=0.4 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.5_relit_cam${cam_id}_seed1000_960x640.mp4=0.5 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.6_relit_cam${cam_id}_seed1000_960x640.mp4=0.6 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.7_relit_cam${cam_id}_seed1000_960x640.mp4=0.7 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.8_relit_cam${cam_id}_seed1000_960x640.mp4=0.8 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlenvmap/sdedit_w.9_relit_cam${cam_id}_seed1000_960x640.mp4=0.9 \
    /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/$scene_idx/dlpbr_rgb_${cam_id}.mp4=1.0 \
    --max_frames 57 \
    --output_plot ./drivestudio_exp/$scene_idx/dlenvmap/sdedit_strength_cam${cam_id}_metric.png 
