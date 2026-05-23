CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/waymo/036/images --num_video_frames 1 --group_mode webdataset \
    --video_save_folder=asset/waymo/036/image_delighting_1280 --save_video=False \
    --resize_resolution 1280 1920 \
    --height 1280  --width 1920 \
    --normalize_normal True \