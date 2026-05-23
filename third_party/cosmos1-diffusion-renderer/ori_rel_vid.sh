CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/test_example_results/video_delighting/gbuffer_frames --num_video_frames 57 \
    --envlight_ind 2 --use_custom_envmap=True \
    --video_save_folder=asset/test_example_results/video_relighting/