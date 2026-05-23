# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
#     --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
#     --dataset_path=asset/waymo/036/images/ \
#     --num_video_frames 57 \
#     --group_mode folder \
#     --normalize_normal True \
#     --video_save_folder=asset/waymo/036/video_delighting_test \
#     --offload_diffusion_transformer --offload_tokenizer \

# s_churn=0.0
# s_noise=1.00
# s_tmin=0.0
# s_tmax=100.0
# num_steps=15
# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
#     --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
#     --dataset_path=asset/waymo_114/video_frames_examples/ --num_video_frames 57 --group_mode folder \
#     --video_save_folder=asset/waymo_114/video_delighting_test_schurn${s_churn}_snoise${s_noise}_numstep${num_steps}_tmin${s_tmin}_tmax${s_tmax} \
#     --normalize_normal True \
#     --s_churn $s_churn --s_noise $s_noise \
#     --s_tmin $s_tmin --s_tmax $s_tmax \
#     --num_steps $num_steps \

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/waymo_3_pbrrgb_invfor/video_frames_examples/ \
    --num_video_frames 57 \
    --group_mode folder \
    --normalize_normal True \
    --video_save_folder=asset/waymo_3_pbrrgb_invfor/video_delighting
