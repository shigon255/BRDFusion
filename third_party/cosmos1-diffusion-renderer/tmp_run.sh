
scene_idx=114
envlight_ind=4
gbuffer_folder_name="drgbuffer_frames"

save_folder_name="${gbuffer_folder_name}_relighting"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=drivestudio_exp/$scene_idx/$gbuffer_folder_name/ --num_video_frames 57 \
    --envlight_ind $envlight_ind --use_custom_envmap=True \
    --video_save_folder=drivestudio_exp/$scene_idx/$save_folder_name/ \
    --offload_diffusion_transformer --offload_tokenizer \
    --seed 1000  

save_folder_name="${gbuffer_folder_name}_relighting_rotate"
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=drivestudio_exp/$scene_idx/$gbuffer_folder_name/ --num_video_frames 57 \
    --envlight_ind $envlight_ind --use_custom_envmap=True \
    --video_save_folder=drivestudio_exp/$scene_idx/$save_folder_name/ \
    --offload_diffusion_transformer --offload_tokenizer \
    --rotate_light=True --use_fixed_frame_ind=True \
    --seed 1000  