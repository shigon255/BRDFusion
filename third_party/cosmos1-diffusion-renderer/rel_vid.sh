# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#     --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#     --dataset_path=asset/waymo/036/video_delighting/gbuffer_frames \
#     --num_video_frames 57 \
#     --envlight_ind 5 --use_custom_envmap=True \
#     --video_save_folder=asset/waymo/036/video_relighting/ \
#     # --rotate_light=True --use_fixed_frame_ind=True \
#     # --offload_diffusion_transformer --offload_tokenizer \

s_churn=0.0
s_noise=1.0
s_tmin=0.0
s_tmax=100.0
envlight_ind=4
num_steps=25
sigma_min=0.002
sigma_max=80.0
rho=7.0
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
    --dataset_path=asset/waymo_3_invw0/video_delighting/gbuffer_frames --num_video_frames 57 \
    --envlight_ind $envlight_ind --use_custom_envmap=True \
    --video_save_folder=asset/waymo_3_invw0/video_relighting_schurn${s_churn}_snoise${s_noise}_lightid${envlight_ind}_numstep${num_steps}_tmin${s_tmin}_tmax${s_tmax}_sigmamin${sigma_min}_sigmamax${sigma_max}_rho${rho} \
    --s_churn $s_churn --s_noise $s_noise \
    --s_tmin $s_tmin --s_tmax $s_tmax \
    --num_steps $num_steps \
    --scheduler_sigma_min $sigma_min \
    --scheduler_sigma_max $sigma_max \
    --scheduler_rho $rho \

# CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#     --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#     --dataset_path=asset/waymo_36_test_new/video_delighting/gbuffer_frames --num_video_frames 57 \
#     --envlight_ind 2 --use_custom_envmap=True \
#     --video_save_folder=asset/waymo_36_test_new/video_relighting_nomodeldiffusionrandom \


# for seed in 1000 1489 2023; do
#     CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#         --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#         --dataset_path=drivestudio_exp/3/gbuffer_frames --num_video_frames 57 \
#         --envlight_ind 6 --use_custom_envmap=True \
#         --video_save_folder=drivestudio_exp/3/video_relighting/ \
#         --offload_diffusion_transformer --offload_tokenizer \
#         --seed $seed 

#     mv /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/3/video_relighting/video1__0000.relit_0006.mp4 \
#         /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/3/video_relighting/video1__0000.relit_0006_seed${seed}.mp4

#     CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#         --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#         --dataset_path=drivestudio_exp/3/gbuffer_frames --num_video_frames 57 \
#         --envlight_ind 6 --use_custom_envmap=True \
#         --video_save_folder=drivestudio_exp/3/video_relighting_rotation/ \
#         --offload_diffusion_transformer --offload_tokenizer \
#         --rotate_light=True --use_fixed_frame_ind=True --fixed_frame_ind 38 \
#         --seed $seed 
    
#     mv /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/3/video_relighting_rotation/video1__0000.relit_0006.mp4 \
#         /project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/3/video_relighting_rotation/video1__0000.relit_0006_seed${seed}.mp4
# done


# scene_idx=114
# envlight_inds=(3)
# gbuffer_folder_names=("drgbuffer_frames" "gbuffer_frames")

# for envlight_ind in "${envlight_inds[@]}"; do
#     for gbuffer_folder_name in "${gbuffer_folder_names[@]}"; do
#         save_folder_name="${gbuffer_folder_name}_relighting_rotate"
#         CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#                 --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#                 --dataset_path=drivestudio_exp/$scene_idx/$gbuffer_folder_name/ --num_video_frames 57 \
#                 --envlight_ind $envlight_ind --use_custom_envmap=True \
#                 --video_save_folder=drivestudio_exp/$scene_idx/$save_folder_name/ \
#                 --offload_diffusion_transformer --offload_tokenizer \
#                 --rotate_light=True --use_fixed_frame_ind=True --fixed_frame_ind 38 \
#                 --seed 1000   

#         save_folder_name="${gbuffer_folder_name}_relighting"
#         CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#                 --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#                 --dataset_path=drivestudio_exp/$scene_idx/$gbuffer_folder_name/ --num_video_frames 57 \
#                 --envlight_ind $envlight_ind --use_custom_envmap=True \
#                 --video_save_folder=drivestudio_exp/$scene_idx/$save_folder_name/ \
#                 --offload_diffusion_transformer --offload_tokenizer \
#                 --seed 1000        
#     done
# done

# scene_idx=3
# envlight_inds=(6 8)
# gbuffer_folder_names=("drgbuffer_frames" "gbuffer_frames")

# for envlight_ind in "${envlight_inds[@]}"; do
#     for gbuffer_folder_name in "${gbuffer_folder_names[@]}"; do
#         save_folder_name="${gbuffer_folder_name}_relighting_rotate"
#         CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
#                 --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
#                 --dataset_path=drivestudio_exp/$scene_idx/$gbuffer_folder_name/ --num_video_frames 57 \
#                 --envlight_ind $envlight_ind --use_custom_envmap=True \
#                 --video_save_folder=drivestudio_exp/$scene_idx/$save_folder_name/ \
#                 --offload_diffusion_transformer --offload_tokenizer \
#                 --rotate_light=True --use_fixed_frame_ind=True --fixed_frame_ind 38 \
#                 --seed 1000        
#     done
# done