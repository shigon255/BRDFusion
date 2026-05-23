from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import numpy as np
import argparse

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)

import json
from tqdm import tqdm

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

@torch.no_grad()
def extract_caminfo_and_gaussian(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
):
    trainer.set_eval()
        
    logger.info("Extracting camera info and Gaussian...")
    # render_results = render_images(
    #     trainer=trainer,
    #     dataset=dataset.full_image_set,
    #     compute_metrics=True,
    #     compute_error_map=cfg.render.vis_error,
    # )
    # import pdb; pdb.set_trace()
    full_dataset = dataset.full_image_set
    if args.camera_output_path is not None:
        num_cams = full_dataset.datasource.num_cams
        intrinsics = []
        hs = []
        ws = []
        c2ws = []
        ids = []
        for id, i in enumerate(tqdm(range(0, len(full_dataset), num_cams))):
            _, cam_infos = full_dataset.get_image(i, 1)
            ids.append(id)
            c2ws.append(cam_infos["camera_to_world"].cpu().numpy().tolist())
            intrinsics.append(cam_infos["intrinsics"].cpu().numpy().tolist())
            hs.append(cam_infos["height"].item())
            ws.append(cam_infos["width"].item())
        
        cameras = []
        for i in range(len(ids)):
            cam_info = {
                "id": ids[i],
                "intrinsics": intrinsics[i],
                "height": hs[i],
                "width": ws[i],
                "extrinsics": c2ws[i]
            }
            cameras.append(cam_info)
    
        # save to json
        os.makedirs(os.path.dirname(args.camera_output_path), exist_ok=True)
        with open(args.camera_output_path, "w") as f:
            json.dump(cameras, f, indent=4)
        
    # save gaussian
    if args.gaussian_output_path is not None:
        os.makedirs(os.path.dirname(args.gaussian_output_path), exist_ok=True)
        vis_timestep = args.vis_timestep
        idx = full_dataset.datasource.num_cams * vis_timestep
        image_infos, _ = full_dataset.get_image(idx, 1)
        
        mask_func = None
        
        # # debugging
        # print("debugging: filtering gaussians...")
        # # get the camera pose at timestep 38 as anchor
        # num_cams = full_dataset.datasource.num_cams
        # idx = num_cams * 38  # timestep 38
        # anchor_cam_info = full_dataset.get_image(idx, 1.0)[1]
        # anchor_c2w = anchor_cam_info["camera_to_world"]
        # anchor_pos = anchor_c2w[:3, 3]
        # # get forward and right direction
        # forward = anchor_c2w[:3, 2]
        # forward = forward / torch.norm(forward)
        # right = anchor_c2w[:3, 0]
        # right = right / torch.norm(right)
        # up = right.cross(forward)
        # up = up / torch.norm(up)
        
        
        # def mask_func(xyz):
        #     # xyz: (N, 3), numpy array
        #     # anchor_pos and right are torch tensors
        #     mask = np.ones(xyz.shape[0], dtype=bool)
            
        #     # filter gs that is right to the anchor
        #     offset = right * 1.0
        #     rel_pos = xyz - (offset + anchor_pos).cpu().numpy().reshape(1, 3)  # (N, 3)
        #     rel_pos_vec = rel_pos / (np.linalg.norm(rel_pos, axis=1, keepdims=True) + 1e-8)
        #     right_np = right.cpu().numpy().reshape(3)
        #     right_dot = np.dot(rel_pos_vec, right_np)  # (N,)
        #     mask = mask & (right_dot > 0)
            
        #     # also filter out gs behind the anchor
        #     forward_np = forward.cpu().numpy().reshape(3)
        #     forward_dot = np.dot(rel_pos_vec, forward_np)  # (N,)
        #     mask = mask & (forward_dot > 0)
            
        #     # # also filter out gs that is too above the anchor
        #     # up_np = up.cpu().numpy().reshape(3)
        #     # up_dot = np.dot(rel_pos_vec, up_np)  # (N,)
        #     # up_dis = up_dot * np.linalg.norm(rel_pos, axis=1)  # (N,)
        #     # mask = mask & (up_dis < 3.0) 
            
        #     return mask    
    
        trainer.save_full_ply(save_path=args.gaussian_output_path, 
                            image_infos=image_infos, 
                            mask_func=mask_func)
        print(f"Saved gaussian to {args.gaussian_output_path}")
    
    
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )
    
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    
    extract_caminfo_and_gaussian(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        args=args,
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    parser.add_argument("--camera_output_path", type=str, default=None, help="camera output path")
    parser.add_argument("--gaussian_output_path", type=str, default='./gaussian.ply', help="gaussian output path")
    parser.add_argument("--vis_timestep", type=int, default=0, help="which timestep to visualize")
    
    args = parser.parse_args()
    main(args)