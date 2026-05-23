from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
import imageio
import numpy as np

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views,
    save_single_image,
    save_single_hdr,
)
from utils.visualization import to8b, depth_visualizer

from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

@torch.no_grad()
def render_rotate_envmap(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True,
    vis_timestep = 0,
):
    trainer.set_eval()
    
    # print("debugging...")
    # trainer.models['Sky'].apply_mask()

    logger.info("Evaluating Full Set...")
    tmp_path = f"{cfg.log_dir}/visualize"
    os.makedirs(tmp_path, exist_ok=True)
    
    num_step = args.num_frames
    start_angle = 0
    end_angle = 360
    angles = np.linspace(start_angle, end_angle, num_step, endpoint=False).tolist()
    
    angles = [angle % 360 for angle in angles]
    
    rotate_vertical = args.rotate_vertical  
    
    total_render_results = None
    for angle in tqdm(angles):
        trainer.set_envmap_rotation(angle, rotate_vertical=rotate_vertical)
        vis_indices = [dataset.pixel_source.num_cams * vis_timestep + i for i in range(dataset.pixel_source.num_cams)]
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=False,
            compute_error_map=cfg.render.vis_error,
            vis_indices=vis_indices,
            tmp_path=tmp_path,
            timing=args.enable_timing,
        )
        if total_render_results is None:
            total_render_results = render_results
        else:
            for key in render_results.keys():
                if key in total_render_results:
                    total_render_results[key] = total_render_results[key] + render_results[key]
                else:
                    total_render_results[key] = render_results[key]

        # save images for every rendering pass
    
        image_output_dir = f"{cfg.log_dir}/videos{post_fix}/images/"
        os.makedirs(image_output_dir, exist_ok=True)
        for i, key in enumerate(render_keys):
            if key not in render_results:
                continue
            num_cams = dataset.pixel_source.num_cams
            if num_cams == 1:
                if "depth" in key and "normal" not in key:
                    opacity = render_results["opacities"]
                    depth = render_results[key]
                    frame = depth_visualizer(depth[0], opacity[0])
                    frame = [to8b(frame)]
                    res = np.concatenate(frame, axis=1)
                else:
                    res = to8b(render_results[key][0])
            else:
                if "depth" in key and "normal" not in key:
                    opacity = render_results["opacities"]
                    depth = render_results[key]
                    frame0 = depth_visualizer(depth[0], opacity[0])
                    frame1 = depth_visualizer(depth[1], opacity[1])
                    frame2 = depth_visualizer(depth[2], opacity[2])
                    frame = [frame1, frame0, frame2]
                    frame = [to8b(f) for f in frame]
                    res = np.concatenate(frame, axis=1)
                else:
                
                    res0 = to8b(render_results[key][0])
                    res1 = to8b(render_results[key][1])
                    res2 = to8b(render_results[key][2])
                
                    res = np.concatenate([res1, res0, res2], axis=1)
            imageio.imwrite(f"{image_output_dir}/{angle:03f}_{key}.png", res)
        
        # save envmap for every rendering pass
        envmap = render_results["envmap_visualize"]
        envmap = to8b(envmap)
        imageio.imwrite(f"{image_output_dir}/{angle:03f}_envmap.png", envmap)
        
        hdr_envmap_path = f"{image_output_dir}/{angle:03f}_envmap.hdr"
        save_single_hdr(hdr_image=render_results['hdr_envmap_visualize'], save_pth=hdr_envmap_path)

    if args.render_video_postfix is None:
        video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
    else:
        video_output_pth = (
            f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
        )
        
    vis_frame_dict = save_videos(
        total_render_results,
        video_output_pth,
        layout=dataset.layout,
        num_timestamps=len(angles),
        keys=render_keys,
        num_cams=dataset.pixel_source.num_cams,
        save_seperate_video=cfg.logging.save_seperate_video,
        fps=cfg.render.fps,
        verbose=True,
    )
    
    if 'envmap_visualize' in render_results:
        envmap_dir = os.path.join(cfg.log_dir, "images", f"{cfg.log_dir}/videos{post_fix}", "envmap")
        os.makedirs(envmap_dir, exist_ok=True)
        for i, angle in enumerate(angles):
            # save envmap
            envmap_path = os.path.join(
                cfg.log_dir, "images", f"{cfg.log_dir}/videos{post_fix}/envmap/envmap_{i}.png"
            )
            save_single_image(image=render_results['envmap_visualize'], save_pth=envmap_path)
            
    del render_results, vis_frame_dict
    torch.cuda.empty_cache()
    

def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    args.enable_wandb = False
    for folder in [f"videos{args.postfix}", f"metrics{args.postfix}"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import pdb; pdb.set_trace()
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
        device=device,
        use_pbr=True,
    )
    
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    
    # load the envmap if specified
    # TODO: test loading envmap, and create new envmap
    if args.new_envmap_path is not None:
        if not os.path.exists(args.new_envmap_path):
            raise FileNotFoundError(f"Envmap path {args.new_envmap_path} does not exist.")
        trainer.load_envmap(envmap_path=args.new_envmap_path, prior_path=cfg.model.Sky.params.prior_path)
        logger.info(f"Using new envmap from {args.new_envmap_path}")
    
    if args.calibrate_envmap:
        logger.info("Calibrating envmap, overriding the stored transform (if any)")
        _, ff_cam_infos = dataset.full_image_set.get_image(
            idx=0,
            camera_downscale=1.0,
        )
        c2w = ff_cam_infos["camera_to_world"]
        camera_dir = c2w[:3, 2]
        camera_dir = camera_dir / (camera_dir.norm() + 1e-8)  # normalize
        trainer.calibrate_envmap(camera_dir=camera_dir)
    
    # print("debugging...")
    # trainer.models['Sky'].apply_mask()
    
    # trainer.models['Sky'].clip_bottom()
    # trainer.models['Sky'].update_pdf()
    # trainer.models['Sky'].activation_name = "none"
    # trainer.models['Sky'].activation = lambda x: x  # disable activation for debugging
    
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "rgb_sky",
        # "Background_rgbs",
        # "RigidNodes_rgbs",
        # "DeformableNodes_rgbs",
        # "SMPLNodes_rgbs",
        "depths",
        "normalized_depths",
        "normals",
        "albedos",
        "full_albedos",
        "opacities",
        "roughnesses",
        "full_roughnesses",
        "metallics",
        "normals_minscale",
        "depth_normals",
        # "raytrace_rgbs",
        # "raytrace_depths",
        # "raytrace_opacities",
        # "raytrace_normals",
        "raytrace_visibilities",
        "raytrace_ind_lights",
        "pbr_colors",
        "pbr_color_fulls",
        "pbr_diffuses",
        "pbr_speculars",
        "pbr_diffuse_brdfs",
        "pbr_specular_brdfs",
        # "pbr_specular_brdfs_NoVs",
        # "pbr_specular_brdfs_NoLs",
        # "pbr_specular_brdfs_NoHs",
        # "pbr_specular_brdfs_VoHs",
        # "pbr_specular_brdfs_Ds",
        # "pbr_specular_brdfs_Gvs",
        # "pbr_specular_brdfs_Gls",
        # "pbr_specular_brdfs_Frs",
        "pbr_transports",
        "pbr_dir_lights",
        "pbr_n_d_is",
        "pbr_lights",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    
    render_rotate_envmap(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix=args.postfix,
        vis_timestep=args.vis_timestep,
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    parser.add_argument("--postfix", type=str, default="_eval", help="an optional postfix for video")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    parser.add_argument("--enable_timing", action="store_true", help="enable per-op timing logs")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    # extra
    parser.add_argument("--new_envmap_path", type=str, default=None, help="path to new envmap to use")
    parser.add_argument("--calibrate_envmap", action="store_true", help="calibrate the envmap by the camera direction of the first frame")
    parser.add_argument("--vis_timestep", type=int, default=0, help="the timestep to visualize the envmap rotation")
    parser.add_argument("--num_frames", type=int, default=10, help="number of frames to render for the rotation")
    parser.add_argument("--rotate_vertical", action="store_true", help="whether to rotate vertically")
    
    args = parser.parse_args()
    main(args)
