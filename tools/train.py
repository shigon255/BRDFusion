from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from tools.eval import do_evaluation
from utils.misc import import_str
from utils.config import resolve_config
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from utils.timing import OpTimer
from models.video_utils import render_images, save_videos, save_single_image, save_single_hdr
from datasets.driving_dataset import DrivingDataset
from models.modules import EnvLight_SG, EnvLight_SH_SG

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)

    cfg = resolve_config(
        config_file=args.config_file,
        opts=args.opts,
        overlays=args.config_overlay,
    )
    assert "data" in cfg, "Please specify dataset in config or data in config"
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    
    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    
    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./", 
        ["configs", "datasets", "models", "utils", "tools"], 
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg

def main(args):
    cfg = setup(args)
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
    
    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )
    
    # load the envmap to replace the one in the checkpoint if provided
    if args.new_envmap_path is not None:
        if not os.path.exists(args.new_envmap_path):
            raise FileNotFoundError(f"New envmap path {args.new_envmap_path} does not exist.")
        trainer.load_envmap(
            envmap_path=args.new_envmap_path,
            activation_name=cfg.model.Sky.params.activation_name,
            resolution=cfg.model.Sky.params.resolution,
            min_res=cfg.model.Sky.params.min_res,
            max_res=cfg.model.Sky.params.max_res,
            min_roughness=cfg.model.Sky.params.min_roughness,
            max_roughness=cfg.model.Sky.params.max_roughness,
            init_value=cfg.model.Sky.params.init_value,
            prior_path=cfg.model.Sky.params.prior_path,
        )
        logger.info(f"Replacing envmap with {args.new_envmap_path}")

    if args.debug_convert_envmap_to_sg:
        logger.warning(
            "DEBUGGING: converting EnvLight_EQ to EnvLight_SH_SG in train.py; remove this block when done."
        )
        if "Sky" not in trainer.models:
            raise ValueError("Sky model not found; cannot convert envmap to SG.")
        sg_model = EnvLight_SH_SG(
            class_name="Sky",
            resolution=cfg.model.Sky.params.resolution,
            device=device,
            sh_degree=args.debug_sh_degree,
            peak_threshold=args.debug_peak_threshold,
            peak_kernel_size=args.debug_peak_kernel_size,
            peak_min_distance=args.debug_peak_min_distance,
            sg_amplitude_scale=args.debug_sg_amplitude,
            path=cfg.model.Sky.params.prior_path,
            prior_path=cfg.model.Sky.params.prior_path,
            sharpness_schedule=args.debug_sg_sharpness_schedule,
            sharpness_start_step=args.debug_sg_sharpness_start_step,
            sharpness_end_step=args.debug_sg_sharpness_end_step,
            sharpness_start=args.debug_sg_sharpness_start,
            sharpness_end=args.debug_sg_sharpness_end,
        )
        sg_model.build_mips()
        sg_model.update_pdf()
        trainer.models["Sky"] = sg_model
    
    # Calibrate the envmap by the camera direction of the first frame
    if not args.no_calibrate_envmap:
        logger.info("Calibrating envmap...")
        _, ff_cam_infos = dataset.full_image_set.get_image(
            idx=0,
            camera_downscale=1.0,
        )
        c2w = ff_cam_infos["camera_to_world"]
        camera_dir = c2w[:3, 2]
        camera_dir = camera_dir / (camera_dir.norm() + 1e-8)  # normalize
        trainer.calibrate_envmap(camera_dir=camera_dir)
    
    # Filter the envmap to remove ground if specified
    if args.filter_ground_envmap:
        logger.info("Filtering envmap to remove ground...")
        for timestep in range(cfg.data.start_timestep, cfg.data.end_timestep + 1):
            for index in range(dataset.pixel_source.num_cams):
                image_infos, _ = dataset.full_image_set.get_image(
                    idx=timestep * dataset.pixel_source.num_cams + index,
                    camera_downscale=1.0,
                )
                viewdirs = image_infos["viewdirs"]
                sky_mask = image_infos["sky_masks"].bool()
                ground_mask = ~sky_mask
                # ground_mask = torch.ones_like(viewdirs).bool()
                ground_dirs = viewdirs[ground_mask]
                trainer.mask_envmap(dirs=ground_dirs)
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "rgb_sky",
        "rgb_sky_blend",
        "rgb_gaussian",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        "normals",
        "opacities",
        "albedos",
        "roughnesses",
        "metallics",
        "normals_minscale",
        "world_normals",
        "depth_normals",
        "distortion_maps",
        # "raytrace_rgbs",
        # "raytrace_depths",
        # "raytrace_opacities",
        # "raytrace_normals",
        "raytrace_visibilities",
        "raytrace_visibilities_refine",
        "raytrace_ind_lights",
        "raytrace_valid_counts",
        "pbr_colors",
        "pbr_color_fulls",
        "pbr_color_alones",
        "pbr_diffuses",
        "pbr_speculars",
        "pbr_transports",
        "pbr_dir_lights",
        "pbr_n_d_is",
        "pbr_lights",
        "Background_depths",
        "Dynamic_depths",
        "RigidNodes_depths",
        "DeformableNodes_depths",
        "SMPLNodes_depths",
        "RigidNodes_opacities",
        "DeformableNodes_opacities",
        "SMPLNodes_opacities",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    # setup optimizer  
    trainer.initialize_optimizer()
    
    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)
    start_step = trainer.step

    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    if args.vis_frame_mode in ["first", "middle"]:
        print(f"Visualizing frames using mode {args.vis_frame_mode}...")
    # check if it is a number
    elif args.vis_frame_mode is not None and args.vis_frame_mode.isdigit():
        print(f"Visualizing frames using fixed timestep {args.vis_frame_mode}...")
    else:
        print("Visualizing frames sequentially...")

    # one-time full render bootstrap for ray-level PBR error cache
    if (
        getattr(trainer, "use_pbr", False)
        and trainer.tracer_cfg is not None
        and trainer.tracer_cfg.get("use_ray_error_cache", False)
        and trainer.tracer_cfg.get("ray_error_bootstrap_full_render", False)
    ):
        logger.info("Bootstrapping ray error cache with a full PBR render pass...")
        bootstrap_indices = np.arange(dataset.pixel_source.num_imgs).tolist()
        with torch.no_grad():
            bootstrap_results = render_images(
                trainer=trainer,
                dataset=dataset.full_image_set,
                compute_metrics=False,
                compute_error_map=False,
                vis_indices=bootstrap_indices,
                verbose=True,
                timing=args.enable_timing,
            )
            dataset.pixel_source.bootstrap_ray_error_maps_from_render(
                render_results=bootstrap_results,
                img_indices=bootstrap_indices,
                use_pbr=True,
                min_error=trainer.tracer_cfg.get("ray_error_weight_min", 1e-6),
            )
            if trainer.tracer_cfg.get("ray_error_bootstrap_save", False):
                per_cam_maps = []
                per_cam_names = []
                vis_timestep = cfg.data.start_timestep
                local_frame = vis_timestep - cfg.data.start_timestep
                for cam_id in dataset.pixel_source.camera_list:
                    unique_cam_idx = dataset.pixel_source.camera_data[cam_id].unique_cam_idx
                    img_idx = local_frame * dataset.pixel_source.num_cams + unique_cam_idx
                    err_map = dataset.pixel_source.get_ray_error_map(int(img_idx)).detach().cpu().numpy()
                    per_cam_maps.append(np.stack([err_map, err_map, err_map], axis=-1))
                    per_cam_names.append(dataset.pixel_source.camera_data[cam_id].cam_name)
                stacked = np.stack([m[..., 0] for m in per_cam_maps], axis=0)
                denom = stacked.max() - stacked.min()
                if denom > 1e-8:
                    stacked = (stacked - stacked.min()) / denom
                else:
                    stacked = np.zeros_like(stacked)
                per_cam_maps = [np.stack([m, m, m], axis=-1) for m in stacked]
                tiled = dataset.layout(per_cam_maps, per_cam_names)
                save_single_image(
                    image=tiled,
                    save_pth=os.path.join(cfg.log_dir, "images", "step_0_ray_error_map_bootstrap.png"),
                )
        del bootstrap_results
        torch.cuda.empty_cache()
        trainer.set_train()
        logger.info("Finished ray error cache bootstrap.")

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        #----------------------------------------------------------------------------
        #----------------------------     Validate     ------------------------------
        
        if (step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0):
            logger.info("Visualizing...")
            trainer.set_eval()
            
            if args.vis_frame_mode == "first":
                # fix the validation timestep to the first timestep
                vis_timestep = 0
            elif args.vis_frame_mode == "middle":
                # fix the validation timestep to the middle of all timesteps
                vis_timestep = cfg.data.start_timestep + (cfg.data.end_timestep - cfg.data.start_timestep) // 2
            elif args.vis_frame_mode is not None and args.vis_frame_mode.isdigit():
                # fix the validation timestep to the specified one
                vis_timestep = int(args.vis_frame_mode)
            else:
                # split the timesteps by logging_vis_freq
                # e.g. if logging_vis_freq=10, and num_iters=100, then
                # vis_timestep will be [0, 10, 20, ..., 90]
                vis_timestep = np.linspace(
                    0,
                    dataset.num_img_timesteps,
                    trainer.num_iters // cfg.logging.vis_freq + 1,
                    endpoint=False,
                    dtype=int,
                )[step // cfg.logging.vis_freq]
            
            # DEBUG
            if args.debug_single_timestep >= 0 and trainer.use_pbr:
                print(f"DEBUGGING: validating on the timestep {args.debug_single_timestep}")
                vis_timestep = args.debug_single_timestep
            
            vis_indices = [
                vis_timestep * dataset.pixel_source.num_cams + i
                for i in range(dataset.pixel_source.num_cams)
            ]

            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=vis_indices,
                    verbose=True,
                    timing=args.enable_timing,
                )

            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": render_results["psnr"],
                        "image_metrics/ssim": render_results["ssim"],
                        "image_metrics/pbr_psnr": render_results["pbr_psnr"],
                        "image_metrics/pbr_ssim": render_results["pbr_ssim"],
                        "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                        "image_metrics/occupied_ssim": render_results["occupied_ssim"],
                        "image_metrics/sky_psnr": render_results["sky_psnr"],
                        "image_metrics/ground_psnr": render_results["ground_psnr"],
                        "image_metrics/ground_pbr_psnr": render_results["ground_pbr_psnr"],
                    }
                )
            vis_frame_dict = save_videos(
                render_results,
                save_pth=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.png"
                ),  # don't save the video
                layout=dataset.layout,
                num_timestamps=1,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=dataset.pixel_source.num_cams,
                fps=cfg.render.fps,
                verbose=False,
            )
            
            if 'envmap_visualize' in render_results:
                # save envmap
                envmap_path = os.path.join(
                    cfg.log_dir, "images", f"step_{step}_envmap.png"
                )
                
                save_single_image(image=render_results['envmap_visualize'], save_pth=envmap_path)
            if 'hdr_envmap_visualize' in render_results:
                hdr_envmap_path = os.path.join(
                    cfg.log_dir, "images", f"step_{step}_envmap.hdr"
                )
                save_single_hdr(hdr_image=render_results['hdr_envmap_visualize'], save_pth=hdr_envmap_path)

            # save cached ray error maps for the logged views
            if (
                getattr(trainer, "use_pbr", False)
                and trainer.tracer_cfg is not None
                and trainer.tracer_cfg.get("use_ray_error_cache", False)
                and dataset.pixel_source.ray_error_maps is not None
            ):
                ray_error_maps = []
                ray_error_cam_names = []
                cam_names = render_results.get("cam_names", [])
                rgbs = render_results.get("rgbs", [])
                for local_idx, img_idx in enumerate(vis_indices):
                    try:
                        err_map = dataset.pixel_source.get_ray_error_map(int(img_idx)).detach()
                    except Exception:
                        continue
                    if local_idx < len(rgbs):
                        target_h, target_w = rgbs[local_idx].shape[:2]
                        if err_map.shape[0] != target_h or err_map.shape[1] != target_w:
                            err_map = torch.nn.functional.interpolate(
                                err_map[None, None, ...],
                                size=(target_h, target_w),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).squeeze(0)
                    ray_error_maps.append(err_map.cpu().numpy())
                    if local_idx < len(cam_names):
                        ray_error_cam_names.append(cam_names[local_idx])
                    else:
                        ray_error_cam_names.append(f"cam_{local_idx}")

                if len(ray_error_maps) > 0:
                    stacked = np.stack(ray_error_maps, axis=0)
                    denom = stacked.max() - stacked.min()
                    if denom < 1e-8:
                        stacked = np.zeros_like(stacked)
                    else:
                        stacked = (stacked - stacked.min()) / denom
                    vis_maps = [np.stack([m, m, m], axis=-1) for m in stacked]
                    tiled_ray_error = dataset.layout(vis_maps, ray_error_cam_names)
                    save_single_image(
                        image=tiled_ray_error,
                        save_pth=os.path.join(
                            cfg.log_dir, "images", f"step_{step}_ray_error_map.png"
                        ),
                    )
            
            
            # omit logging images to save space on wandb
            # if args.enable_wandb:
                # for k, v in vis_frame_dict.items():
                    # wandb.log({"image_rendering/" + k: wandb.Image(v)})
            del render_results
            torch.cuda.empty_cache()
            trainer.set_train()
                
        
        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        
        step_timer = OpTimer(enabled=args.enable_timing, prefix=f"train step {step}")
        with step_timer.time_block("prep"):
            # prepare for training
            trainer.set_train()
            trainer.preprocess_per_train_step(step=step)
            trainer.optimizer_zero_grad() # zero grad
        
        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        
        # DEBUG
        if args.debug_single_timestep >= 0 and trainer.use_pbr:
            # train on a single timestep
            cur_idx = step % dataset.pixel_source.num_cams + args.debug_single_timestep * dataset.pixel_source.num_cams
            idx = cur_idx
            
            # # debug
            # print("Debugging: using the center view for training")
            # cur_idx = args.debug_single_timestep * dataset.pixel_source.num_cams
            # idx = cur_idx
            
            with step_timer.time_block("data"):
                image_infos, cam_infos = dataset.full_image_set.get_image(
                    idx=cur_idx,
                    camera_downscale=train_step_camera_downscale,
                )
        else:
            with step_timer.time_block("data"):
                image_infos, cam_infos, idx = dataset.train_image_set.next(train_step_camera_downscale)
        
        # # debugging: fit on a single image
        # idx = (cfg.data.start_timestep + (cfg.data.end_timestep - cfg.data.start_timestep) // 2) * dataset.pixel_source.num_cams + 2
        # image_infos, cam_infos = dataset.full_image_set.get_image(
        #     idx=idx,
        #     camera_downscale=train_step_camera_downscale,
        # )
        
        # Buggy part commented out: the index is in global, but we should sample training timesteps. Need to fix it later if needed.
        # timestep = idx // dataset.pixel_source.num_cams
        # cam_id = idx % dataset.pixel_source.num_cams
        # multiviewdepth_w = cfg.trainer.losses.multiview_depth.w
        # if multiviewdepth_w > 0:

        #     # randomly sample 1 timestep in timestep-10 to timestep+10
        #     candidate_timesteps = list(range(
        #         max(timestep - 10, 0),
        #         min(timestep + 10, dataset.num_img_timesteps - 1) + 1
        #     ))
        #     candidate_timesteps.remove(timestep)
        #     if len(candidate_timesteps) > 0:
        #         sampled_timestep = random.choice(candidate_timesteps)
        #         candidate_idx = sampled_timestep * dataset.pixel_source.num_cams + cam_id
        #         with step_timer.time_block("data_mv"):
        #             mv_image_infos, mv_cam_infos = dataset.train_image_set.get_image(
        #                 idx=candidate_idx,
        #                 camera_downscale=train_step_camera_downscale,
        #             )
        #     else:
        #         raise ValueError("No candidate timestep for multiview depth loss.")
        
        with step_timer.time_block("to_cuda"):
            for k, v in image_infos.items():
                if isinstance(v, torch.Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, torch.Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)
            if (
                getattr(trainer, "use_pbr", False)
                and trainer.tracer_cfg is not None
                and trainer.tracer_cfg.get("use_ray_error_cache", False)
                and "pixels" in image_infos
                and "img_idx" in image_infos
            ):
                h, w = image_infos["pixels"].shape[:2]
                dataset.pixel_source.ensure_ray_error_maps(height=h, width=w)
                cur_img_idx = int(image_infos["img_idx"].flatten()[0].item())
                image_infos["ray_error_map"] = dataset.pixel_source.get_ray_error_map(cur_img_idx)
            # if multiviewdepth_w > 0:
            #     for k, v in mv_image_infos.items():
            #         if isinstance(v, torch.Tensor):
            #             mv_image_infos[k] = v.cuda(non_blocking=True)
            #     for k, v in mv_cam_infos.items():
            #         if isinstance(v, torch.Tensor):
            #             mv_cam_infos[k] = v.cuda(non_blocking=True)
            
        # forward & backward
        with step_timer.time_block("forward"):
            outputs = trainer(image_infos, cam_infos)
        
        with step_timer.time_block("vis_filter"):
            trainer.update_visibility_filter()

        with step_timer.time_block("loss"):
            loss_dict, loss_stat_dict = trainer.compute_losses(
                outputs=outputs,
                image_infos=image_infos,
                cam_infos=cam_infos,
                step=step, 
            )
        if (
            getattr(trainer, "use_pbr", False)
            and trainer.tracer_cfg is not None
            and trainer.tracer_cfg.get("use_ray_error_cache", False)
        ):
            with step_timer.time_block("update_ray_error_cache"):
                dataset.pixel_source.update_ray_error_map(
                    image_infos=image_infos,
                    outputs=outputs,
                    ema_decay=trainer.tracer_cfg.get("ray_error_ema_decay", 0.9),
                    min_error=trainer.tracer_cfg.get("ray_error_weight_min", 1e-6),
                )

        # if multiviewdepth_w > 0:
        #     # compute multiview depth loss
        #     with step_timer.time_block("mv_depth"):
        #         multiview_loss = trainer.compute_multiview_depth_loss(
        #             outputs=outputs,
        #             image_infos=image_infos,
        #             cam_infos=cam_infos,
        #             nearest_image_infos=mv_image_infos,
        #             nearest_cam_infos=mv_cam_infos,
        #         )
        #     loss_dict.update({
        #         "multiview_depth": multiview_loss
        #     })
        
        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                import pdb; pdb.set_trace()
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                import pdb; pdb.set_trace()
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        with step_timer.time_block("backward"):
            trainer.backward(loss_dict)
        
        # after training step
        with step_timer.time_block("post"):
            trainer.postprocess_per_train_step(step=step, timing=args.enable_timing)
        
        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with step_timer.time_block("metrics"):
            with torch.no_grad():
                # cal stats
                metric_dict = trainer.compute_metrics(
                    outputs=outputs,
                    image_infos=image_infos,
                )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        metric_logger.update(**{"train_stats/"+k: v for k, v in loss_stat_dict.items()})
        
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) # and (args.resume_from is None)
        if do_save:
            with step_timer.time_block("save"):
                trainer.save_checkpoint(
                    log_dir=cfg.log_dir,
                    save_only_model=True,
                    is_final=step == trainer.num_iters,
                )
        
        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            with step_timer.time_block("cache_error"):
                logger.info("Caching image error...")
                trainer.set_eval()
                with torch.no_grad():
                    dataset.pixel_source.update_downscale_factor(
                        1 / dataset.pixel_source.buffer_downscale
                    )
                    render_results = render_images(
                        trainer=trainer,
                        dataset=dataset.full_image_set,
                        verbose=True,
                        timing=args.enable_timing,
                    )
                    dataset.pixel_source.reset_downscale_factor()
                    dataset.pixel_source.update_image_error_maps(render_results)

                    # save error maps
                    merged_error_video = dataset.pixel_source.get_image_error_video(
                        dataset.layout
                    )
                    imageio.mimsave(
                        os.path.join(
                            cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                        ),
                        merged_error_video,
                        fps=cfg.render.fps,
                    )
                logger.info("Done caching rgb error maps")
        step_timer.log()
            
    
    logger.info("Training done!")

    # skip the full evaluation after training for now
    # do_evaluation(
    #     step=step,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
    
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument(
        "--config_overlay",
        action="append",
        default=[],
        help=(
            "optional overlay config; can be repeated. Merge order is "
            "base -> dataset -> overlays -> CLI opts"
        ),
    )
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)
    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    
    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--enable_timing", action="store_true", help="enable per-op timing logs")
    parser.add_argument("--entity", default="shigon", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    # for debugging
    parser.add_argument("--debug_single_timestep", type=int, default=-1, help="debug on a single timestep (-1 for False, positive for True)")
    parser.add_argument("--no_calibrate_envmap", action="store_true", help="do not calibrate envmap, useful for debugging")
    parser.add_argument("--new_envmap_path", type=str, default=None, help="path to new envmap, if not None, will replace the envmap in the checkpoint")
    parser.add_argument("--filter_ground_envmap", action="store_true", help="filter the envmap to remove ground, useful for debugging")
    parser.add_argument("--vis_frame_mode", type=str, default=None, help="the mode for visualization frame, if None, use the one in config")
    parser.add_argument("--debug_convert_envmap_to_sg", action="store_true", help="DEBUG: convert EnvLight_EQ to EnvLight_SH_SG after loading checkpoint")
    parser.add_argument("--debug_sg_sharpness_schedule", action="store_true", help="DEBUG: enable SG sharpness scheduling")
    parser.add_argument("--debug_sg_sharpness_start_step", type=int, default=0, help="DEBUG: SG sharpness schedule start step")
    parser.add_argument("--debug_sg_sharpness_end_step", type=int, default=0, help="DEBUG: SG sharpness schedule end step")
    parser.add_argument("--debug_sg_sharpness_start", type=float, default=16.0, help="DEBUG: SG sharpness schedule start value")
    parser.add_argument("--debug_sg_sharpness_end", type=float, default=16.0, help="DEBUG: SG sharpness schedule end value")
    parser.add_argument("--debug_sh_degree", type=int, default=3, help="DEBUG: SH degree for EnvLight_SH_SG")
    parser.add_argument("--debug_peak_threshold", type=float, default=0.75, help="DEBUG: peak threshold for EnvLight_SH_SG")
    parser.add_argument("--debug_peak_kernel_size", type=int, default=3, help="DEBUG: peak kernel size for EnvLight_SH_SG")
    parser.add_argument("--debug_peak_min_distance", type=int, default=12, help="DEBUG: peak min angle (degrees) for EnvLight_SH_SG")
    parser.add_argument("--debug_sg_amplitude", type=float, default=1.0, help="DEBUG: SG amplitude scale for EnvLight_SH_SG")
    
    # for eval
    parser.add_argument("--eval_start_timestep", type=int, default=None, help="start timestep for rendering, default to the training setting")
    parser.add_argument("--eval_num_timesteps", type=int, default=None, help="number of timesteps to render, default to the training setting")
    
    args = parser.parse_args()
    final_step = main(args)
