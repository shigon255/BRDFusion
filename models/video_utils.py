from typing import Literal, Dict, List, Optional, Callable
from tqdm import tqdm, trange
import numpy as np
import os
import logging
import imageio
import imageio.v3 as iio

import torch
from torch import Tensor
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim

from models.graphics_utils import depth_to_normal
from datasets.base import SplitWrapper
from models.trainers.base import BasicTrainer
from utils.visualization import (
    to8b,
    depth_visualizer,
    vis_surface_normal,
    depth_to_grayscale,
)
from utils.timing import OpTimer

logger = logging.getLogger()

def get_numpy(x: Tensor) -> np.ndarray:
    return x.squeeze().cpu().numpy()

def non_zero_mean(x: Tensor) -> float:
    return sum(x) / len(x) if len(x) > 0 else -1

def compute_psnr_masked(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    """
    Computes the PSNR between prediction and target tensors within a given mask.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.
        mask (torch.Tensor): Boolean or float mask tensor, same shape as prediction/target (or broadcastable).

    Returns:
        float: The PSNR value within the masked region.
    """
    if not isinstance(prediction, Tensor):
        prediction = torch.tensor(prediction)
    if not isinstance(target, Tensor):
        target = torch.tensor(target).to(prediction.device)
    if not isinstance(mask, Tensor):
        mask = torch.tensor(mask).to(prediction.device)
    mask = mask.bool()
    if mask.sum() == 0:
        return -1
    pred_masked = prediction[mask]
    target_masked = target[mask]
    mse = F.mse_loss(pred_masked, target_masked)
    return (-10 * torch.log10(mse)).item()

def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def render_images(
    trainer: BasicTrainer,
    dataset: SplitWrapper,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None,
    verbose: bool = True,
    tmp_path: str = './',
    isosurface_render: bool = False,
    timing: bool = False,
    measure_fps: bool = False,
):
    """
    Render pixel-related outputs from a model.

    Args:
        ....skip obvious args
        compute_metrics (bool, optional): Whether to compute metrics. Defaults to False.
        vis_indices (Optional[List[int]], optional): Indices to visualize. Defaults to None.
    """
    trainer.set_eval()
    render_results = render(
        dataset,
        trainer=trainer,
        compute_metrics=compute_metrics,
        compute_error_map=compute_error_map,
        vis_indices=vis_indices,
        verbose=verbose,
        tmp_path=tmp_path,
        isosurface_render=isosurface_render,
        timing=timing,
        measure_fps=measure_fps,
    )
    if compute_metrics:
        num_samples = len(dataset) if vis_indices is None else len(vis_indices)
        logger.info(f"Eval over {num_samples} images:")
        logger.info(f"\t Full Image  PSNR: {render_results['psnr']:.4f}")
        logger.info(f"\t Full Image  SSIM: {render_results['ssim']:.4f}")
        logger.info(f"\t Full Image LPIPS: {render_results['lpips']:.4f}")
        logger.info(f"\t  PBR Image PSNR: {render_results['pbr_psnr']:.4f}")
        logger.info(f"\t  PBR Image SSIM: {render_results['pbr_ssim']:.4f}")
        logger.info(f"\t  PBR Image LPIPS: {render_results['pbr_lpips']:.4f}")
        logger.info(f"\t     Non-Sky PSNR: {render_results['occupied_psnr']:.4f}")
        logger.info(f"\t     Non-Sky SSIM: {render_results['occupied_ssim']:.4f}")
        logger.info(f"\tDynamic-Only PSNR: {render_results['masked_psnr']:.4f}")
        logger.info(f"\tDynamic-Only SSIM: {render_results['masked_ssim']:.4f}")
        logger.info(f"\t  Human-Only PSNR: {render_results['human_psnr']:.4f}")
        logger.info(f"\t  Human-Only SSIM: {render_results['human_ssim']:.4f}")
        logger.info(f"\tVehicle-Only PSNR: {render_results['vehicle_psnr']:.4f}")
        logger.info(f"\tVehicle-Only SSIM: {render_results['vehicle_ssim']:.4f}")
        logger.info(f"\t     Sky Region PSNR: {render_results['sky_psnr']:.4f}")
        logger.info(f"\t   Ground Region PSNR: {render_results['ground_psnr']:.4f}")
        logger.info(f"\t   Ground Region PBR PSNR: {render_results['ground_pbr_psnr']:.4f}")

    return render_results


def render(
    dataset: SplitWrapper,
    trainer: BasicTrainer = None,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None,
    verbose: bool = True,
    tmp_path: str = './',
    isosurface_render: bool = False,
    timing: bool = False,
    measure_fps: bool = False,
):
    """
    Renders a dataset utilizing a specified render function.

    Parameters:
        dataset: Dataset to render.
        trainer: Gaussian trainer, includes gaussian models and rendering modules
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        compute_error_map: Optional; if True, the function will compute and return error maps. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    # rgbs
    rgbs, gt_rgbs, rgb_sky_blends, rgb_skys = [], [], [], []
    rgb_gaussians = []
    point_light_emitters_rgbs = []
    Background_rgbs, RigidNodes_rgbs, DeformableNodes_rgbs, SMPLNodes_rgbs, Dynamic_rgbs = [], [], [], [], []
    error_maps = []

    # depths
    depths, lidar_on_images = [], []
    Background_depths, RigidNodes_depths, DeformableNodes_depths, SMPLNodes_depths, Dynamic_depths = [], [], [], [], []

    # normals
    world_normals = []
    normals = []
    depth_normals = []
    
    # albedos
    albedos = []
    full_albedos = []
    
    # roughnesses
    roughnesses = []
    full_roughnesses = []
    
    # metallics
    metallics = []

    # normals_minscale
    normals_minscale = []

    # distortion map
    distortion_maps = []

    # raytracing result
    raytrace_rgb = []
    raytrace_opacity = []
    raytrace_depth = []
    raytrace_normal = []
    raytrace_visibility = []
    raytrace_visibility_refine = []
    raytrace_ind_light = []
    raytrace_valid_count = []
    
    
    # pbr results
    pbr_colors, pbr_diffuses, pbr_speculars, pbr_transports = [], [], [], []
    pbr_diffuse_brdfs, pbr_specular_brdfs = [], []
    pbr_specular_brdfs_NoVs, pbr_specular_brdfs_NoLs, pbr_specular_brdfs_NoHs, pbr_specular_brdfs_VoHs = [], [], [], []
    pbr_specular_brdfs_Ds, pbr_specular_brdfs_Gvs, pbr_specular_brdfs_Gls, pbr_specular_brdfs_Frs = [], [], [], []
    pbr_dir_lights = []
    pbr_n_d_is = []
    pbr_lights = []
    pbr_color_fulls = []
    pbr_color_alones = []

    # sky
    opacities, sky_masks = [], []
    Background_opacities, RigidNodes_opacities, DeformableNodes_opacities, SMPLNodes_opacities, Dynamic_opacities = [], [], [], [], []
    envmap_visualize = None
    hdr_envmap_visualize = None
    
    # misc
    cam_names, cam_ids = [], []

    if compute_metrics:
        psnrs, ssim_scores, lpipss = [], [], []
        masked_psnrs, masked_ssims = [], []
        human_psnrs, human_ssims = [], []
        vehicle_psnrs, vehicle_ssims = [], []
        occupied_psnrs, occupied_ssims = [], []
        pbr_psnrs, pbr_ssim_scores, pbr_lpipss = [], [], []
        sky_psnrs, ground_psnrs, ground_pbr_psnrs = [], [], []

    frame_render_times_ms = []

    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        camera_downscale = trainer._get_downscale_factor()
        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            frame_timer = OpTimer(enabled=timing, use_cuda_sync=False, prefix=f"render {dataset.split} idx {i}")
            # get image and camera infos
            with frame_timer.time_block("data"):
                image_infos, cam_infos = dataset.get_image(i, camera_downscale)
            with frame_timer.time_block("to_cuda"):
                for k, v in image_infos.items():
                    if isinstance(v, Tensor):
                        image_infos[k] = v.cuda(non_blocking=True)
                for k, v in cam_infos.items():
                    if isinstance(v, Tensor):
                        cam_infos[k] = v.cuda(non_blocking=True)
            
            # Human-readable directions derived from OpenCV camera axes:
            # +x is right, +y is down, +z is forward.
            cam_position = cam_infos["camera_to_world"][:3, 3]
            cam_forward = cam_infos["camera_to_world"][:3, 2]
            cam_up = -cam_infos["camera_to_world"][:3, 1]
            cam_left = -cam_infos["camera_to_world"][:3, 0]
            print(f"Camera position: {cam_position.cpu().numpy()}")
            print(f"Camera forward: {cam_forward.cpu().numpy()}")
            print(f"Camera up: {cam_up.cpu().numpy()}")
            print(f"Camera left: {cam_left.cpu().numpy()}")

            # render the image
            if (timing or measure_fps) and torch.cuda.is_available():
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()
                results = trainer(image_infos, cam_infos, verbose=verbose, isosurface_render=isosurface_render, timing=timing)
                forward_end.record()
                torch.cuda.synchronize()
                elapsed_ms = forward_start.elapsed_time(forward_end)
                if timing:
                    frame_timer.add_time("forward_gpu", elapsed_ms / 1000.0)
                if measure_fps:
                    frame_render_times_ms.append(elapsed_ms)
            else:
                if measure_fps:
                    _fps_t0 = time.perf_counter()
                with frame_timer.time_block("forward"):
                    results = trainer(image_infos, cam_infos, verbose=verbose, isosurface_render=isosurface_render, timing=timing)
                if measure_fps:
                    frame_render_times_ms.append((time.perf_counter() - _fps_t0) * 1000.0)
            
            if measure_fps:
                print(f"Current FPS: {1000.0 / np.mean(frame_render_times_ms):.2f}, Average render time per frame: {np.mean(frame_render_times_ms):.2f} ms")
            frame_timer.start("post")
            # ------------- clip rgb ------------- #
            if timing and torch.cuda.is_available():
                post_start = torch.cuda.Event(enable_timing=True)
                post_end = torch.cuda.Event(enable_timing=True)
                post_start.record()
            for k, v in results.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    results[k] = v.clamp(0., 1.)
            if timing and torch.cuda.is_available():
                post_end.record()
                torch.cuda.synchronize()
                frame_timer.add_time("post_gpu", post_start.elapsed_time(post_end) / 1000.0)
            frame_timer.stop("post")
            frame_timer.start("collect")
            
            # ------------- cam names ------------- #
            cam_names.append(cam_infos["cam_name"])
            cam_ids.append(
                cam_infos["cam_id"].flatten()[0].cpu().numpy()
            )

            # ------------- rgb ------------- #
            rgb = results["rgb"]
            if torch.isnan(rgb).any():
                print("Warning: rgb has nan")
                rgb[torch.isnan(rgb)] = 1.0
            rgbs.append(get_numpy(rgb))
            if "pixels" in image_infos:
                gt_rgbs.append(get_numpy(image_infos["pixels"]))
                
            green_background = torch.tensor([0.0, 177, 64]) / 255.0
            green_background = green_background.to(rgb.device)
            if "Background_rgb" in results:
                Background_rgb = results["Background_rgb"] * results[
                    "Background_opacity"
                ] + green_background * (1 - results["Background_opacity"])
                Background_rgbs.append(get_numpy(Background_rgb))
            if "RigidNodes_rgb" in results:
                RigidNodes_rgb = results["RigidNodes_rgb"] * results[
                    "RigidNodes_opacity"
                ] + green_background * (1 - results["RigidNodes_opacity"])
                RigidNodes_rgbs.append(get_numpy(RigidNodes_rgb))
            if "DeformableNodes_rgb" in results:
                DeformableNodes_rgb = results["DeformableNodes_rgb"] * results[
                    "DeformableNodes_opacity"
                ] + green_background * (1 - results["DeformableNodes_opacity"])
                DeformableNodes_rgbs.append(get_numpy(DeformableNodes_rgb))
            if "SMPLNodes_rgb" in results:
                SMPLNodes_rgb = results["SMPLNodes_rgb"] * results[
                    "SMPLNodes_opacity"
                ] + green_background * (1 - results["SMPLNodes_opacity"])
                SMPLNodes_rgbs.append(get_numpy(SMPLNodes_rgb))
            if "Dynamic_rgb" in results:
                Dynamic_rgb = results["Dynamic_rgb"] * results[
                    "Dynamic_opacity"
                ] + green_background * (1 - results["Dynamic_opacity"])
                Dynamic_rgbs.append(get_numpy(Dynamic_rgb))
            if compute_error_map:
                # cal mean squared error
                error_map = (rgb - image_infos["pixels"]) ** 2
                error_map = error_map.mean(dim=-1, keepdim=True)
                # scale
                error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
                error_map = error_map.repeat_interleave(3, dim=-1)
                error_maps.append(get_numpy(error_map))
            if "rgb_sky_blend" in results:
                rgb_sky_blend = results["rgb_sky_blend"]
                if torch.isnan(rgb_sky_blend).any():
                    print("Warning: rgb_sky_blend has nan")
                    rgb_sky_blend[torch.isnan(rgb_sky_blend)] = 1.0
                rgb_sky_blends.append(get_numpy(rgb_sky_blend))
            if "rgb_sky" in results:
                rgb_sky = results["rgb_sky"]
                if torch.isnan(rgb_sky).any():
                    print("Warning: rgb_sky has nan")
                    rgb_sky[torch.isnan(rgb_sky)] = 1.0
                rgb_skys.append(get_numpy(rgb_sky))
            if "rgb_gaussians" in results:
                rgb_gaussian = results["rgb_gaussians"]
                if torch.isnan(rgb_gaussian).any():
                    print("Warning: rgb_gaussian has nan")
                    rgb_gaussian[torch.isnan(rgb_gaussian)] = 1.0
                rgb_gaussians.append(get_numpy(rgb_gaussian))
            if "point_light_emitters_rgb" in results:
                point_light_emitters_rgb = results["point_light_emitters_rgb"]
                if torch.isnan(point_light_emitters_rgb).any():
                    print("Warning: point_light_emitters_rgb has nan")
                    point_light_emitters_rgb[torch.isnan(point_light_emitters_rgb)] = 1.0
                point_light_emitters_rgbs.append(get_numpy(point_light_emitters_rgb))
            # ------------- depth ------------- #
            depth = results["depth"]
            depths.append(get_numpy(depth))
            # ------------- mask ------------- #
            if "opacity" in results:
                opacities.append(get_numpy(results["opacity"]))
            if "Background_depth" in results:
                Background_depths.append(get_numpy(results["Background_depth"]))
                Background_opacities.append(get_numpy(results["Background_opacity"]))
            if "RigidNodes_depth" in results:
                RigidNodes_depths.append(get_numpy(results["RigidNodes_depth"]))
                RigidNodes_opacities.append(get_numpy(results["RigidNodes_opacity"]))
            if "DeformableNodes_depth" in results:
                DeformableNodes_depths.append(get_numpy(results["DeformableNodes_depth"]))
                DeformableNodes_opacities.append(get_numpy(results["DeformableNodes_opacity"]))
            if "SMPLNodes_depth" in results:
                SMPLNodes_depths.append(get_numpy(results["SMPLNodes_depth"]))
                SMPLNodes_opacities.append(get_numpy(results["SMPLNodes_opacity"]))
            if "Dynamic_depth" in results:
                Dynamic_depths.append(get_numpy(results["Dynamic_depth"]))
                Dynamic_opacities.append(get_numpy(results["Dynamic_opacity"]))
            if "sky_masks" in image_infos:
                sky_masks.append(get_numpy(image_infos["sky_masks"]))
            
            # ------------- extra ------------- #
            normals.append(vis_surface_normal(get_numpy(results["normal"])))
            normals_minscale.append(vis_surface_normal(get_numpy(results["normal_minscale"])))
            world_normals.append(vis_surface_normal(get_numpy(results["world_normal"])))
            depth_normal = depth_to_normal(
                c2w=cam_infos["camera_to_world"].reshape(4, 4),
                K=cam_infos["intrinsics"].reshape(3, 3),
                depth_map=depth.squeeze(-1),
                img_width=cam_infos["width"].item(),
                img_height=cam_infos["height"].item(),
            )
            depth_normals.append(vis_surface_normal(get_numpy(depth_normal)))
            albedos.append(get_numpy(results["albedo"]))
            roughnesses.append(get_numpy(results["roughness"].unsqueeze(-1)))
            metallics.append(get_numpy(results["metallic"].unsqueeze(-1)))
            full_albedos.append(get_numpy(results["full_albedo"]))
            full_roughnesses.append(get_numpy(results["full_roughness"].unsqueeze(-1)))
            if results["distortion_map"] is not None:
                distortion_map = get_numpy(results["distortion_map"].unsqueeze(-1))
                # # normalize to [0, 1]
                # distortion_map = (distortion_map - distortion_map.min()) / (distortion_map.max() - distortion_map.min() + 1e-10)
                distortion_maps.append(distortion_map)
            
            if "raytrace_rgb" in results:
                raytrace_rgb.append(get_numpy(results["raytrace_rgb"]))
            if "raytrace_opacity" in results:
                raytrace_opacity.append(get_numpy(results["raytrace_opacity"]))
            if "raytrace_depth" in results:
                raytrace_depth.append(get_numpy(results["raytrace_depth"]))
            if "raytrace_normal" in results:
                raytrace_normal.append(vis_surface_normal(get_numpy(results["raytrace_normal"])))
            
            if "raytrace_visibility" in results:
                raytrace_visibility.append(get_numpy(results["raytrace_visibility"]))
            if "raytrace_visibility_refine" in results:
                raytrace_visibility_refine.append(get_numpy(results["raytrace_visibility_refine"]))
            if "raytrace_ind_light" in results:
                raytrace_ind_light.append(get_numpy(results["raytrace_ind_light"]))
            if "raytrace_valid_count" in results:
                raytrace_valid_count.append(get_numpy(results["raytrace_valid_count"]))
            if "pbr_color" in results:
                pbr_rgb = results["pbr_color"]
                # clip
                pbr_rgb = pbr_rgb.clamp(0., 1.)
                if torch.isnan(pbr_rgb).any():
                    print("Warning: pbr_rgb has nan")
                    pbr_rgb[torch.isnan(pbr_rgb)] = 1.0
                pbr_colors.append(get_numpy(pbr_rgb))
            if "pbr_color_full" in results:
                pbr_rgb_full = results["pbr_color_full"]
                # clip
                pbr_rgb_full = pbr_rgb_full.clamp(0., 1.)
                if torch.isnan(pbr_rgb_full).any():
                    print("Warning: pbr_rgb_full has nan")
                    pbr_rgb_full[torch.isnan(pbr_rgb_full)] = 1.0
                pbr_color_fulls.append(get_numpy(pbr_rgb_full))
            if "pbr_color_alone" in results:
                pbr_rgb_alone = results["pbr_color_alone"]
                # clip
                pbr_rgb_alone = pbr_rgb_alone.clamp(0., 1.)
                if torch.isnan(pbr_rgb_alone).any():
                    print("Warning: pbr_rgb_alone has nan")
                    pbr_rgb_alone[torch.isnan(pbr_rgb_alone)] = 1.0
                pbr_color_alones.append(get_numpy(pbr_rgb_alone))
            if "pbr_diffuse" in results:
                pbr_diffuses.append(get_numpy(results["pbr_diffuse"]))
            if "pbr_specular" in results:
                pbr_speculars.append(get_numpy(results["pbr_specular"]))
            if "pbr_diffuse_brdf" in results:
                pbr_diffuse_brdfs.append(get_numpy(results["pbr_diffuse_brdf"]))
            if "pbr_specular_brdf" in results:
                pbr_specular_brdfs.append(get_numpy(results["pbr_specular_brdf"]))
            if "pbr_specular_brdf_NoV" in results:
                pbr_specular_brdfs_NoVs.append(get_numpy(results["pbr_specular_brdf_NoV"]))
            if "pbr_specular_brdf_NoL" in results:
                pbr_specular_brdfs_NoLs.append(get_numpy(results["pbr_specular_brdf_NoL"]))
            if "pbr_specular_brdf_NoH" in results:
                pbr_specular_brdfs_NoHs.append(get_numpy(results["pbr_specular_brdf_NoH"]))
            if "pbr_specular_brdf_VoH" in results:
                pbr_specular_brdfs_VoHs.append(get_numpy(results["pbr_specular_brdf_VoH"]))
            if "pbr_specular_brdf_D" in results:
                pbr_specular_brdfs_Ds.append(get_numpy(results["pbr_specular_brdf_D"]))
            if "pbr_specular_brdf_Gv" in results:
                pbr_specular_brdfs_Gvs.append(get_numpy(results["pbr_specular_brdf_Gv"]))
            if "pbr_specular_brdf_Gl" in results:
                pbr_specular_brdfs_Gls.append(get_numpy(results["pbr_specular_brdf_Gl"]))
            if "pbr_specular_brdf_Fr" in results:
                pbr_specular_brdfs_Frs.append(get_numpy(results["pbr_specular_brdf_Fr"]))
            if "pbr_transport" in results:                
                pbr_transports.append(get_numpy(results["pbr_transport"]))
            if "pbr_dir_light" in results:
                pbr_dir_lights.append(get_numpy(results["pbr_dir_light"]))
            if "pbr_n_d_i" in results:
                pbr_n_d_is.append(get_numpy(results["pbr_n_d_i"]))
            if "pbr_lights" in results:
                pbr_lights.append(get_numpy(results["pbr_lights"]))
            
            if envmap_visualize is None and "envmap_visualize" in results:
                # clip
                envmap_visualize = get_numpy(results["envmap_visualize"].clamp(0., 1.))
                
            if hdr_envmap_visualize is None and "hdr_envmap_visualize" in results:
                hdr_envmap_visualize = get_numpy(results["hdr_envmap_visualize"])
            

            # ------------- lidar ------------- #
            if "lidar_depth_map" in image_infos and "pixels" in image_infos:
                depth_map = image_infos["lidar_depth_map"]
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                lidar_on_image = image_infos["pixels"].cpu().numpy() * (1 - mask) + depth_img * mask
                lidar_on_images.append(lidar_on_image)

            frame_timer.stop("collect")
            if compute_metrics:
                frame_timer.start("metrics")
                psnr = compute_psnr(rgb, image_infos["pixels"])
                ssim_score = ssim(
                    get_numpy(rgb),
                    get_numpy(image_infos["pixels"]),
                    data_range=1.0,
                    channel_axis=-1,
                )
                lpips = trainer.lpips(
                    rgb[None, ...].permute(0, 3, 1, 2),
                    image_infos["pixels"][None, ...].permute(0, 3, 1, 2)
                )
                logger.info(f"Frame {i}: PSNR {psnr:.4f}, SSIM {ssim_score:.4f}")
                psnrs.append(psnr)
                ssim_scores.append(ssim_score)
                lpipss.append(lpips.item())
                
                # sky region psnr
                sky_psnr = compute_psnr_masked(
                    rgb,
                    image_infos["pixels"],
                    image_infos["sky_masks"].bool() if "sky_masks" in image_infos else torch.zeros_like(rgb[..., 0]).bool(),
                )
                sky_psnrs.append(sky_psnr)
                
                # ground region psnr
                ground_psnr = compute_psnr_masked(
                    rgb,
                    image_infos["pixels"],
                    ~(image_infos["sky_masks"].bool()) if "sky_masks" in image_infos else torch.ones_like(rgb[..., 0]).bool(),
                )
                ground_psnrs.append(ground_psnr)
                
                if "pbr_color" in results:
                    pbr_psnr = compute_psnr(pbr_rgb, image_infos["pixels"])
                    pbr_ssim_score = ssim(
                        get_numpy(pbr_rgb),
                        get_numpy(image_infos["pixels"]),
                        data_range=1.0,
                        channel_axis=-1,
                    )
                    pbr_lpips = trainer.lpips(
                        pbr_rgb[None, ...].permute(0, 3, 1, 2),
                        image_infos["pixels"][None, ...].permute(0, 3, 1, 2)
                    )
                    logger.info(f"Frame {i}: PBR PSNR {pbr_psnr:.4f}, PBR SSIM {pbr_ssim_score:.4f}")
                    pbr_psnrs.append(pbr_psnr)
                    pbr_ssim_scores.append(pbr_ssim_score)
                    pbr_lpipss.append(pbr_lpips.item())
                    
                    # ground region pbr psnr
                    ground_pbr_psnr = compute_psnr_masked(
                        pbr_rgb,
                        image_infos["pixels"],
                        ~(image_infos["sky_masks"].bool()) if "sky_masks" in image_infos else torch.ones_like(rgb[..., 0]).bool(),
                    )
                    ground_pbr_psnrs.append(ground_pbr_psnr)
                
                # # testing
                # tmp_path = './tmp4'
                # os.makedirs(tmp_path, exist_ok=True)
                # imageio.imwrite(f'{tmp_path}/sky_blend.png', to8b(get_numpy(results["rgb_sky_blend"])))
                # imageio.imwrite(f'{tmp_path}/sky.png', to8b(get_numpy(results["rgb_sky"])))
                # imageio.imwrite(f'{tmp_path}/rgb_gs.png', to8b(get_numpy(results["rgb_gaussians"])))
                # imageio.imwrite(f'{tmp_path}/rgb.png', to8b(get_numpy(rgb)))
                # imageio.imwrite(f'{tmp_path}/albedo.png', to8b(get_numpy(results["albedo"])))
                # imageio.imwrite(f'{tmp_path}/normal.png', to8b(get_numpy(results["normal"])*0.5+0.5))
                # imageio.imwrite(f'{tmp_path}/roughness.png', to8b(get_numpy(results["roughness"].unsqueeze(-1))))
                # imageio.imwrite(f'{tmp_path}/metallic.png', to8b(get_numpy(results["metallic"].unsqueeze(-1))))
                # imageio.imwrite(f'{tmp_path}/pbr_rgb.png', to8b(get_numpy(pbr_rgb)))
                # imageio.imwrite(f'{tmp_path}/pbr_rgb_full.png', to8b(get_numpy(pbr_rgb)))
                # imageio.imwrite(f'{tmp_path}/gt_rgb_full.png', to8b(get_numpy(image_infos["pixels"])))
                # imageio.imwrite(f'{tmp_path}/envmap.png', to8b(envmap_visualize))
                # imageio.imwrite(f'{tmp_path}/pbr_transport.png', to8b(get_numpy(results["pbr_transport"])))
                # envmap_hdr = results["hdr_envmap_visualize"]
                # save_single_hdr(get_numpy(envmap_hdr), f'{tmp_path}/envmap.hdr')
                # opacity = get_numpy(results["opacity"])
                # opacity = np.stack([opacity] * 3, axis=-1)
                # opacity = to8b(opacity)
                # imageio.imwrite(f'{tmp_path}/opacity.png', opacity)
                # visibility = get_numpy(results["raytrace_visibility"])
                # visibility = np.stack([visibility] * 3, axis=-1)
                # visibility = to8b(visibility)
                # imageio.imwrite(f'{tmp_path}/visibility.png', visibility)
                # visibility_refine = get_numpy(results["raytrace_visibility_refine"])
                # visibility_refine = np.stack([visibility_refine] * 3, axis=-1)
                # visibility_refine = to8b(visibility_refine)
                # imageio.imwrite(f'{tmp_path}/visibility_refine.png', visibility_refine)
                # import pdb; pdb.set_trace()
                
                if "sky_masks" in image_infos:
                    occupied_mask = ~get_numpy(image_infos["sky_masks"]).astype(bool)
                    if occupied_mask.sum() > 0:
                        occupied_psnrs.append(
                            compute_psnr(
                                rgb[occupied_mask], image_infos["pixels"][occupied_mask]
                            )
                        )
                        occupied_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][occupied_mask].mean()
                        )

                if "dynamic_masks" in image_infos:
                    dynamic_mask = get_numpy(image_infos["dynamic_masks"]).astype(bool)
                    if dynamic_mask.sum() > 0:
                        masked_psnrs.append(
                            compute_psnr(
                                rgb[dynamic_mask], image_infos["pixels"][dynamic_mask]
                            )
                        )
                        masked_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][dynamic_mask].mean()
                        )
                
                if "human_masks" in image_infos:
                    human_mask = get_numpy(image_infos["human_masks"]).astype(bool)
                    if human_mask.sum() > 0:
                        human_psnrs.append(
                            compute_psnr(
                                rgb[human_mask], image_infos["pixels"][human_mask]
                            )
                        )
                        human_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][human_mask].mean()
                        )
                
                if "vehicle_masks" in image_infos:
                    vehicle_mask = get_numpy(image_infos["vehicle_masks"]).astype(bool)
                    if vehicle_mask.sum() > 0:
                        vehicle_psnrs.append(
                            compute_psnr(
                                rgb[vehicle_mask], image_infos["pixels"][vehicle_mask]
                            )
                        )
                        vehicle_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][vehicle_mask].mean()
                        )
                frame_timer.stop("metrics")
            frame_timer.log()
            
    # messy aggregation...
    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    results_dict["pbr_psnr"] = non_zero_mean(pbr_psnrs) if compute_metrics else -1
    results_dict["pbr_ssim"] = non_zero_mean(pbr_ssim_scores) if compute_metrics else -1
    results_dict["pbr_lpips"] = non_zero_mean(pbr_lpipss) if compute_metrics else -1
    results_dict["occupied_psnr"] = non_zero_mean(occupied_psnrs) if compute_metrics else -1
    results_dict["occupied_ssim"] = non_zero_mean(occupied_ssims) if compute_metrics else -1
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1
    results_dict["human_psnr"] = non_zero_mean(human_psnrs) if compute_metrics else -1
    results_dict["human_ssim"] = non_zero_mean(human_ssims) if compute_metrics else -1
    results_dict["vehicle_psnr"] = non_zero_mean(vehicle_psnrs) if compute_metrics else -1
    results_dict["vehicle_ssim"] = non_zero_mean(vehicle_ssims) if compute_metrics else -1
    results_dict["sky_psnr"] = non_zero_mean(sky_psnrs) if compute_metrics else -1
    results_dict["ground_psnr"] = non_zero_mean(ground_psnrs) if compute_metrics else -1
    results_dict["ground_pbr_psnr"] = non_zero_mean(ground_pbr_psnrs) if compute_metrics else -1
    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["normals"] = normals
    results_dict["normals_minscale"] = normals_minscale
    results_dict["world_normals"] = world_normals
    results_dict["depth_normals"] = depth_normals
    results_dict["albedos"] = albedos
    results_dict["roughnesses"] = roughnesses
    results_dict["metallics"] = metallics
    results_dict["full_albedos"] = full_albedos
    results_dict["full_roughnesses"] = full_roughnesses
    if len(distortion_maps) > 0:
        results_dict["distortion_maps"] = distortion_maps
    
    if len(raytrace_rgb) > 0:
        results_dict["raytrace_rgbs"] = raytrace_rgb
    if len(raytrace_depth) > 0:
        results_dict["raytrace_depths"] = raytrace_depth
    if len(raytrace_opacity) > 0:
        results_dict["raytrace_opacities"] = raytrace_opacity
    if len(raytrace_normal) > 0:
        results_dict["raytrace_normals"] = raytrace_normal
    if len(raytrace_visibility) > 0:
        results_dict["raytrace_visibilities"] = raytrace_visibility
    if len(raytrace_visibility_refine) > 0:
        results_dict["raytrace_visibilities_refine"] = raytrace_visibility_refine
    if len(raytrace_ind_light) > 0:
        results_dict["raytrace_ind_lights"] = raytrace_ind_light
    if len(raytrace_valid_count) > 0:
        results_dict["raytrace_valid_counts"] = raytrace_valid_count
    if len(pbr_colors) > 0:
        results_dict["pbr_colors"] = pbr_colors
    if len(pbr_color_fulls) > 0:
        results_dict["pbr_color_fulls"] = pbr_color_fulls
    if len(pbr_color_alones) > 0:
        results_dict["pbr_color_alones"] = pbr_color_alones
    if len(pbr_diffuses) > 0:
        results_dict["pbr_diffuses"] = pbr_diffuses
    if len(pbr_speculars) > 0:
        results_dict["pbr_speculars"] = pbr_speculars
    if len(pbr_diffuse_brdfs) > 0:
        results_dict["pbr_diffuse_brdfs"] = pbr_diffuse_brdfs
    if len(pbr_specular_brdfs) > 0:
        results_dict["pbr_specular_brdfs"] = pbr_specular_brdfs
    if len(pbr_specular_brdfs_NoVs) > 0:
        results_dict["pbr_specular_brdfs_NoVs"] = pbr_specular_brdfs_NoVs
    if len(pbr_specular_brdfs_NoLs) > 0:
        results_dict["pbr_specular_brdfs_NoLs"] = pbr_specular_brdfs_NoLs
    if len(pbr_specular_brdfs_NoHs) > 0:
        results_dict["pbr_specular_brdfs_NoHs"] = pbr_specular_brdfs_NoHs
    if len(pbr_specular_brdfs_VoHs) > 0:
        results_dict["pbr_specular_brdfs_VoHs"] = pbr_specular_brdfs_VoHs
    if len(pbr_specular_brdfs_Ds) > 0:
        results_dict["pbr_specular_brdfs_Ds"] = pbr_specular_brdfs_Ds
    if len(pbr_specular_brdfs_Gvs) > 0:
        results_dict["pbr_specular_brdfs_Gvs"] = pbr_specular_brdfs_Gvs
    if len(pbr_specular_brdfs_Gls) > 0:
        results_dict["pbr_specular_brdfs_Gls"] = pbr_specular_brdfs_Gls
    if len(pbr_specular_brdfs_Frs) > 0:
        results_dict["pbr_specular_brdfs_Frs"] = pbr_specular_brdfs_Frs
    if len(pbr_transports) > 0:
        results_dict["pbr_transports"] = pbr_transports
    if len(pbr_dir_lights) > 0:
        results_dict["pbr_dir_lights"] = pbr_dir_lights
    if len(pbr_n_d_is) > 0:
        results_dict["pbr_n_d_is"] = pbr_n_d_is
    if len(pbr_lights) > 0:
        results_dict["pbr_lights"] = pbr_lights
    

    if envmap_visualize is not None:
        results_dict["envmap_visualize"] = envmap_visualize
    
    if hdr_envmap_visualize is not None:
        results_dict["hdr_envmap_visualize"] = hdr_envmap_visualize
    
    results_dict["cam_names"] = cam_names
    results_dict["cam_ids"] = cam_ids
    
    if len(opacities) > 0:
        results_dict["opacities"] = opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(error_maps) > 0:
        results_dict["rgb_error_maps"] = error_maps
    if len(rgb_sky_blends) > 0:
        results_dict["rgb_sky_blend"] = rgb_sky_blends
    if len(rgb_skys) > 0:
        results_dict["rgb_sky"] = rgb_skys
    if len(rgb_gaussians) > 0:
        results_dict["rgb_gaussian"] = rgb_gaussians
    if len(point_light_emitters_rgbs) > 0:
        results_dict["point_light_emitters_rgbs"] = point_light_emitters_rgbs
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(lidar_on_images) > 0:
        results_dict["lidar_on_images"] = lidar_on_images
    if len(Background_rgbs) > 0:
        results_dict["Background_rgbs"] = Background_rgbs
    if len(RigidNodes_rgbs) > 0:
        results_dict["RigidNodes_rgbs"] = RigidNodes_rgbs
    if len(DeformableNodes_rgbs) > 0:
        results_dict["DeformableNodes_rgbs"] = DeformableNodes_rgbs
    if len(SMPLNodes_rgbs) > 0:
        results_dict["SMPLNodes_rgbs"] = SMPLNodes_rgbs
    if len(Dynamic_rgbs) > 0:
        results_dict["Dynamic_rgbs"] = Dynamic_rgbs
    if len(Background_depths) > 0:
        results_dict["Background_depths"] = Background_depths
    if len(RigidNodes_depths) > 0:
        results_dict["RigidNodes_depths"] = RigidNodes_depths
    if len(DeformableNodes_depths) > 0:
        results_dict["DeformableNodes_depths"] = DeformableNodes_depths
    if len(SMPLNodes_depths) > 0:
        results_dict["SMPLNodes_depths"] = SMPLNodes_depths
    if len(Dynamic_depths) > 0:
        results_dict["Dynamic_depths"] = Dynamic_depths
    if len(Background_opacities) > 0:
        results_dict["Background_opacities"] = Background_opacities
    if len(RigidNodes_opacities) > 0:
        results_dict["RigidNodes_opacities"] = RigidNodes_opacities
    if len(DeformableNodes_opacities) > 0:
        results_dict["DeformableNodes_opacities"] = DeformableNodes_opacities
    if len(SMPLNodes_opacities) > 0:
        results_dict["SMPLNodes_opacities"] = SMPLNodes_opacities
    if len(Dynamic_opacities) > 0:
        results_dict["Dynamic_opacities"] = Dynamic_opacities

    if measure_fps and len(frame_render_times_ms) > 0:
        avg_ms = float(np.mean(frame_render_times_ms))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        logger.info(
            f"Rendering FPS: {fps:.2f}  "
            f"(avg {avg_ms:.2f} ms/frame over {len(frame_render_times_ms)} frames)"
        )
        results_dict["render_fps"] = fps
        results_dict["render_ms_per_frame"] = avg_ms

    return results_dict

def save_single_image(
    image: torch.Tensor,
    save_pth: str,
):
    image = to8b(image)
    imageio.imwrite(save_pth, image)

def save_single_hdr(
    hdr_image: np.array,
    save_pth: str,
):
    iio.imwrite(save_pth, hdr_image, plugin='HDR-FI')

def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):  
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def render_novel_views(trainer, render_data: list, save_path: str, fps: int = 30, isosurface_render=False) -> None:
    """
    Perform rendering and save the result as a video.
    
    Args:
        trainer: Trainer object containing the rendering method
        render_data (list): List of dicts, each containing elements required for rendering a single frame
        save_path (str): Path to save the output video
        fps (int): Frames per second for the output video
    """
    trainer.set_eval()  
    
    writer = imageio.get_writer(save_path, mode='I', fps=fps)
    pbr_writer = imageio.get_writer(save_path.replace('.mp4', '_pbr.mp4'), mode='I', fps=fps)
    
    with torch.no_grad():
        for frame_data in render_data:
            # Move data to GPU
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][key] = value.cuda(non_blocking=True)
            
            # Perform rendering
            outputs = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True,
                isosurface_render=isosurface_render,
            )
            
            # Extract RGB image and mask
            rgb = outputs["rgb"].cpu().numpy().clip(
                min=1.e-6, max=1-1.e-6
            )
            
            # Convert to uint8 and write to video
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            writer.append_data(rgb_uint8)
            
            pbr_rgb = outputs["pbr_color"].cpu().numpy().clip(
                min=1.e-6, max=1-1.e-6
            )
            pbr_rgb_uint8 = (pbr_rgb * 255).astype(np.uint8)
            pbr_writer.append_data(pbr_rgb_uint8)
    
    writer.close()
    pbr_writer.close()
    print(f"Video saved to {save_path}")


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
        for key in keys:
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            elif "normalized_depths" in key:
                # extract depths
                new_key = key.replace("normalized_depths", "depths")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key or "opacities" in key or "visibilities" in key or "n_d_is" in key \
                or "brdfs_NoVs" in key or "brdfs_NoLs" in key or "brdfs_NoHs" in key or "brdfs_VoHs" in key \
                or "brdfs_Ds" in key or "brdfs_Gvs" in key or "brdfs_Gls" in key or "brdfs_Frs" in key \
                or "valid_counts" in key or "roughness" in key or "metallic" in key:
                frames = [
                    np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key and "normal" not in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                if "normalized" in key:
                    frames = [
                        depth_to_grayscale(frame, opacity)
                        for frame, opacity in zip(frames, opacities)
                    ]
                else:
                    frames = [
                        depth_visualizer(frame, opacity)
                        for frame, opacity in zip(frames, opacities)
                    ]
            elif "distortion_map" in key:
                # visualize using depth visualizer
                frames = [
                    depth_visualizer(frame, np.ones_like(frame))
                    for frame in frames
                ]
            tiled_img = layout(frames, cam_names)
            # frames = np.concatenate(frames, axis=1)
            merged_list.append(tiled_img)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        logger.info(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}

    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if "mask" not in key and "normalized_depths" not in key:
            if key not in render_results or len(render_results[key]) == 0:
                continue
        for i in range(num_timestamps):
            cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            elif "normalized_depths" in key:
                # extract depths
                new_key = key.replace("normalized_depths", "depths")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key or "opacities" in key or "visibilities" in key or "n_d_is" in key \
                or "brdfs_NoVs" in key or "brdfs_NoLs" in key or "brdfs_NoHs" in key or "brdfs_VoHs" in key \
                or "brdfs_Ds" in key or "brdfs_Gvs" in key or "brdfs_Gls" in key or "brdfs_Frs" in key \
                or "valid_counts" in key:
                frames = [
                    np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key and "normal" not in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            elif "normalized_depths" in key:
                opacities = render_results[key.replace("normalized_depths", "opacities")][
                    i * num_cams : (i + 1) * num_cams
                ]
                frames = [
                    depth_to_grayscale(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            elif "distortion_map" in key:
                # visualize using depth visualizer
                frames = [
                    depth_visualizer(frame, np.ones_like(frame))
                    for frame in frames
                ]
            elif "roughness" in key or "metallic" in key:
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            try:
                tiled_img = layout(frames, cam_names)
            except Exception as e:
                print(e)
            
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"/{i:03d}_{j:03d}.png"),
                        to8b(frame),
                    )
            # frames = to8b(np.concatenate(frames, axis=1))
            frames = to8b(tiled_img)
            writer.append_data(frames)
            if i == return_frame_id:
                return_frame_dict[key] = frames
        print(f"saved video for {key} to {tmp_save_pth}")
        # close the writer
        writer.close()
        del writer
        if verbose:
            logger.info(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict
