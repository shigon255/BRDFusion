from typing import Dict
import math
import re
import torch
import logging
import torch.nn.functional as F

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere
import os
import numpy as np
from plyfile import PlyData, PlyElement
from models.graphics_utils import *

import cv2
from utils.visualization import vis_surface_normal, depth_visualizer, color_map
from models.gaussians.basics import *
from models.graphics_utils import *
from models.modules import EnvLight, EnvLight_EQ
from models.video_utils import vis_surface_normal

import threedgrt_tracer
from threedgrut.datasets.protocols import Batch
from utils.timing import OpTimer

logger = logging.getLogger()

class MultiTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True
        gamma_act_name = self.render_cfg.get('gamma_activation_name', None)
        if gamma_act_name is None:
            self.gamma_activation = None
        elif gamma_act_name == 'softplus':
            self.gamma_activation = F.softplus
        elif gamma_act_name == 'relu':
            self.gamma_activation = F.relu
        elif gamma_act_name == 'clip':
            self.gamma_activation = lambda x: torch.clamp(x, min=0.0, max=1.0)
        
        gamma = self.render_cfg.get('gamma', 2.2)
        if gamma_act_name is None:
            self.gamma_correction = None
            self.inv_gamma_correction = None
        else:
            self.gamma_correction = lambda x: torch.pow(self.gamma_activation(x), 1.0 / gamma)
            self.inv_gamma_correction = lambda x: torch.pow(torch.clamp(x, min=0.0), gamma)
            # # debugging
            # print(f"Debugging: testing ACES tone mapping")
            # def aces_formula(x):
            #     """ Narkowicz ACES Approximation """
            #     a = 2.51
            #     b = 0.03
            #     c = 2.43
            #     d = 0.59
            #     e = 0.14
            #     return (x * (a * x + b)) / (x * (c * x + d) + e)
            # def inv_aces_formula(x):
            #     x = torch.clamp(x, min=0.0, max=1.0)
            #     a = 2.51
            #     b = 0.03
            #     c = 2.43
            #     d = 0.59
                
            #     numerator_linear = d * x - b
            #     disc = -1.0127 * (x ** 2) + 1.3702 * x + 0.0009
            #     disc = torch.clamp(disc, min=0.0)
            #     root_disc = torch.sqrt(disc)
            #     denominator = 2 * (a - c * x)
                
            #     return torch.clamp((numerator_linear + root_disc) / (denominator + 1e-6), min=0.0)
                
            # self.gamma_correction = lambda x: torch.clamp(aces_formula(self.gamma_activation(x)), min=0.0, max=1.0)
            # self.inv_gamma_correction = lambda x: inv_aces_formula(torch.clamp(x, min=0.0, max=1.0))
            
        # self.gamma_correction = lambda x: torch.where(x > 0.0031308, torch.pow(torch.max(x, torch.tensor(0.0031308)), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * x)
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
        # update pdf and mips
        if "Sky" in self.models:
            if self.models['Sky'].forward_mode != "pure_env":
                self.models['Sky'].build_mips()
            self.models['Sky'].update_pdf()
    
    def _init_losses(self) -> None:
        super()._init_losses()

        env_tv_loss_fn = None
        env_tv_loss_cfg = self.losses_dict.get("env_tv", None)
        if env_tv_loss_cfg is not None:
            def tv_loss(envmap):
                # envmap: (H, W, 3) or (6, H, W, 3)
                if len(envmap.shape) == 3:
                    envmap = envmap.unsqueeze(0) # (1, H, W, 3)
                envmap = envmap.permute(0, 3, 1, 2) 
                tv_h = torch.pow(envmap[:, :, 1:, :] - envmap[:, :, :-1, :], 2).mean()
                tv_w = torch.pow(envmap[:, :, :, 1:] - envmap[:, :, :, :-1], 2).mean()
                return tv_h + tv_w
            env_tv_loss_fn = tv_loss
        self.env_tv_loss_fn = env_tv_loss_fn
        
        env_cauchy_loss_fn = None
        env_cauchy_loss_cfg = self.losses_dict.get("env_cauchy", None)
        if env_cauchy_loss_cfg is not None:
            def cauchy_loss(envmap):
                from models.modules import pixel_grid
                Y = pixel_grid(envmap.shape[1], envmap.shape[0])[..., 1]
                delta_angle = torch.sin(np.pi * Y)
                loss = (torch.log(1 + 2 * (envmap**2)) * delta_angle.unsqueeze(-1)).mean()
                return loss
            env_cauchy_loss_fn = cauchy_loss
        self.env_cauchy_loss_fn = env_cauchy_loss_fn
        
        env_white_loss_fn = None
        env_white_loss_cfg = self.losses_dict.get("env_white", None)
        if env_white_loss_cfg is not None:
            def white_loss(envmap):
                # TODO: consider delta angle weight
                avg = envmap.mean(dim=-1, keepdim=True)
                loss = ((envmap - avg) ** 2).mean()
                return loss
            env_white_loss_fn = white_loss
        self.env_white_loss_fn = env_white_loss_fn
        
        env_scaleinv_loss_fn = None
        env_scaleinv_loss_cfg = self.losses_dict.get("env_scaleinv", None)
        if env_scaleinv_loss_cfg is not None:
            from models.losses import compute_scale_and_shift, compute_shift
            def scaleinv_loss(envmap, prior_envmap):
                
                # scale-shift invariant
                # scale, shift = compute_scale_and_shift(envmap.detach(), prior_envmap)
                # aligned_envmap = envmap * scale + shift
                # loss = F.mse_loss(aligned_envmap, prior_envmap, reduction='mean')
                
                # shift-invariant MSE
                # exp_envmap = torch.exp(envmap)
                # exp_prior_envmap = torch.exp(prior_envmap)
                # luminance = torch.log(torch.mean(exp_envmap, dim=-1))
                # prior_luminance = torch.log(torch.mean(exp_prior_envmap, dim=-1))
                # shift = compute_shift(luminance.detach(), prior_luminance)
                # shift = compute_shift(envmap.detach(), prior_envmap)
                # aligned_envmap = envmap + shift
                # loss = F.mse_loss(aligned_envmap, prior_envmap, reduction='mean')
                
                # # cross entropy loss
                # # assume envmap and prior_envmap are both exponentially mapped
                # envmap = envmap / (envmap.sum() + 1e-6)
                # prior_envmap = prior_envmap / (prior_envmap.sum() + 1e-6)
                # loss = -(prior_envmap * torch.log(envmap + 1e-6)).mean()
                
                # # per-channel cross entropy loss
                # # assume envmap and prior_envmap are both exponentially mapped
                # loss = 0.0
                # for i in range(envmap.shape[-1]):
                #     envmap_channel = envmap[..., i]
                #     prior_envmap_channel = prior_envmap[..., i]
                #     envmap_channel = envmap_channel / (envmap_channel.sum() + 1e-6)
                #     prior_envmap_channel = prior_envmap_channel / (prior_envmap_channel.sum() + 1e-6)
                #     loss += -(prior_envmap_channel * torch.log(envmap_channel + 1e-6)).mean()
                # loss = loss / envmap.shape[-1]
                
                # # luminance CE with weight
                # # assume envmap and prior_envmap are both exponentially mapped
                # from models.modules import pixel_grid
                # Y = pixel_grid(envmap.shape[1], envmap.shape[0])[..., 1]
                # delta_angle = torch.sin(np.pi * Y)
                # # envmap_luminance = 0.2126 * envmap[..., 0] + 0.7152 * envmap[..., 1] + 0.0722 * envmap[..., 2]
                # envmap_luminance = torch.mean(envmap, dim=-1)
                # normalized_envmap_luminance = envmap_luminance / ((envmap_luminance * delta_angle).sum() + 1e-12)
                # # prior_envmap_luminance = 0.2126 * prior_envmap[..., 0] + 0.7152 * prior_envmap[..., 1] + 0.0722 * prior_envmap[..., 2]
                # prior_envmap_luminance = torch.mean(prior_envmap, dim=-1)
                # normalized_prior_envmap_luminance = prior_envmap_luminance / ((prior_envmap_luminance * delta_angle).sum() + 1e-12)
                # loss = -(normalized_prior_envmap_luminance * torch.log(normalized_envmap_luminance) * delta_angle).sum()
                
                # # shift-invariant with weight
                # # assume envmap and prior_envmap are both log mapped
                # from models.modules import pixel_grid
                # Y = pixel_grid(envmap.shape[1], envmap.shape[0])[..., 1]
                # exp_prior_envmap = torch.exp(prior_envmap)
                # luminance = torch.max(exp_prior_envmap, dim=-1)[0] * torch.sin(np.pi * Y)
                
                # # get 90-th percentile
                # thres = torch.quantile(luminance, 0.90)
                # tau = 1.0
                # w_hi = 1.0
                # w_lo = 0.0001
                # thres_mask = torch.sigmoid((luminance - thres) / tau)
                # perc_weight = w_hi - (w_hi - w_lo) * thres_mask
                
                # weight = (1 / (luminance + 1e-12))
                # weight = weight * perc_weight
                # weight = weight / torch.max(weight)
                
                # shift = compute_shift(envmap.detach(), prior_envmap)
                # aligned_envmap = envmap + shift
                # loss = (weight[..., None] * ((aligned_envmap - prior_envmap) ** 2)).mean()
                
                # MSE
                loss = F.mse_loss(envmap, prior_envmap, reduction='mean')

                # # weighted MSE: weight = 1 / (exp(prior_envmap.mean(-1)) + c)
                # weight = compute_adaptive_weight_map(torch.exp(prior_envmap), lambda_param=lambda_param, k=k)
                # loss = (weight[..., None] * ((envmap - prior_envmap) ** 2)).mean()
                # # envmap_luminance = envmap.mean(dim=-1) # just use simple mean for now. We can also try other luminance calculations like 0.2126 * R + 0.7152 * G + 0.0722 * B
                # # prior_envmap_luminance = prior_envmap.mean(dim=-1)
                # # loss = (weight * ((envmap_luminance - prior_envmap_luminance) ** 2)).mean()
                
                return loss
            env_scaleinv_loss_fn = scaleinv_loss
        self.env_scaleinv_loss_fn = env_scaleinv_loss_fn
        
        env_color_loss_fn = None
        env_color_loss_cfg = self.losses_dict.get("env_color", None)
        if env_color_loss_cfg is not None:
            # regularize the color of the envmap to be the same as the prior
            # using cosine similarity
            def color_loss(envmap, prior_envmap):
                envmap_norm = torch.norm(envmap, dim=-1) # (H, W)
                prior_envmap_norm = torch.norm(prior_envmap, dim=-1) # (H, W)
                dot_product = (envmap * prior_envmap).sum(dim=-1) # (H, W)
                cos_sim = dot_product / (envmap_norm * prior_envmap_norm + 1e-6) # (H, W)
                loss = (1 - cos_sim).mean()
                return loss
            env_color_loss_fn = color_loss
        self.env_color_loss_fn = env_color_loss_fn

        env_sparse_loss_fn = None
        env_sparse_loss_cfg = self.losses_dict.get("env_sparse", None)
        if env_sparse_loss_cfg is not None:
            def sparse_loss(envmap, prior_envmap): # here envmap and prior_envmap is in linear space, not in log space
                envmap_luminance = envmap.mean(dim=-1) # (H, W)
                lambda_param = 1.5
                k = 10.0
                weight = compute_adaptive_weight_map(prior_envmap, lambda_param=lambda_param, k=k)
                # binarize the weight
                bin_weight = torch.where(weight > 0.5, torch.ones_like(weight), torch.zeros_like(weight))
                strong_light_mask = 1 - bin_weight # only regularize the strong light regions
                power = 0.5
                loss = (envmap_luminance * strong_light_mask + 1e-6).pow(power).mean()
                return loss
            env_sparse_loss_fn = sparse_loss
        self.env_sparse_loss_fn = env_sparse_loss_fn

        diffuse_light_white_loss_fn = None
        diffuse_light_white_loss_cfg = self.losses_dict.get("diffuse_light_white", None)
        if diffuse_light_white_loss_cfg is not None:
            def diffuse_light_white_loss(diffuse_light):
                avg = diffuse_light.mean(dim=-1, keepdim=True)
                loss = ((diffuse_light - avg) ** 2).mean()
                return loss
            diffuse_light_white_loss_fn = diffuse_light_white_loss
        self.diffuse_light_white_loss_fn = diffuse_light_white_loss_fn

        albedo_tv_loss_fn = None
        albedo_tv_loss_cfg = self.losses_dict.get("albedo_tv", None)
        if albedo_tv_loss_cfg is not None:
            from models.losses import AlbedoTVLoss
            albedo_tv_loss_fn = AlbedoTVLoss()
        self.albedo_tv_loss_fn = albedo_tv_loss_fn
        
        roughness_tv_loss_fn = None
        roughness_tv_loss_cfg = self.losses_dict.get("roughness_tv", None)
        if roughness_tv_loss_cfg is not None:
            from models.losses import RoughnessTVLoss
            roughness_tv_loss_fn = RoughnessTVLoss()
        self.roughness_tv_loss_fn = roughness_tv_loss_fn
        
        metallic_tv_loss_fn = None
        metallic_tv_loss_cfg = self.losses_dict.get("metallic_tv", None)
        if metallic_tv_loss_cfg is not None:
            from models.losses import MetallicTVLoss
            metallic_tv_loss_fn = MetallicTVLoss()
        self.metallic_tv_loss_fn = metallic_tv_loss_fn
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',        
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict}
        
        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = [] 
        
        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            empty = False
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                
                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )
                
                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"]
                )
                
            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )
            
            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )
            
            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )
                
            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")
        
        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")
        
        
        logger.info(f"Initialized gaussians from pcd")
    
    def postprocess_per_train_step(self, step: int, timing: bool = False) -> None:
        super().postprocess_per_train_step(step, timing=timing)

        sky_timer = OpTimer(enabled=timing, use_cuda_sync=False, prefix="train post sky")
        def time_cuda(name, fn):
            if timing and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end.record()
                torch.cuda.synchronize()
                sky_timer.add_time(name, start.elapsed_time(end) / 1000.0)
                return out
            with sky_timer.time_block(name):
                return fn()

        if self.models['Sky'].forward_mode != "pure_env":
            time_cuda("build_mips", lambda: self.models['Sky'].build_mips())
        time_cuda("update_pdf", lambda: self.models['Sky'].update_pdf())
        sky_timer.log()
    
    def trace_visibility(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        depth: torch.Tensor, # (H, W)
        normal: torch.Tensor, # (H, W, 3)
        viewdirs: torch.Tensor, # (H, W, 3) 
        roughness: torch.Tensor, # (H, W, 1)
        metallic: torch.Tensor, # (H, W, 1)
        is_train: bool = False,
        rebuild: bool = True,
        frame_id: int = 0,
        num_sample_rays: int = 32, # number of sampling rays for each point
        random_rotate: bool = True, # whether to disturb the sampling directions
        sample_type: str = "uniform", # sampling type, can be 'uniform', 'cosine', 'GGX', 'BRDF'
        mask: torch.Tensor = None, # (H, W), mask for the points to be sampled
        skip_delta: float = None, # skip delta for the ray tracing
        use_metallic: bool = True, # whether to use metallic
        use_vndf: bool = False, # whether to use vndf sampling (GGX sampling)
        envmap: EnvLight_EQ = None, # EnvLight or EnvLight_EQ, for GGX envmap sampling
        timing: bool = False,
    ):
        timer = OpTimer(enabled=timing, use_cuda_sync=False, prefix="trace_visibility")
        def time_cuda(name, fn):
            if timing and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end.record()
                torch.cuda.synchronize()
                timer.add_time(name, start.elapsed_time(end) / 1000.0)
                return out
            with timer.time_block(name):
                return fn()

        # 1. deproject the depth to get the 3d points
        # 2. compute the sampling directions with pdfs
        # 3. build acc if needed
        # 4. build gpu_batch 
        # 5. raytrace to get the visibility
        
        # deproject the depth to get 3d points 
        deprojected_points = time_cuda(
            "deproject_depth",
            lambda: deproject_depth(
                depth_map=depth,
                K=cam.Ks,
                c2w=cam.camtoworlds,
                img_width=cam.W,
                img_height=cam.H,
                device=self.device,
            ).reshape(cam.H, cam.W, 3),
        ) # (H, W, 3)
        
        if mask is None:
            mask = torch.ones_like(depth, dtype=torch.float32)
        
        mask = mask.bool()
        
        deprojected_points = time_cuda("mask_points", lambda: deprojected_points[mask]) # (N, 3)
        normal = time_cuda("mask_normal", lambda: normal[mask]) # (N, 3)
        viewdirs = time_cuda("mask_viewdirs", lambda: viewdirs[mask]) # (N, 3)
        roughness = time_cuda("mask_roughness", lambda: roughness[mask]) # (N, 1)
        metallic = time_cuda("mask_metallic", lambda: metallic[mask]) # (N, 1)
        N = deprojected_points.shape[0]
        
        # compute the sampling directions
        diffuse_mask = None
        if sample_type == "uniform":
            sampling_dirs, pdfs = time_cuda(
                "sample_uniform",
                lambda: fibonacci_sphere_sampling(
                    normals=normal,
                    sample_num=num_sample_rays,
                    random_rotate=random_rotate,
                ),
            ) # (N, num_sample_rays, 3), (N, num_sample_rays)
        elif sample_type == "cosine":
            sampling_dirs, pdfs = time_cuda(
                "sample_cosine",
                lambda: sample_cosine(
                    normals=normal,
                    num_sample_rays=num_sample_rays,
                    device=self.device,
                ),
            ) # (N, num_sample_rays, 3), (N, num_sample_rays)
        elif sample_type == "GGX":
            sampling_dirs, pdfs = time_cuda(
                "sample_GGX",
                lambda: sample_GGX(
                    normals=normal,
                    roughness=roughness.reshape(-1),
                    viewdirs=-viewdirs, # make sure the viewdirs is inward of the camera
                    num=num_sample_rays,
                    device=self.device,
                    use_vndf=use_vndf,
                ),
            ) # (N, num_sample_rays, 3), (N, num_sample_rays)
        elif sample_type == "BRDF":
            if use_metallic:
                sampling_dirs, pdfs, diffuse_mask = time_cuda(
                    "sample_BRDF",
                    lambda: sample_BRDF(
                        normals=normal,
                        viewdirs=-viewdirs, # make sure the viewdirs is inward of the camera
                        roughness=roughness.reshape(-1),
                        diffuse_weight=(1-metallic.reshape(-1)),
                        specular_weight=torch.ones_like(metallic.reshape(-1), device=self.device),
                        num_sample_rays=num_sample_rays,
                        device=self.device,
                        use_vndf=use_vndf,
                    ),
                ) # (N, num_sample_rays, 3), (N, num_sample_rays), (N, num_sample_rays)
            else:
                eq_weight = 0.5 * torch.ones_like(metallic, device=self.device).reshape(-1)
                sampling_dirs, pdfs, diffuse_mask = time_cuda(
                    "sample_BRDF",
                    lambda: sample_BRDF(
                        normals=normal,
                        viewdirs=-viewdirs, # make sure the viewdirs is inward of the camera
                        roughness=roughness.reshape(-1),
                        diffuse_weight=eq_weight,
                        specular_weight=eq_weight,
                        num_sample_rays=num_sample_rays,
                        device=self.device,
                        use_vndf=use_vndf,
                    ),
                ) # (N, num_sample_rays, 3), (N, num_sample_rays), (N, num_sample_rays)
        elif sample_type == "envmap":
            if envmap is None:
                raise ValueError("envmap must be provided for envmap sampling")
            sampling_dirs, pdfs = time_cuda(
                "sample_envmap",
                lambda: sample_envmap(
                    normals=normal,
                    envmap=envmap,
                    num=num_sample_rays,
                    device=self.device,
                    training=is_train,
                    pdf_differentiable=False,
                ),
            )
            
        elif sample_type == "GGX_envmap":
            if envmap is None:
                raise ValueError("envmap must be provided for GGX_envmap sampling")
            sampling_dirs, pdfs, GGX_mask = time_cuda(
                "sample_GGX_envmap",
                lambda: sample_GGX_envmap(
                    normals=normal,
                    viewdirs=-viewdirs, # make sure the viewdirs is inward of the camera
                    roughness=roughness.reshape(-1),
                    envmap=envmap,
                    GGX_weight=0.5,
                    envmap_weight=0.5,
                    num_sample_rays=num_sample_rays,
                    device=self.device,
                    use_vndf=use_vndf,
                ),
            )
        elif sample_type == "uniform_envmap":
            if envmap is None:
                raise ValueError("envmap must be provided for uniform_envmap sampling")
            sampling_dirs, pdfs, uniform_mask = time_cuda(
                "sample_uniform_envmap",
                lambda: sample_uniform_envmap(
                    normals=normal,
                    envmap=envmap,
                    uniform_weight=0.5,
                    envmap_weight=0.5,
                    num_sample_rays=num_sample_rays,
                    device=self.device,
                    random_rotate=random_rotate,
                ),
            )
        elif sample_type == "envmap_main":
            if envmap is None:
                raise ValueError("envmap must be provided for envmap_main sampling")
            sampling_dirs = envmap.get_main_dir(percentage=0.001) # get the main direction of the envmap, which is the direction with the strongest light

            # print("Debug: using debug_dir")
            # sampling_dirs = self.debug_dir / torch.norm(self.debug_dir) # for debugging purpose

            # expand [3] to (N, 1, 3)
            assert num_sample_rays == 1, "envmap_main sampling only supports num_sample_rays=1"
            sampling_dirs = sampling_dirs.unsqueeze(0).unsqueeze(0).expand(N, num_sample_rays, -1) # (N, num_sample_rays, 3)
            pdfs = torch.ones(N, num_sample_rays, device=self.device) # dummy
        else:
            raise NotImplementedError(f"Sampling type {sample_type} is not supported")
    
        
        # build acc if needed
        if rebuild:
            time_cuda("build_acc", lambda: self.build_acc(gs))
        
        
        # raytrace
        # results: rgb, opacity, distance, normal, hit count, frame timing
        if self.gaussian_2d and False: # currently 2D gaussian ray tracing not supported
            
            rays_d = sampling_dirs # (N, num_sample_rays, 3)
            rays_o = deprojected_points.reshape(N, 1, 3).expand(-1, num_sample_rays, -1) # (N, num_sample_rays, 3)
            # skip a delta from the ray origin along the incident direction
            if skip_delta is not None:
                rays_o = rays_d * skip_delta + rays_o
            
            campos = cam.camtoworlds[:3, 3]
            
            raytrace_result = self.tracer.render_dataclass(
                gaussians=gs,
                rays_o=rays_o,
                rays_d=rays_d,
                features=None, # extra features, not used
                camera_center=campos,
                opacity_mask=None,
            )
            
            # (N, num_sample_rays)
            visibilities = 1 - raytrace_result["alpha"]
            # (N, num_sample_rays, 3)
            ind_lights = raytrace_result["color"]
            
        else:
            # build gpu_batch
            # reshape rays_o and rays_d to (num_sample_rays, N, 1, 3)
            rays_o = time_cuda(
                "rays_o",
                lambda: deprojected_points.reshape(1, N, 1, 3).expand(num_sample_rays, -1, -1, -1),
            ) # (num_sample_rays, N, 1, 3)
            rays_d = time_cuda(
                "rays_d",
                lambda: sampling_dirs.permute(1, 0, 2).reshape(num_sample_rays, N, 1, 3),
            ) # (num_sample_rays, N, 1, 3)
            
            # skip a delta from the ray origin along the incident direction
            if skip_delta is not None:
                rays_o = time_cuda("skip_delta", lambda: rays_d * skip_delta + rays_o)
            T_to_world = time_cuda(
                "T_to_world",
                lambda: torch.eye(4, dtype=rays_o.dtype, device=rays_o.device)[None].expand(num_sample_rays, -1, -1),
            ) # (num_sample_rays, 4, 4)
            gpu_batch = time_cuda("gpu_batch", lambda: Batch(T_to_world=T_to_world, rays_ori=rays_o, rays_dir=rays_d))
            raytrace_result = time_cuda(
                "raytrace",
                lambda: self.tracer.render_dataclass(
                    gaussians=gs,
                    gpu_batch=gpu_batch,
                    train=is_train,
                    frame_id=frame_id, # not sure if this matters
                    opacity_mask=None,
                ),
            )
        
            # (num_sample_rays, N, 1, 1) -> (N, num_sample_rays)
            visibilities = time_cuda(
                "visibilities",
                lambda: 1 - raytrace_result["pred_opacity"].squeeze(-1).squeeze(-1).permute(1, 0),
            )
            # (num_sample_rays, N, 1, 3) -> (N, num_sample_rays, 3)
            ind_lights = time_cuda(
                "ind_lights",
                lambda: raytrace_result["pred_rgb"].squeeze(2).permute(1, 0, 2),
            )
        
        # # debugging, binarize the visibility
        # visibilities[visibilities < 0.5] = 0.0
        # visibilities[visibilities >= 0.5] = 1.0
        
        timer.log()
        return visibilities, sampling_dirs, pdfs, ind_lights, diffuse_mask

    def trace_visibility_subpixel(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        normal: torch.Tensor, # (H, W, 3)
        roughness: torch.Tensor, # (H, W, 1)
        metallic: torch.Tensor, # (H, W, 1)
        is_train: bool = False,
        rebuild: bool = True,
        frame_id: int = 0,
        num_sample_rays: int = 32, # number of sampling rays for each point
        random_rotate: bool = True, # whether to disturb the sampling directions
        sample_type: str = "uniform", # sampling type, can be 'uniform', 'cosine', 'GGX', 'BRDF'
        mask: torch.Tensor = None, # (H, W), mask for the points to be sampled
        precomputed_depth: torch.Tensor = None, # (H*sH, W*sW)
        precomputed_normal: torch.Tensor = None, # (H*sH, W*sW, 3)
        precomputed_roughness: torch.Tensor = None, # (H*sH, W*sW, 1)
        precomputed_metallic: torch.Tensor = None, # (H*sH, W*sW, 1)
        skip_delta: float = None, # skip delta for the ray tracing
        use_metallic: bool = True, # whether to use metallic
        use_vndf: bool = False, # whether to use vndf sampling (GGX sampling)
        envmap: EnvLight_EQ = None, # EnvLight or EnvLight_EQ, for GGX envmap sampling
    ):
        
        # sample num_sample_rays subpixel in each pixel
        # trace the depth of them, and sample one direction for each subpixel
        # share the same normal and material for each pixel
        
        viewdirs, sH, sW = generate_subpixel_rays(
            c2w=cam.camtoworlds,
            K=cam.Ks,
            H=cam.H,
            W=cam.W,
            num_sample_rays=num_sample_rays,
            device=self.device,
        )# (H, W, num_sample_rays, 3)
        viewdirs = viewdirs.reshape(cam.H, cam.W, -1)
        
        # build acc if needed
        if rebuild:
            self.build_acc(gs)
        
        if precomputed_depth is not None:
            depth = precomputed_depth
        else:
            if self.gaussian_2d:
                raise NotImplementedError("Subpixel tracing not implemented for 2D gaussians")
            else:
                ray_o = cam.camtoworlds[:3, 3]
                rays_o = ray_o.reshape(1, 1, 1, 3).expand(num_sample_rays, cam.H, cam.W, -1) 
                rays_d = viewdirs.reshape(cam.H, cam.W, num_sample_rays, 3).permute(2, 0, 1, 3) # (num_sample_rays, H, W, 3)
                T_to_world = torch.eye(4, dtype=rays_o.dtype, device=rays_o.device)[None].expand(num_sample_rays, -1, -1)
                gpu_batch = Batch(T_to_world=T_to_world, rays_ori=rays_o, rays_dir=rays_d)
                
                raytrace_result = self.tracer.render_dataclass(
                    gaussians=gs,
                    gpu_batch=gpu_batch,
                    train=is_train,
                    frame_id=frame_id,
                    opacity_mask=None,
                )
                
                depth = raytrace_result["pred_dist"].squeeze(-1).permute(1, 2, 0)
                depth = depth.reshape(cam.H, cam.W, sH, sW).permute(0, 2, 1, 3).reshape(cam.H*sH, cam.W*sW)
        
        # deproject the depth to get 3d points 
        deprojected_points = deproject_depth(
            depth_map=depth,
            K=cam.Ks,
            c2w=cam.camtoworlds,
            img_width=cam.W*sW,
            img_height=cam.H*sH,
            device=self.device,
        ) # (H*sH*W*sW, 3)
        
        # (H, W, num_sample_rays*3)
        deprojected_points = deprojected_points.reshape(cam.H, sH, cam.W, sW, 3).permute(0, 2, 1, 3, 4).reshape(cam.H, cam.W, num_sample_rays*3)
        
        if mask is None:
            mask = torch.ones(cam.H, cam.W, dtype=torch.float32)
        
        mask = mask.bool()
        
        deprojected_points = deprojected_points[mask] # (N, num_sample_rays*3)
        viewdirs = viewdirs[mask].reshape(-1, num_sample_rays, 3) # (N, num_sample_rays, 3)
        if precomputed_normal is not None:
            precomputed_normal = precomputed_normal.reshape(cam.H, sH, cam.W, sW, 3).permute(0, 2, 1, 3, 4).reshape(cam.H, cam.W, -1)
            normal = precomputed_normal[mask].reshape(-1, num_sample_rays, 3) # (N, num_sample_rays, 3)
        else:
            # share the normal for all rays in the same pixel
            normal = normal[mask].reshape(-1, 1, 3).expand(-1, num_sample_rays, 3) # (N, num_sample_rays, 3)
        
        if precomputed_roughness is not None:
            precomputed_roughness = precomputed_roughness.reshape(cam.H, sH, cam.W, sW, 1).permute(0, 2, 1, 3, 4).reshape(cam.H, cam.W, -1)
            roughness = precomputed_roughness[mask].reshape(-1, num_sample_rays, 1)
        else:
            # share the roughness for all rays in the same pixel
            roughness = roughness[mask].reshape(-1, 1, 1).expand(-1, num_sample_rays, 1)
        
        if precomputed_metallic is not None:
            precomputed_metallic = precomputed_metallic.reshape(cam.H, sH, cam.W, sW, 1).permute(0, 2, 1, 3, 4).reshape(cam.H, cam.W, -1)
            metallic = precomputed_metallic[mask].reshape(-1, num_sample_rays, 1)
        else:
            # share the metallic for all rays in the same pixel
            metallic = metallic[mask].reshape(-1, 1, 1).expand(-1, num_sample_rays, 1)
        
        N = deprojected_points.shape[0]
        
        # compute the sampling directions
        diffuse_mask = None
        # NOTE: share the same normal and material for each ray in the same pixel for now
        if sample_type == "uniform":
            sampling_dirs, pdfs = fibonacci_sphere_sampling(
                normals=normal.reshape(-1, 3),
                sample_num=1,
                random_rotate=random_rotate,
            ) # [N*num_sample_rays, 1, 3], [N*num_sample_rays, 1]
        elif sample_type == "cosine":
            sampling_dirs, pdfs = sample_cosine(
                normals=normal.reshape(-1, 3),
                num_sample_rays=1,
                device=self.device,
            ) # [N*num_sample_rays, 1, 3], [N*num_sample_rays, 1]
        elif sample_type == "GGX":
            sampling_dirs, pdfs = sample_GGX(
                normals=normal.reshape(-1, 3),
                roughness=roughness.reshape(-1),
                viewdirs=-viewdirs.reshape(-1, 3), # make sure the viewdirs is inward of the camera
                num=1,
                device=self.device,
                use_vndf=use_vndf,
            ) # [N*num_sample_rays, 1, 3], [N*num_sample_rays, 1]
        elif sample_type == "BRDF":
            if use_metallic:
                sampling_dirs, pdfs, diffuse_mask = sample_BRDF(
                    normals=normal.reshape(-1, 3),
                    viewdirs=-viewdirs.reshape(-1, 3), # make sure the viewdirs is inward of the camera
                    roughness=roughness.reshape(-1),
                    diffuse_weight=(1-metallic.reshape(-1)),
                    specular_weight=torch.ones_like(metallic.reshape(-1), device=self.device),
                    num_sample_rays=1,
                    device=self.device,
                    use_vndf=use_vndf,
                ) # [N*num_sample_rays, 1, 3], [N*num_sample_rays, 1]
            else:
                eq_weight = 0.5 * torch.ones_like(metallic, device=self.device).reshape(-1)
                sampling_dirs, pdfs, diffuse_mask = sample_BRDF(
                    normals=normal.reshape(-1, 3),
                    viewdirs=-viewdirs.reshape(-1, 3), # make sure the viewdirs is inward of the camera
                    roughness=roughness.reshape(-1),
                    diffuse_weight=eq_weight,
                    specular_weight=eq_weight,
                    num_sample_rays=1,
                    device=self.device,
                    use_vndf=use_vndf,
                )
        elif sample_type == "envmap":
            if envmap is None:
                raise ValueError("envmap must be provided for envmap sampling")
            sampling_dirs, pdfs = sample_envmap(
                normals=normal.reshape(-1, 3),
                envmap=envmap,
                num=1,
                device=self.device,
                training=is_train,
            )
        elif sample_type == "GGX_envmap":
            if envmap is None:
                raise ValueError("envmap must be provided for GGX_envmap sampling")
            sampling_dirs, pdfs, GGX_mask = sample_GGX_envmap(
                normals=normal.reshape(-1, 3),
                viewdirs=-viewdirs.reshape(-1, 3), # make sure the viewdirs is inward of the camera
                roughness=roughness.reshape(-1),
                envmap=envmap,
                GGX_weight=0.5,
                envmap_weight=0.5,
                num_sample_rays=1,
                device=self.device,
                use_vndf=use_vndf,
            )
        elif sample_type == "uniform_envmap":
            if envmap is None:
                raise ValueError("envmap must be provided for uniform_envmap sampling")
            sampling_dirs, pdfs, uniform_mask = sample_uniform_envmap(
                normals=normal.reshape(-1, 3),
                envmap=envmap,
                uniform_weight=0.5,
                envmap_weight=0.5,
                num_sample_rays=1,
                device=self.device,
                random_rotate=random_rotate,
            )
        else:
            raise NotImplementedError(f"Sampling type {sample_type} is not supported")
        
        sampling_dirs = sampling_dirs.reshape(N, num_sample_rays, 3)
        pdfs = pdfs.reshape(N, num_sample_rays)
        
        if self.gaussian_2d:
            raise NotImplementedError("Subpixel tracing not implemented for 2D gaussians")
        else:
            # build gpu_batch
            # reshape rays_o and rays_d to (num_sample_rays, N, 1, 3)
            rays_o = deprojected_points.reshape(N, num_sample_rays, 3).permute(1, 0, 2).reshape(num_sample_rays, N, 1, 3)
            rays_d = sampling_dirs.permute(1, 0, 2).reshape(num_sample_rays, N, 1, 3) # (num_sample_rays, N, 1, 3)
            
            # skip a delta from the ray origin along the incident direction
            if skip_delta is not None:
                rays_o = rays_d * skip_delta + rays_o
            
            T_to_world = torch.eye(4, dtype=rays_o.dtype, device=rays_o.device)[None].expand(num_sample_rays, -1, -1) # (num_sample_rays, 4, 4)
            gpu_batch = Batch(T_to_world=T_to_world, rays_ori=rays_o, rays_dir=rays_d)
            
            # raytrace
            # results: rgb, opacity, distance, normal, hit count, frame timing
            raytrace_result = self.tracer.render_dataclass(
                gaussians=gs,
                gpu_batch=gpu_batch,
                train=is_train,
                frame_id=frame_id, # not sure if this matters
                opacity_mask=None,
            )
            
            # (num_sample_rays, N, 1, 1) -> (N, num_sample_rays)
            visibilities = 1 - raytrace_result["pred_opacity"].squeeze(-1).squeeze(-1).permute(1, 0)
            # (num_sample_rays, N, 1, 3) -> (N, num_sample_rays, 3)
            ind_lights = raytrace_result["pred_rgb"].squeeze(2).permute(1, 0, 2)
        
        return visibilities, sampling_dirs, pdfs, ind_lights, diffuse_mask

    def _load_point_lights(self):
        point_lights_cfg = self.tracer_cfg.get("point_lights", None) if self.tracer_cfg is not None else None
        all_point_lights = []
        if point_lights_cfg is not None and len(point_lights_cfg) > 0:
            all_point_lights.extend(list(point_lights_cfg))
        all_point_lights.extend(self._load_point_lights_from_file())
        if len(all_point_lights) == 0:
            return None, None, None, None, None, None

        positions, intensities = [], []
        spot_dirs, spot_inner_cos, spot_outer_cos, spot_enabled = [], [], [], []
        for i, light in enumerate(all_point_lights):
            if isinstance(light, str):
                try:
                    import ast
                    light = ast.literal_eval(light)
                except Exception:
                    pass

            # Support both plain Python containers and OmegaConf ListConfig/DictConfig.
            if isinstance(light, dict) or hasattr(light, "keys"):
                def get_any(keys, default=None):
                    for k in keys:
                        if k in light and light.get(k) is not None:
                            return light.get(k)
                    return default
                pos = light.get("position", None)
                inten = light.get("intensity", None)
                spot_dir = get_any(["direction", "spot_direction", "dir"], None)
                inner_angle_deg = get_any(["inner_angle_deg", "inner_angle", "spot_inner_angle_deg"], 15.0)
                outer_angle_deg = get_any(["outer_angle_deg", "outer_angle", "spot_outer_angle_deg"], 25.0)
                # Fallback for odd CLI parse cases where lookup returns None but text still contains values.
                if spot_dir is None or inner_angle_deg is None or outer_angle_deg is None:
                    light_text = str(light)
                    if spot_dir is None:
                        m_dir = re.search(r"(?:direction|spot_direction|dir)\s*[:=]\s*\[([^\]]+)\]", light_text)
                        if m_dir is not None:
                            try:
                                spot_dir = [float(x.strip()) for x in m_dir.group(1).split(",")]
                            except Exception:
                                spot_dir = None
                    if inner_angle_deg is None:
                        m_in = re.search(
                            r"(?:inner_angle_deg|inner_angle|spot_inner_angle_deg)\s*[:=]\s*([-+]?[\d\.eE]+)",
                            light_text,
                        )
                        if m_in is not None:
                            inner_angle_deg = float(m_in.group(1))
                    if outer_angle_deg is None:
                        m_out = re.search(
                            r"(?:outer_angle_deg|outer_angle|spot_outer_angle_deg)\s*[:=]\s*([-+]?[\d\.eE]+)",
                            light_text,
                        )
                        if m_out is not None:
                            outer_angle_deg = float(m_out.group(1))
            else:
                light_seq = list(light)
                if len(light_seq) == 2:
                    pos, inten = light_seq[0], light_seq[1]
                elif len(light_seq) == 6:
                    pos, inten = light_seq[:3], light_seq[3:]
                else:
                    pos, inten = None, None
                spot_dir = None
                inner_angle_deg = 15.0
                outer_angle_deg = 25.0
            if pos is None or inten is None:
                raise ValueError(f"Invalid point light at index {i}. Expected position and intensity.")
            positions.append(list(pos))
            intensities.append(list(inten))
            if spot_dir is None:
                spot_enabled.append(False)
                spot_dirs.append([0.0, 0.0, 1.0])
                spot_inner_cos.append(1.0)
                spot_outer_cos.append(1.0)
            else:
                spot_enabled.append(True)
                spot_dir = torch.as_tensor(spot_dir, dtype=torch.float32, device=self.device)
                if spot_dir.numel() != 3:
                    raise ValueError(f"Invalid spotlight direction at index {i}. Expected 3D direction.")
                spot_dir = spot_dir / spot_dir.norm().clamp_min(1e-6)
                spot_dirs.append(spot_dir.tolist())
                inner_rad = float(inner_angle_deg) * math.pi / 180.0
                outer_rad = float(outer_angle_deg) * math.pi / 180.0
                if outer_rad < inner_rad:
                    outer_rad = inner_rad
                spot_inner_cos.append(float(math.cos(inner_rad)))
                spot_outer_cos.append(float(math.cos(outer_rad)))

        point_light_positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        point_light_intensities = torch.as_tensor(intensities, dtype=torch.float32, device=self.device)
        if point_light_positions.ndim != 2 or point_light_positions.shape[-1] != 3:
            raise ValueError("point_lights positions must have shape [L, 3].")
        if point_light_intensities.ndim != 2 or point_light_intensities.shape[-1] != 3:
            raise ValueError("point_lights intensities must have shape [L, 3].")
        point_light_spot_dirs = torch.as_tensor(spot_dirs, dtype=torch.float32, device=self.device)
        point_light_spot_inner_cos = torch.as_tensor(spot_inner_cos, dtype=torch.float32, device=self.device)
        point_light_spot_outer_cos = torch.as_tensor(spot_outer_cos, dtype=torch.float32, device=self.device)
        point_light_spot_enabled = torch.as_tensor(spot_enabled, dtype=torch.bool, device=self.device)
        return (
            point_light_positions,
            point_light_intensities,
            point_light_spot_dirs,
            point_light_spot_inner_cos,
            point_light_spot_outer_cos,
            point_light_spot_enabled,
        )

    def _load_point_lights_from_file(self):
        if self.tracer_cfg is None:
            return []
        light_file = self.tracer_cfg.get("point_lights_file", None)
        if light_file is None or str(light_file).strip() == "":
            return []
        if not os.path.exists(light_file):
            raise FileNotFoundError(f"point_lights_file not found: {light_file}")

        file_format = str(self.tracer_cfg.get("point_lights_file_format", "blender_area_txt")).lower()
        if file_format != "blender_area_txt":
            raise ValueError(f"Unsupported point_lights_file_format: {file_format}")

        energy_scale = float(self.tracer_cfg.get("point_lights_file_energy_scale", 1.0))
        spread_angle_deg = float(self.tracer_cfg.get("point_lights_file_spread_angle_deg", 180.0))
        spread_angle_deg = max(0.0, min(180.0, spread_angle_deg))
        inner_ratio = float(self.tracer_cfg.get("point_lights_file_inner_ratio", 1.0))
        inner_ratio = max(0.0, min(1.0, inner_ratio))
        outer_half_angle_deg = 0.5 * spread_angle_deg
        inner_half_angle_deg = inner_ratio * outer_half_angle_deg

        axis_name = str(self.tracer_cfg.get("point_lights_file_direction_axis", "z")).lower()
        axis_sign = float(self.tracer_cfg.get("point_lights_file_direction_sign", 1.0))
        axis_map = {
            "x": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            "y": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            "z": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device),
        }
        if axis_name not in axis_map:
            raise ValueError("point_lights_file_direction_axis must be one of {'x','y','z'}")
        local_axis = axis_map[axis_name] * axis_sign
        local_axis = local_axis / local_axis.norm().clamp_min(1e-6)

        # Optional re-anchoring: convert absolute light poses into dataset-anchored world frame.
        # Priority:
        #   1) explicit 4x4 transform via point_lights_file_reanchor_transform
        #   2) derive from first valid row in point_lights_file_reanchor_pose_path (T = inv(ref_pose))
        apply_reanchor = bool(self.tracer_cfg.get("point_lights_file_reanchor", False))
        reanchor_transform = None
        if apply_reanchor:
            transform_cfg = self.tracer_cfg.get("point_lights_file_reanchor_transform", None)
            pose_path_cfg = self.tracer_cfg.get("point_lights_file_reanchor_pose_path", None)
            if transform_cfg is not None:
                t = torch.as_tensor(transform_cfg, dtype=torch.float32, device=self.device)
                if t.numel() == 16:
                    t = t.reshape(4, 4)
                if t.shape != (4, 4):
                    raise ValueError(
                        "point_lights_file_reanchor_transform must be 16 values or shape [4,4]."
                    )
                reanchor_transform = t
            elif pose_path_cfg is not None and str(pose_path_cfg).strip() != "":
                pose_path = str(pose_path_cfg)
                if not os.path.exists(pose_path):
                    raise FileNotFoundError(f"point_lights_file_reanchor_pose_path not found: {pose_path}")
                # Keep re-anchoring consistent with self dataset loader:
                # use the same parser (frame dedup + quaternion convention).
                from datasets.self.self_sourceloader import _load_pose_file

                _, ref_poses = _load_pose_file(pose_path)
                if len(ref_poses) == 0:
                    raise ValueError(f"No valid pose row found in {pose_path}")
                ref_pose = torch.from_numpy(ref_poses[0]).to(
                    device=self.device, dtype=torch.float32
                )
                reanchor_transform = torch.linalg.inv(ref_pose)
            else:
                raise ValueError(
                    "point_lights_file_reanchor=true requires either "
                    "point_lights_file_reanchor_transform or point_lights_file_reanchor_pose_path."
                )
            logger.info("Applying point lights re-anchor transform from absolute frame to dataset frame.")

        parsed_lights = []
        with open(light_file, "r", encoding="utf-8") as f:
            for line_idx, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 11:
                    raise ValueError(
                        f"Invalid light line at {light_file}:{line_idx}. "
                        "Expected format: name tx ty tz qx qy qz qw energy size_x size_y"
                    )
                _, tx, ty, tz, qx, qy, qz, qw, energy, *_ = parts[:11]
                tx, ty, tz = float(tx), float(ty), float(tz)
                qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
                energy = float(energy)

                quat_wxyz = torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32, device=self.device)
                rot = quat_to_rotmat(quat_wxyz)[0]
                spot_dir = rot @ local_axis
                spot_dir = spot_dir / spot_dir.norm().clamp_min(1e-6)
                if reanchor_transform is not None:
                    pos_h = torch.tensor([tx, ty, tz, 1.0], dtype=torch.float32, device=self.device)
                    pos_reanchored = reanchor_transform @ pos_h
                    tx, ty, tz = (
                        float(pos_reanchored[0].item()),
                        float(pos_reanchored[1].item()),
                        float(pos_reanchored[2].item()),
                    )
                    spot_dir = reanchor_transform[:3, :3] @ spot_dir
                    spot_dir = spot_dir / spot_dir.norm().clamp_min(1e-6)
                intensity = max(0.0, energy * energy_scale)

                parsed_lights.append(
                    {
                        "position": [tx, ty, tz],
                        "intensity": [intensity, intensity, intensity],
                        "direction": spot_dir.tolist(),
                        "inner_angle_deg": inner_half_angle_deg,
                        "outer_angle_deg": outer_half_angle_deg,
                    }
                )
        
        
        return parsed_lights

    def _build_eval_headlights(self, cam: dataclass_camera):
        # Two simple camera-relative headlights: left/right, slightly below camera.
        c2w = cam.camtoworlds
        cam_pos = c2w[:3, 3]
        cam_right = c2w[:3, 0]
        cam_up = c2w[:3, 1]
        cam_forward = c2w[:3, 2]

        lateral = float(self.tracer_cfg.get("eval_headlight_lateral_offset", 0.35))
        down = float(self.tracer_cfg.get("eval_headlight_down_offset", 0.20))
        forward = float(self.tracer_cfg.get("eval_headlight_forward_offset", 0.20))
        intensity = float(self.tracer_cfg.get("eval_headlight_intensity", 200.0))

        base = cam_pos + cam_forward * forward - cam_up * down
        left_pos = base - cam_right * lateral
        right_pos = base + cam_right * lateral
        positions = torch.stack([left_pos, right_pos], dim=0).to(self.device)
        intensities = torch.full((2, 3), intensity, dtype=torch.float32, device=self.device)
        return positions, intensities

    def _build_eval_headlight_spot_params(self, cam: dataclass_camera, num_lights: int):
        """Build optional spotlight parameters for eval headlights (default disabled)."""
        enabled = bool(self.tracer_cfg.get("eval_headlight_as_spotlight", False))
        if not enabled:
            dirs = torch.zeros((num_lights, 3), dtype=torch.float32, device=self.device)
            dirs[:, 2] = 1.0
            inner_cos = torch.ones((num_lights,), dtype=torch.float32, device=self.device)
            outer_cos = torch.ones((num_lights,), dtype=torch.float32, device=self.device)
            mask = torch.zeros((num_lights,), dtype=torch.bool, device=self.device)
            return dirs, inner_cos, outer_cos, mask

        c2w = cam.camtoworlds
        cam_rot = c2w[:3, :3]
        spot_dir_cam = torch.as_tensor(
            self.tracer_cfg.get("eval_headlight_spot_direction_cam", [0.0, 0.0, 1.0]),
            dtype=torch.float32,
            device=self.device,
        )
        if spot_dir_cam.numel() != 3:
            raise ValueError("eval_headlight_spot_direction_cam must be a 3D vector.")
        spot_dir_cam = spot_dir_cam / spot_dir_cam.norm().clamp_min(1e-6)
        spot_dir_world = cam_rot @ spot_dir_cam
        spot_dir_world = spot_dir_world / spot_dir_world.norm().clamp_min(1e-6)
        dirs = spot_dir_world.reshape(1, 3).expand(num_lights, -1).clone()

        inner_deg = float(self.tracer_cfg.get("eval_headlight_inner_angle_deg", 10.0))
        outer_deg = float(self.tracer_cfg.get("eval_headlight_outer_angle_deg", 20.0))
        inner_rad = inner_deg * math.pi / 180.0
        outer_rad = outer_deg * math.pi / 180.0
        if outer_rad < inner_rad:
            outer_rad = inner_rad
        inner_cos = torch.full((num_lights,), float(math.cos(inner_rad)), dtype=torch.float32, device=self.device)
        outer_cos = torch.full((num_lights,), float(math.cos(outer_rad)), dtype=torch.float32, device=self.device)
        mask = torch.ones((num_lights,), dtype=torch.bool, device=self.device)
        return dirs, inner_cos, outer_cos, mask

    def _build_point_light_emitter_gaussians(
        self,
        base_gs: dataclass_gs,
        point_light_positions: torch.Tensor,
        point_light_intensities: torch.Tensor,
    ) -> dataclass_gs:
        """Build tiny eval-only emitter gaussians for point lights."""
        L = point_light_positions.shape[0]
        dtype = base_gs._means.dtype
        device = self.device

        emitter_radius = float(self.tracer_cfg.get("point_light_emitter_radius", 0.3))
        emitter_opacity = float(self.tracer_cfg.get("point_light_emitter_opacity", 0.98))
        emitter_color_scale = float(self.tracer_cfg.get("point_light_emitter_color_scale", 1.0))

        intensities = point_light_intensities.to(dtype=dtype, device=device).clamp_min(0.0)
        max_channel = intensities.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        emitter_rgbs = (intensities / max_channel) * emitter_color_scale
        emitter_rgbs = emitter_rgbs.clamp(0.0, 1.0)

        emitter_quats = torch.zeros((L, 4), dtype=dtype, device=device)
        emitter_quats[:, 0] = 1.0
        emitter_scales = torch.full((L, 3), emitter_radius, dtype=dtype, device=device)
        emitter_opacities = torch.full((L, 1), emitter_opacity, dtype=dtype, device=device)

        emitter_features_dc = None
        if base_gs._features_dc is not None:
            emitter_features_dc = torch.zeros(
                (L,) + tuple(base_gs._features_dc.shape[1:]),
                dtype=base_gs._features_dc.dtype,
                device=device,
            )
        emitter_features_rest = None
        if base_gs._features_rest is not None:
            emitter_features_rest = torch.zeros(
                (L,) + tuple(base_gs._features_rest.shape[1:]),
                dtype=base_gs._features_rest.dtype,
                device=device,
            )

        emitter_extras = {}
        if base_gs.extras is not None:
            for k, v in base_gs.extras.items():
                emitter_extras[k] = torch.zeros(
                    (L,) + tuple(v.shape[1:]),
                    dtype=v.dtype,
                    device=device,
                )

        return dataclass_gs(
            _means=point_light_positions.to(dtype=dtype, device=device),
            _scales=emitter_scales,
            _quats=emitter_quats,
            _rgbs=emitter_rgbs,
            _opacities=emitter_opacities,
            _features_dc=emitter_features_dc,
            _features_rest=emitter_features_rest,
            detach_keys=[],
            extras=emitter_extras,
        )

    def trace_visibility_point_lights(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        depth: torch.Tensor, # (H, W)
        mask: torch.Tensor, # (H, W)
        point_light_positions: torch.Tensor, # (L, 3)
        is_train: bool = False,
        rebuild: bool = False,
        frame_id: int = 0,
        skip_delta: float = None,
    ):
        deprojected_points = deproject_depth(
            depth_map=depth,
            K=cam.Ks,
            c2w=cam.camtoworlds,
            img_width=cam.W,
            img_height=cam.H,
            device=self.device,
        ).reshape(cam.H, cam.W, 3)

        mask = mask.bool()
        points = deprojected_points[mask]  # (N, 3)
        N = points.shape[0]
        if N == 0:
            L = point_light_positions.shape[0]
            empty = torch.zeros((0, L), device=self.device)
            empty_dirs = torch.zeros((0, L, 3), device=self.device)
            empty_dist = torch.zeros((0, L), device=self.device)
            return empty, empty_dirs, empty_dist

        L = point_light_positions.shape[0]
        light_vecs = point_light_positions[None, :, :] - points[:, None, :]  # (N, L, 3)
        light_dists = torch.norm(light_vecs, dim=-1).clamp_min(1e-6)  # (N, L)
        sampling_dirs = light_vecs / light_dists[..., None]  # (N, L, 3)

        if rebuild:
            self.build_acc(gs, rebuild=True)

        rays_o = points.reshape(1, N, 1, 3).expand(L, -1, -1, -1)  # (L, N, 1, 3)
        rays_d = sampling_dirs.permute(1, 0, 2).reshape(L, N, 1, 3)  # (L, N, 1, 3)
        if skip_delta is not None:
            rays_o = rays_o + rays_d * skip_delta

        T_to_world = torch.eye(4, dtype=rays_o.dtype, device=rays_o.device)[None].expand(L, -1, -1)
        gpu_batch = Batch(T_to_world=T_to_world, rays_ori=rays_o, rays_dir=rays_d)

        raytrace_result = self.tracer.render_dataclass(
            gaussians=gs,
            gpu_batch=gpu_batch,
            train=is_train,
            frame_id=frame_id,
            opacity_mask=None,
        )

        visibilities = 1 - raytrace_result["pred_opacity"].squeeze(-1).squeeze(-1).permute(1, 0)  # (N, L)
        return visibilities, sampling_dirs, light_dists

    def pbr_point_lights(
        self,
        albedos: torch.Tensor, # (H, W, 3)
        roughnesses: torch.Tensor, # (H, W, 1)
        metallics: torch.Tensor, # (H, W, 1)
        normals: torch.Tensor, # (H, W, 3)
        viewdirs: torch.Tensor, # (H, W, 3)
        sampling_dirs: torch.Tensor, # (N, L, 3)
        visibilities: torch.Tensor, # (N, L)
        light_distances: torch.Tensor, # (N, L)
        point_light_intensities: torch.Tensor, # (L, 3)
        point_light_spot_dirs: torch.Tensor = None, # (L, 3)
        point_light_spot_inner_cos: torch.Tensor = None, # (L,)
        point_light_spot_outer_cos: torch.Tensor = None, # (L,)
        point_light_spot_enabled: torch.Tensor = None, # (L,)
        use_metallic: bool = True,
        mask: torch.Tensor = None,
        verbose: bool = False,
        only_diffuse: bool = False,
    ):
        if mask is None:
            mask = torch.ones_like(albedos[..., 0], dtype=torch.float32)
        mask = mask.bool()

        albedos = albedos[mask]  # (N, 3)
        roughnesses = roughnesses[mask]  # (N, 1)
        metallics = metallics[mask]  # (N, 1)
        normals = normals[mask]  # (N, 3)
        viewdirs = viewdirs[mask]  # (N, 3)

        if sampling_dirs.shape[0] == 0:
            empty3 = torch.zeros((0, 3), device=self.device)
            results = {"pbr_color": empty3, "mean_transport": empty3}
            results["dir_light"] = torch.zeros((0, 0, 3), device=self.device)
            if verbose:
                results.update({
                    "dir_light": torch.zeros((0, 0, 3), device=self.device),
                    "n_d_i": torch.zeros((0, 0, 1), device=self.device),
                    "light": torch.zeros((0, 0, 3), device=self.device),
                    "diffuse": empty3,
                    "specular": empty3,
                    "mean_fd": torch.zeros((0, 3), device=self.device),
                    "mean_fs": torch.zeros((0, 3), device=self.device),
                })
            return results

        N, L = sampling_dirs.shape[:2]
        intensity_mode = self.tracer_cfg.get("point_light_intensity_mode", "radiant_intensity")
        falloff_exponent = float(self.tracer_cfg.get("point_light_falloff_exponent", 2.0))
        min_distance = float(self.tracer_cfg.get("point_light_min_distance", 0.0))
        if falloff_exponent < 0.0:
            raise ValueError("point_light_falloff_exponent must be non-negative.")
        if min_distance < 0.0:
            raise ValueError("point_light_min_distance must be non-negative.")
        direct_lights = point_light_intensities[None, :, :].expand(N, L, 3)
        if intensity_mode == "radiant_intensity":
            safe_distances = light_distances.clamp_min(min_distance)
            distance_term = safe_distances[..., None].pow(falloff_exponent).clamp_min(1e-6)
            direct_lights = direct_lights / distance_term
        elif intensity_mode == "irradiance_like":
            pass
        else:
            raise ValueError(f"Unsupported point_light_intensity_mode: {intensity_mode}")

        spot_attenuation = torch.ones((N, L, 1), device=self.device, dtype=direct_lights.dtype)
        if point_light_spot_enabled is not None and point_light_spot_dirs is not None \
            and point_light_spot_inner_cos is not None and point_light_spot_outer_cos is not None:
            enabled = point_light_spot_enabled.reshape(1, L, 1)
            if enabled.any():
                spot_dirs = point_light_spot_dirs / point_light_spot_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                dirs_to_surface = -sampling_dirs  # from light to shaded point
                cos_theta = (dirs_to_surface * spot_dirs.reshape(1, L, 3)).sum(dim=-1, keepdim=True)
                inner = point_light_spot_inner_cos.reshape(1, L, 1)
                outer = point_light_spot_outer_cos.reshape(1, L, 1)
                denom = (inner - outer).clamp_min(1e-6)
                t = ((cos_theta - outer) / denom).clamp(0.0, 1.0)
                smooth = t * t * (3.0 - 2.0 * t)
                step_mask = (inner - outer).abs() < 1e-6
                if step_mask.any():
                    step_val = (cos_theta >= inner).to(smooth.dtype)
                    smooth = torch.where(step_mask.reshape(1, L, 1), step_val, smooth)
                spot_attenuation = torch.where(enabled, smooth, spot_attenuation)

        lights = direct_lights * visibilities[..., None] * spot_attenuation
        n_d_i = (normals[:, None, :] * sampling_dirs).sum(dim=-1, keepdim=True).clamp_min(0.0)  # (N, L, 1)

        f_d = brdf_diffuse(albedos=albedos, lightdirs=sampling_dirs)
        if use_metallic:
            f_d = f_d * (1.0 - metallics[:, None, :])

        if use_metallic:
            f0 = metallics * albedos + (1.0 - metallics) * torch.full_like(albedos, 0.04)
        else:
            f0 = 0.04
        f_s, extra_infos = brdf_GGX(
            normals=normals,
            viewdirs=-viewdirs,
            lightdirs=sampling_dirs,
            roughness=roughnesses.reshape(-1),
            fresnel=f0,
            verbose=verbose,
        )

        transport = lights * n_d_i  # (N, L, 3)
        specular = (f_s * transport).sum(dim=-2)  # (N, 3)
        diffuse = (f_d * transport).sum(dim=-2)  # (N, 3)
        if only_diffuse:
            pbr_color = diffuse
        else:
            pbr_color = ((f_d + f_s) * transport).sum(dim=-2)  # (N, 3)
        mean_transport = transport.mean(dim=-2)  # (N, 3)

        results = {
            "pbr_color": pbr_color,
            "mean_transport": mean_transport,
            "dir_light": direct_lights,
        }
        if verbose:
            results.update({
                "n_d_i": n_d_i,
                "light": lights,
                "diffuse": diffuse,
                "specular": specular,
                "mean_fd": f_d.mean(dim=-2),
                "mean_fs": f_s.mean(dim=-2),
            })
            results.update(extra_infos)
        return results

    
    # doing ray-tracing rendering (just for testing)
    def raytrace_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        rays_o: torch.Tensor, # [H*W, 3]
        rays_d: torch.Tensor, # [H*W, 3]
        is_train: bool = False,
        frame_id: int = 0,
        opacity_mask: torch.Tensor = None,
        rebuild: bool = True, # whether to rebuild the acc
    ) -> Dict[str, torch.Tensor]:
        # TODO: use nvtx range to do profiling
        
        if self.gaussian_2d:
            raise NotImplementedError("Pure raytracing is not implemented for 2D gaussians")
    
        # 1. build acc
        # 2. build gpu_batch (camera)
        # 3. raytrace

        # build acc
        if rebuild:
            self.build_acc(gs)
        
        # build gpu_batch (all parameter )
        # rays_o and rays_d are in world space
        # TBD: pass them (in images_info to this function)
        rays_o = rays_o.reshape(1, cam.H, cam.W, 3)
        rays_d = rays_d.reshape(1, cam.H, cam.W, 3)
        T_to_world = torch.eye(4, dtype=rays_o.dtype, device=rays_o.device)[None]
        gpu_batch = Batch(T_to_world=T_to_world, rays_ori=rays_o, rays_dir=rays_d)
        
        # raytrace
        # results: rgb, opacity, distance, normal, hit count, frame timing
        raytrace_result = self.tracer.render_dataclass(
            gaussians=gs,
            gpu_batch=gpu_batch,
            train=is_train,
            frame_id=frame_id,
            opacity_mask=opacity_mask,
        )
        
        return raytrace_result

    # Adapted from Relightable 3D Gaussian: https://github.com/NJU-3DV/Relightable3DGaussian
    def pbr(
        self, 
        albedos: torch.Tensor, # (H, W, 3)
        roughnesses: torch.Tensor, # (H, W, 1)
        metallics: torch.Tensor, # (H, W, 1)
        normals: torch.Tensor, # (H, W, 3)
        viewdirs: torch.Tensor, # (H, W, 3) 
        sampling_dirs: torch.Tensor, # (N, num_sample_rays, 3)
        pdfs: torch.Tensor, # (N, num_sample_rays)
        ind_lights: torch.Tensor, # (N, num_sample_rays, 3)
        visibilities: torch.Tensor, # (N, num_sample_rays), N is valid points indicated in the mask
        sky_model, # EnvLight or EnvLight_EQ
        use_metallic: bool = True, # whether to use metallic
        mask: torch.Tensor = None, # (H, W), mask for the points to be sampled
        verbose: bool = False, # whether to print debug information
        use_ind_light: bool = True, # whether to use the indirect light
        only_diffuse: bool = False, # whether to use only diffuse
    ):
        if mask is None:
            mask = torch.ones_like(albedos[..., 0], dtype=torch.float32)
        mask = mask.bool()
        
        # H, W, num_sample_rays = sampling_dirs.shape[0], sampling_dirs.shape[1], sampling_dirs.shape[2]
        albedos = albedos[mask] # (N, 3)
        roughnesses = roughnesses[mask] # (N, 1)
        metallics = metallics[mask] # (N, 1)
        normals = normals[mask] # (N, 3)
        viewdirs = viewdirs[mask] # (N, 3)
        N, num_sample_rays = sampling_dirs.shape[0], sampling_dirs.shape[1]
        
        
        # query light -> (N, num_sample_rays, 3)
        direct_lights = sky_model.query_light(l=sampling_dirs)
     
        # (N, num_sample_rays, 3)
        lights = direct_lights * visibilities.reshape(N, num_sample_rays, 1)
        if use_ind_light:
            lights.add_(ind_lights)  # Inplace addition
            # lights.add_(ind_lights * (1-visibilities).reshape(N, num_sample_rays, 1))  # Inplace addition
        # (N, num_sample_rays, 1)
        n_d_i = (normals.unsqueeze(1) * sampling_dirs).sum(dim=-1, keepdim=True)
        n_d_i.clamp_(min=0.0)  # Inplace clamp (N, num_sample_rays, 1)
        # (N, num_sample_rays, 3)
        f_d = brdf_diffuse(
            albedos=albedos,
            lightdirs=sampling_dirs,
        )    
        
        if use_metallic:
            f_d = f_d * (1.0 - metallics.unsqueeze(-1))
        
        if use_metallic:
            # Compute f0 blend between dielectric and metallic base
            f0_metal = albedos                         # metals use albedo as F0
            f0_dielectric = torch.full_like(albedos, 0.04)  # default dielectric reflectance
            f0 = metallics * f0_metal + (1 - metallics) * f0_dielectric  # [N, 3]
        else:
            f0 = 0.04 # assume dielectric reflectance for all materials
        # (N, num_sample_rays, 3)
        f_s, extra_infos = brdf_GGX(
            normals=normals,
            viewdirs=-viewdirs, # make sure the viewdirs is inward of the camera
            lightdirs=sampling_dirs,
            roughness=roughnesses.reshape(-1),
            fresnel=f0,
            verbose=verbose,
        )
        
        # # mask out the direction which pdf is negative (direction is not in the hemisphere)
        # valid_mask = (pdfs > 0.0) # (N, num_sample_rays)
        # safe_pdfs = torch.where(valid_mask, pdfs, torch.ones_like(pdfs))
        # transport = lights * n_d_i / safe_pdfs.unsqueeze(-1) # (N, num_sample_rays, 3)
        # transport = transport * valid_mask.unsqueeze(-1) # (N, num_sample_rays, 3)
        # f_d = f_d * valid_mask.unsqueeze(-1) # (N, num_sample_rays, 3)
        # f_s = f_s * valid_mask.unsqueeze(-1) # (N, num_sample_rays, 3)
        # valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (N, 1)
        # specular = (f_s * transport).sum(dim=-2) / valid_count # (N, 3)
        # diffuse = (f_d * transport).sum(dim=-2) / valid_count # (N, 3)
        
        
        incident_areas = 1 / pdfs.clamp_min(1e-6).detach() # don't backprop through pdfs from transport / pbr loss
        transport = lights * incident_areas.unsqueeze(-1) * n_d_i # (N, num_sample_rays, 3)
        specular = (f_s * transport).mean(dim=-2) # (N, 3)
        diffuse = (f_d * transport).mean(dim=-2) # (N, 3)
        
        if only_diffuse:
            pbr_color = diffuse
        else:
            pbr_color = ((f_d + f_s) * transport).mean(dim=-2) # (N, 3)
        mean_transport = transport.mean(dim=-2)  # (N, 3)

        if torch.isnan(pbr_color).any() or torch.isinf(pbr_color).any():
            print("pbr_color is nan or inf")
            import pdb; pdb.set_trace()

        results = {
            "pbr_color": pbr_color,
            "mean_transport": mean_transport,
            "dir_light": direct_lights,
        }

        if verbose:
            # Optimization: Only compute these means when verbose=True (eval mode)
            mean_fd = f_d.mean(dim=-2) # (N, 3)
            mean_fs = f_s.mean(dim=-2) # (N, 3)
            results.update({
                "n_d_i": n_d_i,
                "light": lights,
                "diffuse": diffuse,
                "specular": specular,
                "mean_fd": mean_fd,
                "mean_fs": mean_fs,
            })
            results.update(extra_infos)
        
        return results
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False,
        verbose: bool = False,
        isosurface_render: bool = False,
        timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        timer = OpTimer(enabled=timing, use_cuda_sync=False, prefix="scene_graph forward")
        def time_cuda(name, fn):
            if timing and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end.record()
                torch.cuda.synchronize()
                timer.add_time(name, start.elapsed_time(end) / 1000.0)
                return out
            with timer.time_block(name):
                return fn()

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = time_cuda(
            "process_camera",
            lambda: self.process_camera(
                camera_infos=camera_infos,
                image_ids=image_infos["img_idx"].flatten()[0],
                novel_view=novel_view,
            ),
        )
        gs = time_cuda(
            "collect_gaussians",
            lambda: self.collect_gaussians(
                cam=processed_cam,
                image_ids=image_infos["img_idx"].flatten()[0],
            ),
        )

        # render gaussians
        outputs, render_fn = time_cuda(
            "render_gaussians",
            lambda: self.render_gaussians(
                gs=gs,
                cam=processed_cam,
                isosurface_render=isosurface_render,
                near_plane=self.render_cfg.near_plane,
                far_plane=self.render_cfg.far_plane,
                render_mode="RGB+ED",
                radius_clip=self.render_cfg.get('radius_clip', 0.),
            ),
        )
        outputs["point_light_emitters_rgb"] = torch.zeros_like(outputs["rgb_gaussians"])
        
        # render sky.
        sky_model = self.models['Sky']
        pbr_sky_model = self.models.get('SkyPBR', sky_model)
        
        # for smoothing envmap
        outputs["base_envmap"] = sky_model.get_base()
        outputs["envmap"] = sky_model.get_activated_base()
    
        # get envmap visualization
        if verbose:
            envmap_visualize = sky_model.export_vis()
            outputs["envmap_visualize"] = envmap_visualize

        # eval-only visible emitters for point lights (independent of PBR toggle)
        render_point_light_emitters = (not self.training) and self.tracer_cfg.get("render_point_light_emitters", False)
        if render_point_light_emitters:
            emitter_direct_light_mode = self.tracer_cfg.get("direct_light_mode", "envmap")
            emitter_point_light_positions, emitter_point_light_intensities = None, None
            eval_headlight_mode = self.tracer_cfg.get("eval_headlight_mode", False)
            emit_for_eval_headlight = self.tracer_cfg.get("emit_for_eval_headlight", False)
            if eval_headlight_mode:
                emitter_direct_light_mode = "point_lights"
                emitter_point_light_positions, emitter_point_light_intensities = self._build_eval_headlights(processed_cam)
            
            if emitter_direct_light_mode == "point_lights":
                cfg_light_positions, cfg_light_intensities, _, _, _, _ = self._load_point_lights()
                if emitter_point_light_positions is None or emitter_point_light_positions.shape[0] == 0:
                    emitter_point_light_positions, emitter_point_light_intensities = cfg_light_positions, cfg_light_intensities
                elif cfg_light_positions is not None and cfg_light_positions.shape[0] > 0:
                    emitter_point_light_positions = torch.cat([emitter_point_light_positions, cfg_light_positions], dim=0)
                    emitter_point_light_intensities = torch.cat([emitter_point_light_intensities, cfg_light_intensities], dim=0)

                if eval_headlight_mode and not emit_for_eval_headlight:
                    if cfg_light_positions is not None and cfg_light_positions.shape[0] > 0:
                        emitter_point_light_positions, emitter_point_light_intensities = cfg_light_positions, cfg_light_intensities
                    else:
                        emitter_point_light_positions, emitter_point_light_intensities = None, None
                    if verbose:
                        print("Skipping headlight emitter rendering (emit_for_eval_headlight=False)")
                
                if emitter_point_light_positions is not None and emitter_point_light_positions.shape[0] > 0:
                    emitter_gs = self._build_point_light_emitter_gaussians(
                        base_gs=gs,
                        point_light_positions=emitter_point_light_positions,
                        point_light_intensities=emitter_point_light_intensities,
                    )
                    emitter_outputs, _ = time_cuda(
                        "render_point_light_emitters",
                        lambda: self.render_gaussians(
                            gs=emitter_gs,
                            cam=processed_cam,
                            isosurface_render=False,
                            update_info=False,
                            near_plane=self.render_cfg.near_plane,
                            far_plane=self.render_cfg.far_plane,
                            render_mode="RGB+ED",
                            radius_clip=self.tracer_cfg.get("point_light_emitter_radius_clip", 0.0),
                        ),
                    )
                    outputs["point_light_emitters_rgb"] = emitter_outputs["rgb_gaussians"]
                    if verbose:
                        emitter_max = float(outputs["point_light_emitters_rgb"].max().item())
                        emitter_mean = float(outputs["point_light_emitters_rgb"].mean().item())
                        print(f"Point light emitters rendered: max={emitter_max:.6f}, mean={emitter_mean:.6f}")

        
        if self.use_pbr:
            # for envmap prior
            if hasattr(pbr_sky_model, "get_prior"):
                outputs["prior_envmap"] = pbr_sky_model.get_prior()
            else:
                outputs["prior_envmap"] = None
            
            # build the BVH
            time_cuda("build_acc", lambda: self.build_acc(gs, rebuild=True))
            
            use_raytraced_depth = self.tracer_cfg.get('use_raytraced_depth', False)
            if use_raytraced_depth:
                # raytrace gaussians
                raytrace_results = time_cuda(
                    "raytrace_gaussians",
                    lambda: self.raytrace_gaussians(
                        gs=gs,
                        cam=processed_cam,
                        rays_o=image_infos["origins"],
                        rays_d=image_infos["viewdirs"],
                        is_train=self.training,
                        frame_id=image_infos["frame_idx"][0, 0].item(),
                        opacity_mask=None,
                        rebuild=False,
                    ),
                )
                # outputs["raytrace_rgb"] = raytrace_results["pred_rgb"].squeeze(0)
                # outputs["raytrace_opacity"] = raytrace_results["pred_opacity"].squeeze(0)
                outputs["raytrace_depth"] = raytrace_results["pred_dist"].squeeze(0)
                # outputs["raytrace_normal"] = raytrace_results["pred_normals"].squeeze(0)
            
            # trace visibility
            num_sample_rays = self.tracer_cfg.get('train_num_sample_rays') \
                if self.training else self.tracer_cfg.get('eval_num_sample_rays')
            if num_sample_rays is None:
                if verbose:
                    print("Warning: num_sample_rays is None, using default value 64")
                num_sample_rays = 64
            env_num_sample_rays = num_sample_rays
                
            sample_type = self.tracer_cfg.get('train_sample_type', None) \
                if self.training else self.tracer_cfg.get('eval_sample_type', None)
            if sample_type is None:
                if verbose:
                    print("Warning: sample_type is None, using default type uniform")
                sample_type = 'uniform'
            direct_light_mode = self.tracer_cfg.get("direct_light_mode", "envmap")
            point_light_positions, point_light_intensities = None, None
            point_light_spot_dirs = point_light_spot_inner_cos = point_light_spot_outer_cos = point_light_spot_enabled = None
            eval_headlight_mode = (not self.training) and self.tracer_cfg.get("eval_headlight_mode", False)
            if eval_headlight_mode:
                direct_light_mode = "point_lights"
                point_light_positions, point_light_intensities = self._build_eval_headlights(processed_cam)
                Lh = int(point_light_positions.shape[0])
                point_light_spot_dirs, point_light_spot_inner_cos, point_light_spot_outer_cos, point_light_spot_enabled = \
                    self._build_eval_headlight_spot_params(processed_cam, Lh)
            point_lights_use_envmap = self.tracer_cfg.get("point_lights_use_envmap", True)
            if direct_light_mode == "point_lights":
                cfg_light_positions, cfg_light_intensities, cfg_spot_dirs, cfg_spot_inner_cos, cfg_spot_outer_cos, cfg_spot_enabled = self._load_point_lights()
                if point_light_positions is None or point_light_positions.shape[0] == 0:
                    point_light_positions, point_light_intensities = cfg_light_positions, cfg_light_intensities
                    point_light_spot_dirs, point_light_spot_inner_cos, point_light_spot_outer_cos, point_light_spot_enabled = (
                        cfg_spot_dirs, cfg_spot_inner_cos, cfg_spot_outer_cos, cfg_spot_enabled
                    )
                elif cfg_light_positions is not None and cfg_light_positions.shape[0] > 0:
                    point_light_positions = torch.cat([point_light_positions, cfg_light_positions], dim=0)
                    point_light_intensities = torch.cat([point_light_intensities, cfg_light_intensities], dim=0)
                    point_light_spot_dirs = torch.cat([point_light_spot_dirs, cfg_spot_dirs], dim=0)
                    point_light_spot_inner_cos = torch.cat([point_light_spot_inner_cos, cfg_spot_inner_cos], dim=0)
                    point_light_spot_outer_cos = torch.cat([point_light_spot_outer_cos, cfg_spot_outer_cos], dim=0)
                    point_light_spot_enabled = torch.cat([point_light_spot_enabled, cfg_spot_enabled], dim=0)
                if point_light_positions is None or point_light_positions.shape[0] == 0:
                    raise ValueError("direct_light_mode=point_lights but no valid point_lights are configured.")
                num_sample_rays = int(point_light_positions.shape[0])
                if verbose:
                    print(f"Using point light mode with {num_sample_rays} lights")
                    if point_lights_use_envmap:
                        print(f"Adding envmap direct light with {env_num_sample_rays} env rays")
            elif direct_light_mode != "envmap":
                raise ValueError(f"Unsupported direct_light_mode: {direct_light_mode}")

            use_metallic = self.tracer_cfg.get('use_metallic', False)
            if use_metallic and sample_type != 'BRDF':
                if verbose:
                    print("Warning: use_metallic is True, but sample_type is not BRDF")
                    print("metallic will not be used")
                
            if sample_type == 'BRDF' and not use_metallic:
                if verbose:
                    print("Warning: sample_type is BRDF, but use_metallic is False, using equal weight for diffuse and specular")
                # sample_type = 'uniform'
            
            trace_subpixel = self.tracer_cfg.get('trace_subpixel', False)
            if direct_light_mode == "point_lights" and trace_subpixel:
                if verbose:
                    print("Warning: trace_subpixel is disabled in point light mode")
                trace_subpixel = False
            if trace_subpixel and verbose:
                print("Tracing subpixel rays")
            
            use_ind_light = self.tracer_cfg.get('use_ind_light', True)
            if use_ind_light and verbose:
                print("Using indirect light for PBR")
                
            only_diffuse = self.tracer_cfg.get('only_diffuse', False)
            if only_diffuse and verbose:
                print("Using only diffuse for PBR")
            
            use_vndf = self.tracer_cfg.get('use_vndf', False)
            if use_vndf and verbose:
                print("Using VNDf sampling for GGX")
            
            subpixel_use_rasterized_depth = self.tracer_cfg.get('subpixel_use_rasterized_depth', False)
            subpixel_use_rasterized_feature = self.tracer_cfg.get('subpixel_use_rasterized_feature', False)
            if subpixel_use_rasterized_depth or subpixel_use_rasterized_feature :
                if trace_subpixel:
                    if verbose:
                        print("Using rasterized depth or feature for subpixel tracing")
                    sH, sW = find_closest_factors(num_sample_rays)
                    _, upsampled_depth, _, upsampled_extras = render_fn(
                        new_H=processed_cam.H * sH,
                        new_W=processed_cam.W * sW,
                    )
                    upsampled_depth = upsampled_depth.reshape(processed_cam.H * sH, processed_cam.W * sW)
                    upsampled_extras["_world_normals"] = upsampled_extras.get("_normals", None)
                else:
                    if verbose:
                        print("Warning: subpixel_use_rasterized_depth or subpixel_use_rasterized_feature is True, but trace_subpixel is False")
                    subpixel_use_rasterized_depth = False
                    subpixel_use_rasterized_feature = False
            
            skip_delta = self.tracer_cfg.get('skip_delta', None)
            if skip_delta is not None and verbose:
                print(f"Skip delta: {skip_delta}")
            
            if self.training:
                if "egocar_masks" in image_infos:
                # in the case of egocar, we need to mask out the egocar region
                    valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
                else:
                    valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
                # only do PBR on non-sky regions
                mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
                
                # # debugging: only use a region of pixels for PBR
                # print("Debugging: using a predefined region for training")
                # # x: [480, 530], y: [300, 330]
                # debug_mask = torch.zeros_like(mask)
                # debug_mask[550:560, 680:690] = 1
                # mask = mask * debug_mask
            else:
                
                # # but use rendered opacity to identify non-sky region
                # # also filter out those sky region but with floaters (small opacity)
                # rendered_opacity = outputs["opacity"].squeeze(-1) # [H, W]
                # opacity_threshold = self.render_cfg.get('opacity_threshold', 0.1)
                # mask = (rendered_opacity > opacity_threshold).float()

                # test: no mask
                mask = torch.ones(processed_cam.H, processed_cam.W).to(self.device)
                
                # # debugging: only render (740, 600)
                # print("Debugging: only rendering pixel (740, 600) for testing")
                # mask = torch.zeros_like(mask)
                # mask[600, 740] = 1.0
                
            mask = mask.bool() # [H, W]
            
            # restrict the number of rendered pixel in training
            # only render max_num_pixels in training
            max_num_ray = self.tracer_cfg.get('max_num_rays', 18)
            max_num_ray = 2 ** max_num_ray
            max_num_pixels = max_num_ray // num_sample_rays
            if self.training:
                indices = torch.nonzero(mask == 1, as_tuple=False)
                # If num_pixels is more than available 1s, just use the original mask
                if max_num_pixels < indices.size(0):
                    use_ray_error_cache = self.tracer_cfg.get("use_ray_error_cache", False)
                    has_ray_error_map = "ray_error_map" in image_infos
                    selected_indices = None
                    if use_ray_error_cache and has_ray_error_map:
                        ray_error_map = image_infos["ray_error_map"]
                        candidate_weights = ray_error_map[indices[:, 0], indices[:, 1]].float()
                        weight_power = self.tracer_cfg.get("ray_error_weight_power", 1.0)
                        if weight_power != 1.0:
                            candidate_weights = candidate_weights.pow(weight_power)
                        weight_min = self.tracer_cfg.get("ray_error_weight_min", 1e-6)
                        candidate_weights = candidate_weights.clamp_min(weight_min)
                        uniform_ratio = self.tracer_cfg.get("ray_error_uniform_ratio", 0.1)
                        if uniform_ratio > 0:
                            candidate_weights = (
                                (1.0 - uniform_ratio) * candidate_weights
                                + uniform_ratio * torch.ones_like(candidate_weights)
                            )
                        if torch.isfinite(candidate_weights).all() and candidate_weights.sum() > 0:
                            sampled = torch.multinomial(
                                candidate_weights, max_num_pixels, replacement=False
                            )
                            selected_indices = indices[sampled]
                    if selected_indices is None:
                        # fallback: random sampling
                        perm = torch.randperm(indices.size(0))[:max_num_pixels]
                        selected_indices = indices[perm]
                    # Create a new mask with all zeros
                    new_mask = torch.zeros_like(mask)
                    # Set the selected indices to 1
                    new_mask[selected_indices[:, 0], selected_indices[:, 1]] = 1
                    mask = new_mask.bool()
            outputs["pbr_mask"] = mask
            
            # split the number of pixels into chunk with length max_num_pixels to prevent OOM
            # render max_num_pixels in one pass
            indices = torch.nonzero(mask == 1, as_tuple=False)
            num_chunks = (indices.shape[0] + max_num_pixels - 1) // max_num_pixels
            chunks = torch.chunk(indices, num_chunks, dim=0)
            # Optimization: Store indices only, create mask on-the-fly to save memory
            chunk_indices = chunks

            # initialize outputs
            outputs["pbr_color"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
            outputs["pbr_transport"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
            outputs["pbr_dir_light"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
            # outputs["raytrace_visibility_raw"] = torch.zeros(indices.shape[0], num_sample_rays).to(self.device)
            # outputs["raytrace_pdfs"] = torch.zeros(indices.shape[0], num_sample_rays).to(self.device)
            if verbose:
                outputs["raytrace_visibility"] = torch.zeros(processed_cam.H, processed_cam.W).to(self.device)
                outputs["raytrace_ind_light"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
                outputs["pbr_diffuse"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
                outputs["pbr_specular"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
                outputs["pbr_diffuse_brdf"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
                outputs["pbr_specular_brdf"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
                outputs["pbr_specular_brdf_NoV"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_specular_brdf_NoL"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_specular_brdf_NoH"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_specular_brdf_VoH"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_specular_brdf_D"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_specular_brdf_Gv"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_specular_brdf_Gl"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_specular_brdf_Fr"] = torch.zeros(processed_cam.H, processed_cam.W, 3 if use_metallic else 1).to(self.device)
                outputs["pbr_n_d_i"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                outputs["pbr_light"] = torch.zeros(processed_cam.H, processed_cam.W, 3).to(self.device)
                outputs["raytrace_valid_count"] = torch.zeros(processed_cam.H, processed_cam.W, 1).to(self.device)
                # outputs["raytrace_sampling_dirs"] = torch.zeros(processed_cam.H, processed_cam.W, num_sample_rays, 3).to(self.device)

            # print(f"Rendering {indices.shape[0]} pixels in {num_chunks} chunks with {num_sample_rays} rays per pixel")
            # print(f"Rendering with {sample_type} sampling")
            # Optimization: Reuse single mask tensor instead of creating N masks
            chunk_mask = torch.zeros((processed_cam.H, processed_cam.W), dtype=torch.bool, device=self.device)
            for chunk_id, chunk_coords in enumerate(chunk_indices):
                chunk_mask.fill_(False)
                chunk_mask[chunk_coords[:, 0], chunk_coords[:, 1]] = True
                if direct_light_mode == "point_lights":
                    visibilities, sampling_dirs, light_distances = time_cuda(
                        "trace_visibility_point_lights",
                        lambda: self.trace_visibility_point_lights(
                            gs=gs,
                            cam=processed_cam,
                            depth=outputs["depth"] if not use_raytraced_depth else outputs["raytrace_depth"],
                            mask=chunk_mask,
                            point_light_positions=point_light_positions,
                            is_train=self.training,
                            rebuild=False,
                            frame_id=image_infos["frame_idx"][0, 0].item(),
                            skip_delta=skip_delta,
                        ),
                    )  # [N, L], [N, L, 3], [N, L]
                    pdfs = None
                    ind_lights = torch.zeros_like(sampling_dirs)
                    diffuse_mask = None
                else:
                    if trace_subpixel:
                        visibilities, sampling_dirs, pdfs, ind_lights, diffuse_mask = time_cuda(
                            "trace_visibility_subpixel",
                            lambda: self.trace_visibility_subpixel(
                                gs=gs,
                                cam=processed_cam,
                                normal=outputs["world_normal"],
                                roughness=outputs["roughness"],
                                metallic=outputs["metallic"],
                                is_train=self.training,
                                rebuild=False,
                                frame_id=image_infos["frame_idx"][0, 0].item(),
                                num_sample_rays=num_sample_rays,
                                random_rotate=self.training,
                                sample_type=sample_type,
                                mask=chunk_mask,
                                precomputed_depth=upsampled_depth if subpixel_use_rasterized_depth else None,
                                precomputed_normal=upsampled_extras["_world_normals"] if subpixel_use_rasterized_feature else None,
                                precomputed_roughness=upsampled_extras["_roughnesses"] if subpixel_use_rasterized_feature else None,
                                precomputed_metallic=upsampled_extras["_metallics"] if subpixel_use_rasterized_feature else None,
                                skip_delta=skip_delta,
                                use_metallic=use_metallic,
                                use_vndf=use_vndf,
                                envmap=pbr_sky_model,
                            ),
                        )
                    else:
                        visibilities, sampling_dirs, pdfs, ind_lights, diffuse_mask = time_cuda(
                            "trace_visibility",
                            lambda: self.trace_visibility(
                                gs=gs,
                                cam=processed_cam,
                                depth=outputs["depth"] if not use_raytraced_depth else outputs["raytrace_depth"],
                                normal=outputs["world_normal"],
                                viewdirs=image_infos["viewdirs"],
                                roughness=outputs["roughness"],
                                metallic=outputs["metallic"],
                                is_train=self.training,
                                rebuild=False,
                                frame_id=image_infos["frame_idx"][0, 0].item(),
                                num_sample_rays=num_sample_rays,
                                random_rotate=self.training,
                                sample_type=sample_type,
                                mask=chunk_mask,
                                skip_delta=skip_delta,
                                use_metallic=use_metallic,
                                use_vndf=use_vndf,
                                envmap=pbr_sky_model,
                                timing=timing,
                            ),
                        ) # [N, num_sample_rays], [N, num_sample_rays, 3], [N, num_sample_rays], [N, num_sample_rays, 3], [N, num_sample_rays]
                
                # # debugging: count visibility
                # print("Debugging visibility statistics:")
                # # assume N = 1
                # # find the main sampled direction and print statistics
                # if visibilities.shape[0] == 1:  # debugging, assume N = 1
                #     vis = visibilities[0]  # (num_sample_rays,)
                #     dirs = sampling_dirs[0]  # (num_sample_rays, 3)
                    
                #     print(f"Visibilities mean: {vis.mean().item():.6f}, min: {vis.min().item():.6f}, max: {vis.max().item():.6f}")
                    
                #     # find the main sampled direction using clustering
                #     # compute pairwise distances between directions
                #     dir_distances = torch.cdist(dirs, dirs)  # (num_sample_rays, num_sample_rays)
                    
                #     # find the direction with minimum average distance to others (most central)
                #     mean_distances = dir_distances.mean(dim=1)  # (num_sample_rays,)
                #     main_dir_idx = torch.argmin(mean_distances)
                #     main_dir = dirs[main_dir_idx]  # (3,)
                    
                #     # mark directions close to main direction as main cluster
                #     # use a distance threshold (e.g., cosine distance or L2 distance)
                #     direction_threshold = 0.2  # adjust this threshold as needed
                #     distances_to_main = dir_distances[main_dir_idx]  # (num_sample_rays,)
                #     main_cluster_mask = distances_to_main < direction_threshold
                    
                #     main_count = main_cluster_mask.sum().item()
                #     print(f"Main direction cluster size: {main_count}")
                    
                #     # visibility statistics for main cluster
                #     main_vis = vis[main_cluster_mask]
                #     main_mean_vis = main_vis.mean().item()
                #     print(f"Main direction: {main_dir.cpu().numpy()}, mean visibility: {main_mean_vis:.6f}")
                    
                #     # visibility statistics for non-main directions
                #     non_main_mask = ~main_cluster_mask
                #     if non_main_mask.sum() > 0:
                #         non_main_vis = vis[non_main_mask]
                #         non_main_mean_vis = non_main_vis.mean().item()
                #         print(f"Non-main directions count: {non_main_mask.sum().item()}, mean visibility: {non_main_mean_vis:.6f}")
                #     else:
                #         print("All rays in main direction cluster")
                
                # import pdb; pdb.set_trace()
                # outputs["raytrace_visibility_raw"][chunk_id * max_num_pixels : chunk_id * max_num_pixels + visibilities.shape[0]] = visibilities # (N, num_sample_rays)
                # outputs["raytrace_pdfs"][chunk_id * max_num_pixels : chunk_id * max_num_pixels + pdfs.shape[0]] = pdfs # (N, num_sample_rays)
                
                if verbose:
                    # valid is just for debug
                    valid_mask = torch.sum(
                        outputs["world_normal"][chunk_mask][:, None, :] * sampling_dirs, dim=-1
                    ) > 0.0 # (N, num_sample_rays)
                    valid_count = valid_mask.sum(dim=-1, keepdim=True) # (N, 1)
                    outputs["raytrace_visibility"][chunk_mask] = visibilities.mean(dim=-1)
                    # outputs["raytrace_visibility"][chunk_mask] = (visibilities / pdfs.clamp_min(1e-6)).mean(dim=-1) # (N,)
                    outputs["raytrace_ind_light"][chunk_mask] = ind_lights.mean(dim=1)
                    outputs["raytrace_valid_count"][chunk_mask] = valid_count.reshape(-1, 1) / num_sample_rays
                    # outputs["raytrace_sampling_dirs"][chunk_mask] = sampling_dirs
                    
                # pbr
                if direct_light_mode == "point_lights":
                    point_pbr_results = time_cuda(
                        "pbr_point_lights",
                        lambda: self.pbr_point_lights(
                            albedos=outputs["albedo"],
                            roughnesses=outputs["roughness"],
                            metallics=outputs["metallic"],
                            normals=outputs["world_normal"],
                            viewdirs=image_infos["viewdirs"],
                            sampling_dirs=sampling_dirs,
                            visibilities=visibilities,
                            light_distances=light_distances,
                            point_light_intensities=point_light_intensities,
                            point_light_spot_dirs=point_light_spot_dirs,
                            point_light_spot_inner_cos=point_light_spot_inner_cos,
                            point_light_spot_outer_cos=point_light_spot_outer_cos,
                            point_light_spot_enabled=point_light_spot_enabled,
                            use_metallic=use_metallic,
                            mask=chunk_mask,
                            verbose=verbose,
                            only_diffuse=only_diffuse,
                        ),
                    )
                    pbr_color = point_pbr_results["pbr_color"]
                    mean_transport = point_pbr_results["mean_transport"]
                    mean_dir_light = point_pbr_results["dir_light"].mean(dim=1)
                    if verbose:
                        diffuse = point_pbr_results["diffuse"]
                        specular = point_pbr_results["specular"]
                        mean_fd = point_pbr_results["mean_fd"]
                        mean_fs = point_pbr_results["mean_fs"]
                        mean_NoV = point_pbr_results["NoV"].mean(dim=1)
                        mean_NoL = point_pbr_results["NoL"].mean(dim=1)
                        mean_NoH = point_pbr_results["NoH"].mean(dim=1)
                        mean_VoH = point_pbr_results["VoH"].mean(dim=1)
                        mean_D = point_pbr_results["D"].mean(dim=1)
                        mean_Gv = point_pbr_results["Gv"].mean(dim=1)
                        mean_Gl = point_pbr_results["Gl"].mean(dim=1)
                        mean_Fr = point_pbr_results["Fr"].mean(dim=1)
                        mean_n_d_i = point_pbr_results["n_d_i"].mean(dim=1)
                        mean_light = point_pbr_results["light"].mean(dim=1)

                    if point_lights_use_envmap:
                        env_visibilities, env_sampling_dirs, env_pdfs, env_ind_lights, _ = time_cuda(
                            "trace_visibility_envmap_in_point_mode",
                            lambda: self.trace_visibility(
                                gs=gs,
                                cam=processed_cam,
                                depth=outputs["depth"] if not use_raytraced_depth else outputs["raytrace_depth"],
                                normal=outputs["world_normal"],
                                viewdirs=image_infos["viewdirs"],
                                roughness=outputs["roughness"],
                                metallic=outputs["metallic"],
                                is_train=self.training,
                                rebuild=False,
                                frame_id=image_infos["frame_idx"][0, 0].item(),
                                num_sample_rays=env_num_sample_rays,
                                random_rotate=self.training,
                                sample_type=sample_type,
                                mask=chunk_mask,
                                skip_delta=skip_delta,
                                use_metallic=use_metallic,
                                use_vndf=use_vndf,
                                envmap=pbr_sky_model,
                                timing=timing,
                            ),
                        )
                        env_pbr_results = time_cuda(
                            "pbr_envmap_in_point_mode",
                            lambda: self.pbr(
                                albedos=outputs["albedo"],
                                roughnesses=outputs["roughness"],
                                metallics=outputs["metallic"],
                                normals=outputs["world_normal"],
                                viewdirs=image_infos["viewdirs"],
                                sampling_dirs=env_sampling_dirs,
                                pdfs=env_pdfs,
                                ind_lights=env_ind_lights,
                                visibilities=env_visibilities,
                                sky_model=pbr_sky_model,
                                use_metallic=use_metallic,
                                mask=chunk_mask,
                                verbose=verbose,
                                use_ind_light=use_ind_light,
                                only_diffuse=only_diffuse,
                            ),
                        )
                        pbr_color = pbr_color + env_pbr_results["pbr_color"]
                        mean_transport = mean_transport + env_pbr_results["mean_transport"]
                        mean_dir_light = mean_dir_light + env_pbr_results["dir_light"].mean(dim=1)
                        if verbose:
                            diffuse = diffuse + env_pbr_results["diffuse"]
                            specular = specular + env_pbr_results["specular"]
                            mean_fd = mean_fd + env_pbr_results["mean_fd"]
                            mean_fs = mean_fs + env_pbr_results["mean_fs"]
                            mean_NoV = mean_NoV + env_pbr_results["NoV"].mean(dim=1)
                            mean_NoL = mean_NoL + env_pbr_results["NoL"].mean(dim=1)
                            mean_NoH = mean_NoH + env_pbr_results["NoH"].mean(dim=1)
                            mean_VoH = mean_VoH + env_pbr_results["VoH"].mean(dim=1)
                            mean_D = mean_D + env_pbr_results["D"].mean(dim=1)
                            mean_Gv = mean_Gv + env_pbr_results["Gv"].mean(dim=1)
                            mean_Gl = mean_Gl + env_pbr_results["Gl"].mean(dim=1)
                            mean_Fr = mean_Fr + env_pbr_results["Fr"].mean(dim=1)
                            mean_n_d_i = mean_n_d_i + env_pbr_results["n_d_i"].mean(dim=1)
                            mean_light = mean_light + env_pbr_results["light"].mean(dim=1)

                    outputs["pbr_color"][chunk_mask] = pbr_color
                    outputs["pbr_transport"][chunk_mask] = mean_transport
                    outputs["pbr_dir_light"][chunk_mask] = mean_dir_light
                    if verbose:
                        outputs["pbr_diffuse"][chunk_mask] = diffuse
                        outputs["pbr_specular"][chunk_mask] = specular
                        outputs["pbr_diffuse_brdf"][chunk_mask] = mean_fd
                        outputs["pbr_specular_brdf"][chunk_mask] = mean_fs
                        outputs["pbr_specular_brdf_NoV"][chunk_mask] = mean_NoV
                        outputs["pbr_specular_brdf_NoL"][chunk_mask] = mean_NoL
                        outputs["pbr_specular_brdf_NoH"][chunk_mask] = mean_NoH
                        outputs["pbr_specular_brdf_VoH"][chunk_mask] = mean_VoH
                        outputs["pbr_specular_brdf_D"][chunk_mask] = mean_D
                        outputs["pbr_specular_brdf_Gv"][chunk_mask] = mean_Gv
                        outputs["pbr_specular_brdf_Gl"][chunk_mask] = mean_Gl
                        outputs["pbr_specular_brdf_Fr"][chunk_mask] = mean_Fr
                        outputs["pbr_n_d_i"][chunk_mask] = mean_n_d_i
                        outputs["pbr_light"][chunk_mask] = mean_light
                else:
                    pbr_results = time_cuda(
                        "pbr",
                        lambda: self.pbr(
                            albedos=outputs["albedo"],
                            roughnesses=outputs["roughness"],
                            metallics=outputs["metallic"],
                            normals=outputs["world_normal"],
                            viewdirs=image_infos["viewdirs"], # note that the viewdirs should be outward from the surface
                            sampling_dirs=sampling_dirs,
                            pdfs=pdfs,
                            ind_lights=ind_lights,
                            visibilities=visibilities,
                            sky_model=pbr_sky_model,
                            use_metallic=use_metallic,
                            mask=chunk_mask,
                            verbose=verbose,
                            use_ind_light=use_ind_light,
                            only_diffuse=only_diffuse,
                        ),
                    )
                    outputs["pbr_color"][chunk_mask] = pbr_results["pbr_color"]
                    outputs["pbr_transport"][chunk_mask] = pbr_results["mean_transport"]
                    outputs["pbr_dir_light"][chunk_mask] = pbr_results["dir_light"].mean(dim=1)
                    if verbose:
                        outputs["pbr_diffuse"][chunk_mask] = pbr_results["diffuse"]
                        outputs["pbr_specular"][chunk_mask] = pbr_results["specular"]
                        outputs["pbr_diffuse_brdf"][chunk_mask] = pbr_results["mean_fd"]
                        outputs["pbr_specular_brdf"][chunk_mask] = pbr_results["mean_fs"]
                        outputs["pbr_specular_brdf_NoV"][chunk_mask] = pbr_results["NoV"].mean(dim=1)
                        outputs["pbr_specular_brdf_NoL"][chunk_mask] = pbr_results["NoL"].mean(dim=1)
                        outputs["pbr_specular_brdf_NoH"][chunk_mask] = pbr_results["NoH"].mean(dim=1)
                        outputs["pbr_specular_brdf_VoH"][chunk_mask] = pbr_results["VoH"].mean(dim=1)
                        outputs["pbr_specular_brdf_D"][chunk_mask] = pbr_results["D"].mean(dim=1)
                        outputs["pbr_specular_brdf_Gv"][chunk_mask] = pbr_results["Gv"].mean(dim=1)
                        outputs["pbr_specular_brdf_Gl"][chunk_mask] = pbr_results["Gl"].mean(dim=1)
                        outputs["pbr_specular_brdf_Fr"][chunk_mask] = pbr_results["Fr"].mean(dim=1)
                        outputs["pbr_n_d_i"][chunk_mask] = pbr_results["n_d_i"].mean(dim=1)
                        outputs["pbr_light"][chunk_mask] = pbr_results["light"].mean(dim=1)

            # additionally trace visibility toward main envmap direction for visibility refinement
            if direct_light_mode != "point_lights" and self.training:
                if "egocar_masks" in image_infos:
                # in the case of egocar, we need to mask out the egocar region
                    valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
                else:
                    valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
                # only do PBR on non-sky regions
                mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
                
                # also filter out high metallic region since the predicted shadow might be incorrect 
                metallic_threshold = self.tracer_cfg.get("vis_refine_metallic_thres", 0.3)
                gt_metallic = image_infos.get("gt_metallic", None)
                if gt_metallic is not None:
                    valid_metallic_mask = (gt_metallic < metallic_threshold).float()
                    mask = mask * valid_metallic_mask
                    
            elif direct_light_mode != "point_lights":
                # test: no mask
                mask = torch.ones(processed_cam.H, processed_cam.W).to(self.device)
            
            if direct_light_mode != "point_lights":
                mask = mask.bool() # [H, W]

                max_num_rays_vis_refine = self.tracer_cfg.get('max_num_rays_vis_refine', 17)
                max_num_rays_vis_refine = 2 ** max_num_rays_vis_refine
                max_num_pixels_vis_refine = max_num_rays_vis_refine # 1 ray per pixel
                if self.training:
                    indices = torch.nonzero(mask == 1, as_tuple=False)
                    # If num_pixels is more than available 1s, just use the original mask
                    if max_num_pixels_vis_refine < indices.size(0):
                        # Randomly select num_pixels indices
                        perm = torch.randperm(indices.size(0))[:max_num_pixels_vis_refine]
                        selected_indices = indices[perm]
                        # Create a new mask with all zeros
                        new_mask = torch.zeros_like(mask)
                        # Set the selected indices to 1
                        new_mask[selected_indices[:, 0], selected_indices[:, 1]] = 1
                        mask = new_mask.bool()
                outputs["pbr_mask_vis_refine"] = mask

                # split the number of pixels into chunk with length max_num_pixels_vis_refine to prevent OOM
                indices = torch.nonzero(mask == 1, as_tuple=False)
                num_chunks = (indices.shape[0] + max_num_pixels_vis_refine - 1) // max_num_pixels_vis_refine
                chunks = torch.chunk(indices, num_chunks, dim=0)
                chunk_indices = chunks

                outputs["raytrace_visibility_refine"] = torch.zeros(processed_cam.H, processed_cam.W).to(self.device)
                chunk_mask = torch.zeros((processed_cam.H, processed_cam.W), dtype=torch.bool, device=self.device)
                for chunk_id, chunk_coords in enumerate(chunk_indices):
                    chunk_mask.fill_(False)
                    chunk_mask[chunk_coords[:, 0], chunk_coords[:, 1]] = True
                    visibilities, _, _, _, _ = time_cuda(
                        "trace_visibility_vis_refine",
                        lambda: self.trace_visibility(
                            gs=gs,
                            cam=processed_cam,
                            depth=outputs["depth"],
                            normal=outputs["world_normal"],
                            viewdirs=image_infos["viewdirs"],
                            roughness=outputs["roughness"],
                            metallic=outputs["metallic"],
                            is_train=self.training,
                            rebuild=False,
                            frame_id=image_infos["frame_idx"][0, 0].item(),
                            num_sample_rays=1,  # only trace one ray toward main envmap direction
                            random_rotate=False,
                            sample_type='envmap_main',
                            mask=chunk_mask,
                            skip_delta=skip_delta,
                            use_metallic=use_metallic,
                            use_vndf=use_vndf,
                            envmap=pbr_sky_model,
                            timing=timing,
                        ),
                    ) # [N, 1], ...
                    
                    # store the visibility for this chunk
                    outputs["raytrace_visibility_refine"][chunk_mask] = visibilities.squeeze(-1)  # (N,)
            else:
                outputs["pbr_mask_vis_refine"] = torch.zeros(processed_cam.H, processed_cam.W, dtype=torch.bool, device=self.device)
                outputs["raytrace_visibility_refine"] = torch.zeros(processed_cam.H, processed_cam.W).to(self.device)
                

        # import pdb; pdb.set_trace()
        
        outputs["rgb_sky"] = time_cuda("render_sky", lambda: sky_model(image_infos))
        
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        outputs["rgb"] = outputs["rgb_gaussians"] + outputs["point_light_emitters_rgb"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        if self.use_pbr:
            outputs["pbr_color"] = outputs["pbr_color"] + outputs["point_light_emitters_rgb"]
            if verbose:
                outputs["pbr_color_alone"] = outputs["pbr_color"]
            outputs["pbr_color"] = outputs["pbr_color"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        use_affine = self.render_cfg.get('use_affine', True)
        if verbose and not use_affine:
            print("Warning: use_affine is False, affine transformation will not be applied")
        if use_affine:
            outputs["rgb"] = time_cuda(
                "affine_rgb",
                lambda: self.affine_transformation(outputs["rgb"], image_infos),
            )
            if self.use_pbr:
                if verbose:
                    outputs["pbr_color_alone"] = time_cuda(
                        "affine_pbr_color_alone",
                        lambda: self.affine_transformation(outputs["pbr_color_alone"], image_infos),
                    )
                outputs["pbr_color"] = time_cuda(
                    "affine_pbr_color",
                    lambda: self.affine_transformation(outputs["pbr_color"], image_infos),
                )
                # note: envmap is not affine transformed since it can only be done on specific resolution  
            with torch.no_grad():
                outputs["rgb_gaussians"] = time_cuda(
                    "affine_rgb_gaussians",
                    lambda: self.affine_transformation(outputs["rgb_gaussians"], image_infos),
                )
                outputs["rgb_sky"] = time_cuda(
                    "affine_rgb_sky",
                    lambda: self.affine_transformation(outputs["rgb_sky"], image_infos),
                )
                outputs["rgb_sky_blend"] = time_cuda(
                    "affine_rgb_sky_blend",
                    lambda: self.affine_transformation(outputs["rgb_sky_blend"], image_infos),
                )
                outputs["point_light_emitters_rgb"] = time_cuda(
                    "affine_point_light_emitters_rgb",
                    lambda: self.affine_transformation(outputs["point_light_emitters_rgb"], image_infos),
                )
        # import pdb; pdb.set_trace
        
        # convert to sRGB space (assume prior albedo images are in sRGB space, so we apply gamma correction to albedo to convert it to sRGB space)
        outputs["albedo"] = time_cuda(
            "albedo_srgb",
            lambda: torch.pow(outputs["albedo"].clamp(0.0, 1.0), 1.0/2.2),
        )


        # roughness and metallic should be linear
        rou_met_srgb = self.render_cfg.get('rou_met_srgb', False)
        if rou_met_srgb:
            outputs["roughness"] = time_cuda(
                "roughness_srgb",
                lambda: torch.pow(outputs["roughness"].clamp(0.0, 1.0), 1.0/2.2),
            )
            if self.tracer_cfg.get('use_metallic', False):
                outputs["metallic"] = time_cuda(
                    "metallic_srgb",
                    lambda: torch.pow(outputs["metallic"].clamp(0.0, 1.0), 1.0/2.2),
                )
        
        # soft clip
        if self.use_pbr:
            outputs["pbr_color"] = time_cuda(
                "soft_clip_pbr_color",
                lambda: soft_clip(outputs["pbr_color"]),
            )
            outputs["pbr_transport"] = time_cuda(
                "soft_clip_pbr_transport",
                lambda: soft_clip(outputs["pbr_transport"]),
            )
        
        # gamma correction
        use_gamma = self.render_cfg.get('gamma_correction', False)
        if verbose and use_gamma:
            print("Warning: gamma_correction is True, gamma correction will be applied")
        if use_gamma:
            outputs["rgb_gaussians"] = time_cuda(
                "gamma_rgb_gaussians",
                lambda: self.gamma_correction(outputs["rgb_gaussians"]),
            )
            outputs["rgb_sky"] = time_cuda(
                "gamma_rgb_sky",
                lambda: self.gamma_correction(outputs["rgb_sky"]),
            )
            outputs["rgb_sky_blend"] = time_cuda(
                "gamma_rgb_sky_blend",
                lambda: self.gamma_correction(outputs["rgb_sky_blend"]),
            )
            outputs["rgb"] = time_cuda(
                "gamma_rgb",
                lambda: self.gamma_correction(outputs["rgb"]),
            )
            outputs["point_light_emitters_rgb"] = time_cuda(
                "gamma_point_light_emitters_rgb",
                lambda: self.gamma_correction(outputs["point_light_emitters_rgb"]),
            )
            if verbose:
                outputs["hdr_envmap_visualize"] = outputs["envmap_visualize"]
                outputs["envmap_visualize"] = time_cuda(
                    "gamma_envmap_visualize",
                    lambda: self.gamma_correction(outputs["envmap_visualize"]),
                )
            if self.use_pbr:
                outputs["pbr_color"] = time_cuda(
                    "gamma_pbr_color",
                    lambda: self.gamma_correction(outputs["pbr_color"]),
                )
                outputs["pbr_transport"] = time_cuda(
                    "gamma_pbr_transport",
                    lambda: self.gamma_correction(outputs["pbr_transport"]),
                )
                if verbose:
                    outputs["pbr_color_alone"] = time_cuda(
                        "gamma_pbr_color_alone",
                        lambda: self.gamma_correction(outputs["pbr_color_alone"]),
                    )
                    outputs["raytrace_ind_light"] = time_cuda(
                        "gamma_raytrace_ind_light",
                        lambda: self.gamma_correction(outputs["raytrace_ind_light"]),
                    )
                    outputs["pbr_diffuse_brdf"] = time_cuda(
                        "gamma_pbr_diffuse_brdf",
                        lambda: self.gamma_correction(outputs["pbr_diffuse_brdf"]),
                    )
                    outputs["pbr_specular_brdf"] = time_cuda(
                        "gamma_pbr_specular_brdf",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf"]),
                    )
                    outputs["pbr_specular_brdf_NoV"] = time_cuda(
                        "gamma_pbr_specular_brdf_NoV",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_NoV"]),
                    )
                    outputs["pbr_specular_brdf_NoL"] = time_cuda(
                        "gamma_pbr_specular_brdf_NoL",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_NoL"]),
                    )
                    outputs["pbr_specular_brdf_NoH"] = time_cuda(
                        "gamma_pbr_specular_brdf_NoH",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_NoH"]),
                    )
                    outputs["pbr_specular_brdf_VoH"] = time_cuda(
                        "gamma_pbr_specular_brdf_VoH",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_VoH"]),
                    )
                    outputs["pbr_specular_brdf_D"] = time_cuda(
                        "gamma_pbr_specular_brdf_D",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_D"]),
                    )
                    outputs["pbr_specular_brdf_Gv"] = time_cuda(
                        "gamma_pbr_specular_brdf_Gv",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_Gv"]),
                    )
                    outputs["pbr_specular_brdf_Gl"] = time_cuda(
                        "gamma_pbr_specular_brdf_Gl",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_Gl"]),
                    )
                    outputs["pbr_specular_brdf_Fr"] = time_cuda(
                        "gamma_pbr_specular_brdf_Fr",
                        lambda: self.gamma_correction(outputs["pbr_specular_brdf_Fr"]),
                    )                                                   
                    outputs["pbr_diffuse"] = time_cuda(
                        "gamma_pbr_diffuse",
                        lambda: self.gamma_correction(outputs["pbr_diffuse"]),
                    )
                    outputs["pbr_specular"] = time_cuda(
                        "gamma_pbr_specular",
                        lambda: self.gamma_correction(outputs["pbr_specular"]),
                    )
                    # outputs["pbr_dir_light"] = time_cuda(
                    #     "gamma_pbr_dir_light",
                    #     lambda: self.gamma_correction(outputs["pbr_dir_light"]),
                    # )
                    outputs["pbr_light"] = time_cuda(
                        "gamma_pbr_light",
                        lambda: self.gamma_correction(outputs["pbr_light"]),
                    )

        outputs["full_albedo"] = outputs["albedo"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]) # fill sky color for sky region
        outputs["full_roughness"] = outputs["roughness"] + 0.5 * (1.0 - outputs["opacity"]) # fill 1 for sky
            

        if self.rigid_mono_depth_loss_fn is not None and "RigidNodes" in self.gaussian_classes.keys() \
            and self.losses_dict.rigid_mono_depth.w > 0.0:
            gaussian_mask = self.pts_labels == self.gaussian_classes["RigidNodes"]
            rigid_rgb, rigid_depth, rigid_opacity, _ = render_fn(gaussian_mask)
            outputs["RigidNodes_rgb"] = rigid_rgb
            outputs["RigidNodes_depth"] = rigid_depth
            outputs["RigidNodes_opacity"] = rigid_opacity
            if use_affine:
                outputs["RigidNodes_rgb"] = self.affine_transformation(
                    outputs["RigidNodes_rgb"], image_infos 
                )
            if use_gamma:
                outputs["RigidNodes_rgb"] = self.gamma_correction(
                    outputs["RigidNodes_rgb"]
                )

        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    if "RigidNodes_rgb" in outputs.keys() and class_name == "RigidNodes":
                        continue
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity, sep_extras = render_fn(gaussian_mask)
                    # outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_rgb"] = sep_rgb
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth
                    if use_affine:
                        outputs[class_name+"_rgb"] = self.affine_transformation(
                            outputs[class_name+"_rgb"], image_infos 
                        )
                    if use_gamma:
                        outputs[class_name+"_rgb"] = self.gamma_correction(
                            outputs[class_name+"_rgb"]
                        )
                    # TBD: return extra outputs

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity, sep_extras = render_fn(gaussian_mask)
                # outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_rgb"] = sep_rgb
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
                
                if use_affine:
                    outputs["Dynamic_rgb"] = self.affine_transformation(
                        outputs["Dynamic_rgb"], image_infos
                    )
                if use_gamma:
                    outputs["Dynamic_rgb"] = self.gamma_correction(
                        outputs["Dynamic_rgb"]
                    )
                
                # TBD: return dynamic extra outputs
        
        timer.log()
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        loss_dict, loss_stat_dict = super().compute_losses(outputs, image_infos, cam_infos, step=step)
        
        # smooth environment
        if self.env_tv_loss_fn is not None \
            and (self.losses_dict.env_tv.get('end_step', None) is None or step <= self.losses_dict.env_tv.end_step):
            env_tv_loss = self.env_tv_loss_fn(outputs["envmap"])
            loss_dict.update({
                "env_tv_loss": self.losses_dict.env_tv.w * env_tv_loss,
            })
        
        if self.env_cauchy_loss_fn is not None \
            and (self.losses_dict.env_cauchy.get('end_step', None) is None or step <= self.losses_dict.env_cauchy.end_step):
            env_cauchy_loss = self.env_cauchy_loss_fn(outputs["envmap"])
            loss_dict.update({
                "env_cauchy_loss": self.losses_dict.env_cauchy.w * env_cauchy_loss,
            })
        
        env_white_start_step = self.losses_dict.env_white.start_step
        if self.env_white_loss_fn is not None \
            and (env_white_start_step is None or step >= env_white_start_step):
            env_white_loss = self.env_white_loss_fn(outputs["envmap"])
            loss_dict.update({
                "env_white_loss": self.losses_dict.env_white.w * env_white_loss,
            })
        
        env_sg_reg_cfg = self.losses_dict.get('env_sg_reg', None)
        if env_sg_reg_cfg is not None and env_sg_reg_cfg.w > 0.0:
            env_sg_reg_start_step = env_sg_reg_cfg.start_step
            if env_sg_reg_start_step is None or step >= env_sg_reg_start_step:
                # try to call compute_reg method of envmap object
                sky_model = self.models['Sky']
                if hasattr(sky_model, 'compute_sg_reg'):
                    env_sg_reg_loss = sky_model.compute_sg_reg()
                    loss_dict.update({
                        "env_sg_reg_loss": env_sg_reg_cfg.w * env_sg_reg_loss,
                    })
        
        
        if self.use_pbr:
            if "egocar_masks" in image_infos:
                # in the case of egocar, we need to mask out the egocar region
                valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
            else:
                valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
            gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
            
            gt_rgb = image_infos["pixels"] * gt_occupied_mask[..., None]
            
            pbr_rgb = outputs["pbr_color"] * gt_occupied_mask[..., None]
            
            # pbr rgb loss (ignore ssim since not all of the pixel is rendered)
            pbr_mask = outputs["pbr_mask"] # (H, W)
            Ll1_pbr = torch.sum(pbr_mask.float().unsqueeze(-1) * torch.abs(gt_rgb - pbr_rgb)) / (torch.sum(pbr_mask) + 1e-6)
            if torch.sum(pbr_mask) == 0:
                logger.warning("No pixels rendered in PBR loss")
                Ll1_pbr = torch.tensor(0.0).to(self.device)
            loss_dict.update({
                "pbr_rgb_loss": self.losses_dict.pbr_rgb.w * Ll1_pbr,
            })
            
            if torch.isnan(Ll1_pbr).any()\
                or torch.isinf(Ll1_pbr).any():
                logger.warning("NaN loss detected in PBR loss")
                import pdb; pdb.set_trace()
            
            
            # transport loss
            if "shading_maps" in image_infos:
                in_luminance = self.losses_dict.pbr_transport.in_luminance
                # NOTE: GT shading map is in sRGB, and here the output pbr_transport is also in sRGB (we've applied gamma correction)
                transport = outputs["pbr_transport"] * gt_occupied_mask[..., None]
                gt_transport = image_infos["shading_maps"] * gt_occupied_mask[..., None]
                if in_luminance:
                    transport = torch.mean(transport, dim=-1, keepdim=True)
                    gt_transport = torch.mean(gt_transport, dim=-1, keepdim=True)
            
                transport_mask = pbr_mask.bool().unsqueeze(-1)
                Ll1_transport = torch.sum(transport_mask * torch.abs(gt_transport - transport)) / (torch.sum(transport_mask) + 1e-6)
                if torch.sum(transport_mask) == 0:
                    Ll1_transport = torch.tensor(0.0).to(self.device)
                loss_dict.update({
                    "pbr_transport_loss": self.losses_dict.pbr_transport.w * Ll1_transport,
                })
            else:
                if self.losses_dict.pbr_transport.w > 0:
                    logger.warning("shading_maps not found in image_infos for pbr_transport_loss")
        
            # # visibility loss
            # if "shadow_masks" in image_infos:
            #     num_sample_rays = outputs["raytrace_visibility_raw"].shape[1]
            #     gt_visibility = (image_infos["shadow_masks"])[pbr_mask.bool()].unsqueeze(-1).expand(-1, num_sample_rays) # (N, num_sample_rays)
            #     visibility = outputs["raytrace_visibility_raw"] # (N, num_sample_rays)
            #     pdfs = outputs["raytrace_pdfs"] # (N, num_sample_rays)
            #     print(f"GT visibility sum: {gt_visibility.sum().item()}, mean: {gt_visibility.mean().item()}")
            #     print(f"Predicted visibility sum: {visibility.sum().item()}, mean: {visibility.mean().item()}")
            #     print(f"Weighted Predicted visibility sum: {(visibility / pdfs.clamp_min(1e-6)).sum().item()}, mean: {(visibility / pdfs.clamp_min(1e-6)).mean().item()}")
            #     # cross entropy loss in pbr mask
            #     if pbr_mask.sum() > 0:
            #         visibility_loss = F.binary_cross_entropy(
            #             visibility,
            #             gt_visibility,
            #             reduction='none'
            #         ) # (N, num_sample_rays)
            #         sample_weight = 1.0 / pdfs.clamp_min(1e-6)
            #         visibility_loss = (visibility_loss * sample_weight).mean()
            #     else:
            #         visibility_loss = torch.tensor(0.0).to(self.device)
            #     loss_dict.update({
            #         "visibility_loss": self.losses_dict.visibility.w * visibility_loss,
            #     })
            # else:
            #     if self.losses_dict.visibility.w > 0:
            #         logger.warning("shadow_masks not found in image_infos for visibility_loss")
            
            # diffuse light white loss
            if self.diffuse_light_white_loss_fn is not None and self.losses_dict.diffuse_light_white.w > 0:
                # NOTE: dir_light is not gamma corrected, so it's in linear space, and the loss is computed in linear space as well
                diffuse_light_white_loss = self.diffuse_light_white_loss_fn(outputs["pbr_dir_light"])
                loss_dict.update({
                    "diffuse_light_white_loss": self.losses_dict.diffuse_light_white.w * diffuse_light_white_loss,
                })
            
            # visibility refine loss
            direct_light_mode = self.tracer_cfg.get("direct_light_mode", "envmap")
            if self.losses_dict.visibility_refine.w > 0 \
                and (self.losses_dict.visibility_refine.start_step is None or step >= self.losses_dict.visibility_refine.start_step) \
                and "shadow_masks" in image_infos \
                and direct_light_mode != "point_lights":
                refine_mask = outputs["pbr_mask_vis_refine"].bool()
                if torch.any(refine_mask):
                    gt_visibility_refine = (image_infos["shadow_masks"] * gt_occupied_mask)[refine_mask].unsqueeze(-1) # (N, 1)
                    visibility_refine = outputs["raytrace_visibility_refine"][refine_mask].unsqueeze(-1) # (N, 1)
                    visibility_refine_loss = F.binary_cross_entropy(
                        visibility_refine,
                        gt_visibility_refine,
                    )
                else:
                    visibility_refine_loss = torch.tensor(0.0, device=self.device)
                loss_dict.update({
                    "visibility_refine_loss": self.losses_dict.visibility_refine.w * visibility_refine_loss,
                })

            # albedo tv loss
            if self.albedo_tv_loss_fn is not None and self.losses_dict.albedo_tv.w > 0:
                # TODO: compute in linear space? (currently in sRGB space)
                albedo_tv_loss = self.albedo_tv_loss_fn(outputs["albedo"], gt_occupied_mask)
                loss_dict.update({"albedo_tv_loss": self.losses_dict.albedo_tv.w * albedo_tv_loss})
            
            # roughness tv loss
            if self.roughness_tv_loss_fn is not None and self.losses_dict.roughness_tv.w > 0:
                roughness_tv_loss = self.roughness_tv_loss_fn(outputs["roughness"], gt_occupied_mask)
                loss_dict.update({"roughness_tv_loss": self.losses_dict.roughness_tv.w * roughness_tv_loss})
            
            # metallic tv loss
            if self.metallic_tv_loss_fn is not None and self.losses_dict.metallic_tv.w > 0:
                metallic_tv_loss = self.metallic_tv_loss_fn(outputs["metallic"], gt_occupied_mask)
                loss_dict.update({"metallic_tv_loss": self.losses_dict.metallic_tv.w * metallic_tv_loss})
            
            # env scale invariant loss
            if self.env_scaleinv_loss_fn is not None:
                
                if ("prior_envmap" in outputs and outputs["prior_envmap"] is not None and self.losses_dict.env_scaleinv.w > 0 \
                    and (self.losses_dict.env_scaleinv.get('start_step', None) is None or step >= self.losses_dict.env_scaleinv.start_step)
                    and (self.losses_dict.env_scaleinv.get('end_step', None) is None or step <= self.losses_dict.env_scaleinv.end_step)):
                    
                    # lambda_param = self.losses_dict.env_scaleinv.get("lambda_param", 1.5)
                    # k = self.losses_dict.env_scaleinv.get("k", 10.0)
                    
                    # env_scaleinv_loss = self.env_scaleinv_loss_fn(
                    #     envmap=outputs["envmap"],
                    #     prior_envmap=outputs["prior_envmap"]
                    # )
                    env_scaleinv_loss = self.env_scaleinv_loss_fn(
                        envmap=outputs["base_envmap"],
                        prior_envmap=torch.log(outputs["prior_envmap"]),
                    )
                    
                    w = self.losses_dict.env_scaleinv.w
                    decay_start_step = self.losses_dict.env_scaleinv.get("decay_start_step", None)
                    decay_end_step = self.losses_dict.env_scaleinv.get("decay_end_step", None)
                    w_final = self.losses_dict.env_scaleinv.get("w_final", None)
                    if decay_end_step is not None and decay_start_step is not None \
                        and step >= decay_start_step and w_final is not None:
                        if step >= decay_end_step:
                            w = w_final
                        else:
                            w = w + (w_final - w) * ((step - decay_start_step) / (decay_end_step - decay_start_step))
                    
                    loss_stat_dict["env_scaleinv_w"] = w
                    loss_dict.update({"env_scaleinv_loss": w * env_scaleinv_loss})
                # else:
                #     if step % 100 == 0:
                #         logger.warning("prior_envmap not found in outputs for env_scaleinv_loss")
            
            # env color loss
            if self.env_color_loss_fn is not None and self.losses_dict.env_color.w > 0 \
                and (self.losses_dict.env_color.get("start_step", None) is None or step >= self.losses_dict.env_color.start_step):
                env_color_loss = self.env_color_loss_fn(outputs["envmap"], outputs["prior_envmap"])
                loss_dict.update({"env_color_loss": self.losses_dict.env_color.w * env_color_loss})
            
            # env sparse loss
            if self.env_sparse_loss_fn is not None and self.losses_dict.env_sparse.w > 0 \
                and (self.losses_dict.env_sparse.get("start_step", None) is None or step >= self.losses_dict.env_sparse.start_step):
                env_sparse_loss = self.env_sparse_loss_fn(outputs["envmap"], outputs["prior_envmap"])
                loss_dict.update({"env_sparse_loss": self.losses_dict.env_sparse.w * env_sparse_loss})
            

        return loss_dict, loss_stat_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict

    def construct_list_of_attributes(self):
        # hardcode attribute order/names
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # add float colors
        l += ['red', 'green', 'blue']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l

    # save all properties to a full ply file
    def save_full_ply(self, save_path: str, image_infos: Dict[str, torch.Tensor], mask_func=None) -> None:
        # Note: can only be done in evaluation mode!
        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
                
        gs_dict = {
            "_means": [],
            "_opacities": [],
            "_features_dc": [],
            "_features_rest": [],
            "_scales": [],
            "_quats": [],
            "_normals": [],
            "_albedos": [],
            "_roughnesses": [],
            "_metallics": [],
            "_normals_minscale": [],
            "_min_scales": [],
            "_max_scales": [],
        }
        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussian_indep()
            if gs is None:
                continue
            # collect gaussians
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        # since sh degree of SMPL Node is 1, but other is 3
        # we pad zeros to it
        replace_dict = {}
        for i, feature_rest in enumerate(gs_dict['_features_rest']):
            if feature_rest.shape[1] < 15:
                extended_feature_rest = torch.zeros(feature_rest.shape[0], 15, 3).to(self.device)
                extended_feature_rest[:, :feature_rest.shape[1], :] = feature_rest
                replace_dict[i] = extended_feature_rest
            elif feature_rest.shape[1] > 15:
                raise ValueError("SH degree larger than 3, not supported")
            
        for k, v in replace_dict.items():
            gs_dict['_features_rest'][k] = v
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        xyz = gs_dict["_means"].detach().cpu().numpy()
        normals = gs_dict["_normals"].detach().cpu().numpy()
        f_dc = gs_dict["_features_dc"].detach().contiguous().cpu().numpy()
        f_rest = gs_dict["_features_rest"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = gs_dict["_opacities"].detach().cpu().numpy()
        scales = gs_dict["_scales"].detach().cpu().numpy()
        rotations = gs_dict["_quats"].detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        
        # --- Colors (float) ---
        colors = torch.sigmoid(gs_dict["_albedos"])
        
        # normals_ = gs_dict["_normals"]
        # colors = (normals_.clone() + 1.0) * 0.5
        
        # normals_minscale = gs_dict["_normals_minscale"]
        # normals_minscale_norm = torch.norm(normals_minscale, dim=-1, keepdim=True)
        # normals_minscale = normals_minscale / (normals_minscale_norm + 1e-8)
        # colors = (normals_minscale.clone() + 1.0) * 0.5
        
        # minscale = gs_dict["_min_scales"]
        # maxscale = gs_dict["_max_scales"]
        # # flat rate = (|maxscale| - |minscale|) / |maxscale|
        # flat_rate = ((maxscale - minscale) / (maxscale + 1e-8))
        # colors = color_map(flat_rate)
        
        # maxscale = gs_dict["_max_scales"]
        # # normalize to [0, 1]
        # maxscale = (maxscale - torch.min(maxscale)) / (torch.max(maxscale) - torch.min(maxscale) + 1e-8)
        # colors = color_map(maxscale)
        
        colors.clamp_(0, 1)  # Inplace clamp
        colors = colors.detach().cpu().numpy().astype(np.float32, copy=False)
            
        if mask_func is not None:
            mask = mask_func(xyz)
            xyz = xyz[mask]
            normals = normals[mask]
            f_dc = f_dc[mask]
            f_rest = f_rest[mask]
            opacities = opacities[mask]
            scales = scales[mask]
            rotations = rotations[mask]
            colors = colors[mask]
            print(f"Filter out {np.sum(~mask)} gaussians by mask_func, {xyz.shape[0]} gaussians left")
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, colors, f_dc, f_rest, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_path)
    
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        super().resume_from_checkpoint(ckpt_path, load_only_model=load_only_model)
        # after loading the checkpoint, we need to rebuild the mips and pdf of envmap
        if 'Sky' in self.models:
            if self.models['Sky'].forward_mode != "pure_env":
                self.models['Sky'].build_mips()
            self.models['Sky'].update_pdf()

            # # debug
            # print("Debug: initialize debug dir from envmap")
            # self.debug_dir.data = self.models['Sky'].get_main_dir()
    
    def calibrate_envmap(self, camera_dir: torch.Tensor):
        """
        Calibrate the envmap using the camera direction of the first frame
        """
        if hasattr(self.models['Sky'], 'calibrate_direction'):
            self.models['Sky'].calibrate_direction(camera_dir)
        else:
            logger.warning("Sky model does not support calibrate_direction method.")
    
    def set_envmap_rotation(self, angle_degrees: float = 0, rotate_vertical: bool = False):
        """Set the rotation of the environment map.

        Args:
            angle_degrees (float, optional): The angle in degrees to rotate the environment map. Defaults to 0.
        """
        if hasattr(self.models['Sky'], 'set_rotate'):
            self.models['Sky'].set_rotate(angle_degrees, rotate_vertical)
        else:
            logger.warning("Sky model does not support set_rotation method.")
    
    def load_envmap(self, envmap_path: str, activation_name: str = 'none',
                    resolution: int = 1024, min_res: int = 64, max_res: int = 1024,
                    min_roughness: float = 0.08, max_roughness: float = 0.5, init_value: float = 0.5, prior_path: str = None,):
        """Load the environment map from a file.

        Args:
            envmap_path (str): The path to the environment map file.
        """
        del self.models['Sky']
        self.models['Sky'] = EnvLight_EQ(
            class_name='Sky',
            resolution=resolution,
            device=self.device,
            activation_name=activation_name,
            min_res=min_res,
            max_res=max_res,
            min_roughness=min_roughness,
            max_roughness=max_roughness,
            init_value=init_value,
            prior_path=prior_path,
            path=envmap_path,
        )
        if self.models['Sky'].forward_mode != "pure_env":
            self.models['Sky'].build_mips()
        self.models['Sky'].update_pdf()
        # import pdb; pdb.set_trace()
    
    def mask_envmap(self, dirs: torch.Tensor):
        """Mask the environment map based on the provided directions.

        Args:
            dirs (torch.Tensor): The directions to mask the environment map.
        """
        
        self.models['Sky'].apply_mask_by_dirs(dirs)
        
