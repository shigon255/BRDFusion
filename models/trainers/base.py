from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn

import kornia
from enum import IntEnum
import viser
import nerfview
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from plyfile import PlyData, PlyElement

from models.gaussians.basics import *
from models.graphics_utils import *
from models.modules import EnvLight

import threedgrt_tracer
from threedgrut.datasets.protocols import Batch

from models.custom_surfel_tracer import CustomSurfelTracer
from utils.timing import OpTimer

logger = logging.getLogger()

class GSModelType(IntEnum):
    Background = 0
    RigidNodes = 1
    SMPLNodes = 2
    DeformableNodes = 3

def lr_scheduler_fn(
    cfg: OmegaConf,
    lr_init: float
):
    if cfg.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = cfg.lr_final

    def func(step):
        step = step - cfg.opt_after
        if step < 0:
            return 0.
        
        if step < cfg.warmup_steps:
            if cfg.ramp == "cosine":
                lr = cfg.lr_pre_warmup + (lr_init - cfg.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / cfg.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    cfg.lr_pre_warmup
                    + (lr_init - cfg.lr_pre_warmup) * step / cfg.warmup_steps
                )
        else:
            t = np.clip(
                (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps), 0, 1
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            
        return lr  # divided by lr_init because the multiplier is with the initial learning rate

    return func

class BasicTrainer(nn.Module):
    def __init__(
        self,
        type: str = "basic",
        optim: OmegaConf = None,
        losses: OmegaConf = None,
        render: OmegaConf = None,
        res_schedule: OmegaConf = None,
        gaussian_optim_general_cfg: OmegaConf = None,
        gaussian_ctrl_general_cfg: OmegaConf = None,
        tracer: OmegaConf = None,
        model_config: OmegaConf = None,
        num_train_images: int = 0,
        num_full_images: int = 0,
        test_set_indices: List[int] = None,
        scene_aabb: torch.Tensor = None,
        device=None,
        use_pbr: bool = None,
    ):
        super().__init__()
        self._type = type
        self.optim_general = optim
        self.losses_dict = losses
        self.render_cfg = render
        self.res_schedule = res_schedule
        self.model_config = model_config
        self.num_iters = self.optim_general.get("num_iters", 30000)
        self.gaussian_optim_general_cfg = gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = gaussian_ctrl_general_cfg
        self.tracer_cfg = tracer
        self.step = 0
        self.device = device
        self.gaussian_2d = self.gaussian_ctrl_general_cfg.get("gaussian_2d", False)
        
        # dataset infos
        self.num_train_images = num_train_images
        self.num_full_images = num_full_images
        
        # init scene scale
        self._init_scene(scene_aabb=scene_aabb)
        
        # init models
        self.models = {}
        self.misc_classes_keys = [
            'Sky', 'Affine', 'CamPose', 'CamPosePerturb'
        ]
        self.gaussian_classes = {}
        self._init_models()
        self.pts_labels = None # will be overwritten in forward
        self.render_dynamic_mask = False
        
        # init losses fn
        self._init_losses()
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.step = 0

        # background color
        self.back_color = torch.zeros(3).to(self.device)
    
        # for evaluation
        self.cur_frame = torch.tensor(0, device=self.device)
        self.test_set_indices = test_set_indices # will be override
        
        # a simple viewer for background visualization
        self.viewer = None
        
        # initialize 3dgs ray tracer
        if self.gaussian_2d:
            # self.tracer = CustomSurfelTracer(transmittance_min=0.03)
            # since surfel has bug now, use threedgrt tracer temporarily
            self.tracer = threedgrt_tracer.Tracer(self.tracer_cfg)
        else:
            self.tracer = threedgrt_tracer.Tracer(self.tracer_cfg)
        
        if use_pbr is not None:
            # override
            self.use_pbr = use_pbr
        else:
            self.use_pbr = self.tracer_cfg.get('use_pbr', False)
        self.freeze_geometry_flag = self.render_cfg.get("freeze_geometry", False)
        self.freeze_material_flag = self.render_cfg.get("freeze_material", False)
        self.freeze_envmap_flag = self.render_cfg.get("freeze_envmap", False)

        # # debug: trainable direction
        # print("Debugging: adding trainable direction for visualization")
        # self.debug_dir = torch.tensor([0.0, 1.0, 0.0], device=self.device) # (3,)
        # self.debug_dir.requires_grad = True
    
    @property
    def in_test_set(self):
        return self.cur_frame.item() in self.test_set_indices
    
    def set_train(self):
        for model in self.models.values():
            model.train()
        self.train()
        self.training = True
    
    def set_eval(self):
        for model in self.models.values():
            model.eval()
        self.eval()
        self.training = False

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.res_schedule.downscale_times - self.step // self.res_schedule.double_steps), 0)
        else:
            return 1
        
    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get('optim', None)
        class_ctrl_cfg = model_cfg.get('ctrl', None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg['optim'] = new_optim_cfg
        model_cfg['ctrl'] = new_ctrl_cfg

        return model_cfg
        
    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin
        logger.info(f"scene origin: {scene_origin}")
        logger.info(f"scene radius: {scene_radius}")
    
    def _init_models(self) -> None:
        raise NotImplementedError("Please implement the _init_models function")
    
    def initialize_optimizer(self) -> None:
        # get param groups first
        self.param_groups = {}
        for class_name, model in self.models.items():
            self.param_groups.update(model.get_param_groups())
                 
        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            class_optim_cfg = class_cfg["optim"]
            raw_optim_cfg = class_optim_cfg.get(component_name, None)
            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
            if raw_optim_cfg.get("lr_final", None) is not None:
                print(f"Using lr scheduler for {params_name}")
                # import pdb; pdb.set_trace()
                sched_cfg = OmegaConf.create({
                    "opt_after": raw_optim_cfg.get('opt_after', 0),
                    "warmup_steps": raw_optim_cfg.get('warmup_steps', 0),
                    "max_steps": raw_optim_cfg.get('max_steps', self.num_iters),
                    "lr_pre_warmup": raw_optim_cfg.get('lr_pre_warmup', 1.0e-8),
                    "lr_final": raw_optim_cfg.get('lr_final', None),
                    "ramp": raw_optim_cfg.get('ramp', "cosine"),
                })
                # scale the learning rate according to the scene scale
                sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                sched_cfg.lr_final = sched_cfg.lr_final * lr_scale_factor if sched_cfg.lr_final is not None else None
                # adjust max_steps to account for opt_after
                sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)

        # # debug: add trainable direction to optimizer
        # print("Debug: adding debug_dir to optimizer")
        # groups.append({
        #     'params': [self.debug_dir],
        #     'name': "debug_dir",
        #     'lr': 0.01,
        #     'eps': 1.0e-15,
        #     'weight_decay': 0,
        # })

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
    
        # freeze if needed
        if self.freeze_geometry_flag:
            self.freeze_geometry()
        if self.freeze_material_flag:
            self.freeze_material()
        if self.freeze_envmap_flag:
            self.freeze_envmap()
    
    def _init_losses(self) -> None:
        sky_opacity_loss_fn = None
        if "Sky" in self.models:
            if self.losses_dict.mask.opacity_loss_type == "bce":
                from models.losses import binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: binary_cross_entropy(pred, gt, reduction="mean")
            elif self.losses_dict.mask.opacity_loss_type == "safe_bce":
                from models.losses import safe_binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: safe_binary_cross_entropy(pred, gt, limit=0.1, reduction="mean")
        self.sky_opacity_loss_fn = sky_opacity_loss_fn
        
        depth_loss_fn = None
        depth_loss_cfg = self.losses_dict.get("depth", None)
        if depth_loss_cfg is not None:
            from models.losses import DepthLoss
            depth_loss_fn = DepthLoss(
                loss_type=depth_loss_cfg.loss_type,
                normalize=depth_loss_cfg.normalize,
                use_inverse_depth=depth_loss_cfg.inverse_depth,
            )
        self.depth_loss_fn = depth_loss_fn
        
        mono_depth_loss_fn = None
        mono_depth_loss_cfg = self.losses_dict.get("mono_depth", None)
        if mono_depth_loss_cfg is not None:
            from models.losses import MonoDepthLoss, MonoDepth2Loss
            mono_depth_loss_fn = MonoDepthLoss()
            # mono_depth_loss_fn = MonoDepth2Loss()
        self.mono_depth_loss_fn = mono_depth_loss_fn
        
        rigid_mono_depth_loss_fn = None
        rigid_mono_depth_loss_cfg = self.losses_dict.get("rigid_mono_depth", None)
        if rigid_mono_depth_loss_cfg is not None:
            from models.losses import MonoDepthLoss
            rigid_mono_depth_loss_fn = MonoDepthLoss()
        self.rigid_mono_depth_loss_fn = rigid_mono_depth_loss_fn
        
        depth_tv_loss_fn = None
        depth_tv_loss_cfg = self.losses_dict.get("depth_tv", None)
        if depth_tv_loss_cfg is not None:
            from models.losses import DepthTVLoss
            depth_tv_loss_fn = DepthTVLoss()
        self.depth_tv_loss_fn = depth_tv_loss_fn

        normal_loss_fn = None
        normal_loss_cfg = self.losses_dict.get("normal", None)
        if normal_loss_cfg is not None:
            from models.losses import NormalLoss
            normal_loss_fn = NormalLoss()
        self.normal_loss_fn = normal_loss_fn
        
        normal_cos_loss_fn = None
        normal_cos_loss_cfg = self.losses_dict.get("normal_cos", None)
        if normal_cos_loss_cfg is not None:
            from models.losses import NormalCosLoss
            normal_cos_loss_fn = NormalCosLoss()
        self.normal_cos_loss_fn = normal_cos_loss_fn
    
        normal_tv_loss_fn = None
        normal_tv_loss_cfg = self.losses_dict.get("normal_tv", None)
        if normal_tv_loss_cfg is not None:
            from models.losses import NormalTVLoss
            normal_tv_loss_fn = NormalTVLoss()
        self.normal_tv_loss_fn = normal_tv_loss_fn
    
        normal_minscale_loss_fn = None
        normal_minscale_loss_cfg = self.losses_dict.get("normal_minscale", None)
        if normal_minscale_loss_cfg is not None:
            from models.losses import NormalLoss
            normal_minscale_loss_fn = NormalLoss()
        self.normal_minscale_loss_fn = normal_minscale_loss_fn
        
        normal_minscale_cos_loss_fn = None
        normal_minscale_cos_loss_cfg = self.losses_dict.get("normal_minscale_cos", None)
        if normal_minscale_cos_loss_cfg is not None:
            from models.losses import NormalCosLoss
            normal_minscale_cos_loss_fn = NormalCosLoss()
        self.normal_minscale_cos_loss_fn = normal_minscale_cos_loss_fn
    
        road_normal_loss_fn = None
        road_normal_loss_cfg = self.losses_dict.get("road_normal", None)
        if road_normal_loss_cfg is not None:
            from models.losses import RoadNormalLoss
            road_normal_loss_fn = RoadNormalLoss()
        self.road_normal_loss_fn = road_normal_loss_fn
        
        albedo_loss_fn = None
        albedo_loss_cfg = self.losses_dict.get("albedo", None)
        if albedo_loss_cfg is not None:
            from models.losses import AlbedoLoss
            albedo_loss_fn = AlbedoLoss()
        self.albedo_loss_fn = albedo_loss_fn
        
        metallic_loss_fn = None
        metallic_loss_cfg = self.losses_dict.get("metallic", None)
        if metallic_loss_cfg is not None:
            from models.losses import MetallicLoss
            metallic_loss_fn = MetallicLoss()
        self.metallic_loss_fn = metallic_loss_fn
        
        roughness_loss_fn = None
        roughness_loss_cfg = self.losses_dict.get("roughness", None)
        if roughness_loss_cfg is not None:
            from models.losses import RoughnessLoss
            roughness_loss_fn = RoughnessLoss()
        self.roughness_loss_fn = roughness_loss_fn
    
    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()
    
    def optimizer_step(self) -> None:
        # for params_name, optimizer in self.optimizers.items():
        #     class_name = params_name.split("#")[0]
        #     component_name = params_name.split("#")[1]
        #     max_norm = self.model_config[class_name]["optim"][component_name].get("max_norm", None)
        #     if max_norm is not None:
        #         self.grad_scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(self.param_groups[params_name], max_norm)
        #     if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
        #         self.grad_scaler.step(optimizer)
        self.optimizer.step()

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step
        for class_name in self.gaussian_classes.keys():
            self.models[class_name].preprocess_per_train_step(step)
        for model in self.models.values():
            if hasattr(model, "update_sharpness"):
                model.update_sharpness(step)

        # viewer
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()
        
    def postprocess_per_train_step(self, step: int, timing: bool = False) -> None:
        timer = OpTimer(enabled=timing, use_cuda_sync=False, prefix="train post")
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

        radii = time_cuda("radii", lambda: self.info["radii"])
        
        if self.render_cfg.absgrad:
            grads = time_cuda("grads_abs", lambda: self.info["means2d"].absgrad.clone())
        else:
            grads = time_cuda("grads", lambda: self.info["means2d"].grad.clone())
        
        def scale_grads():
            grads[..., 0] *= self.info["width"] / 2.0 * self.render_cfg.batch_size
            grads[..., 1] *= self.info["height"] / 2.0 * self.render_cfg.batch_size
        time_cuda("scale_grads", scale_grads)
        
        # for gof rendering
        if len(grads.shape) == 2:
            grads = time_cuda("grads_unsqueeze", lambda: grads.unsqueeze(0))
        
        # check state of each parameters
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            def do_post():
                return self.models[class_name].postprocess_per_train_step(
                    step=step,
                    optimizer=self.optimizer,
                    radii=radii[0, gaussian_mask],
                    xys_grad=grads[0, gaussian_mask],
                    last_size=max(self.info["width"], self.info["height"])
                )
            time_cuda(f"post_{class_name}", do_post)
        
        # viewer
        if self.viewer is not None:
            with timer.time_block("viewer"):
                num_train_rays_per_step = self.render_cfg.batch_size * self.info["width"] * self.info["height"]
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
        timer.log()
    
    def update_visibility_filter(self) -> None:
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            self.models[class_name].cur_radii = self.info["radii"][0, gaussian_mask]

    def process_camera(
        self,
        camera_infos: Dict[str, torch.Tensor],
        image_ids: torch.Tensor,
        novel_view: bool = False
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]
        
        if "CamPosePerturb" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

        if "CamPose" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPose"](camtoworlds, image_ids)
        
        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,
            camtoworlds_gt=camtoworlds_gt,
            Ks=camera_infos["intrinsics"],
            H=camera_infos["height"],
            W=camera_infos["width"]
        )
        
        return camera_dict

    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor # leave it here for future use
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_features_dc": [],
            "_features_rest": [],
            "_opacities": [],
            "_normals": [],
            "_albedos": [],
            "_roughnesses": [],
            "_metallics": [],
            "_normals_minscale": [],
            "class_labels": [],
        }
        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
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
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()

        extras = {}
        for k, v in gs_dict.items():
            if k not in ["_means", "_scales", "_quats", "_rgbs", "_opacities", "_features_dc", "_features_rest"]:
                extras[k] = v

        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            _features_dc=gs_dict["_features_dc"],
            _features_rest=gs_dict["_features_rest"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras=extras        # to save some extra information (TODO) more flexible way
        )
        
        return gaussians

    def construct_list_of_attributes(self):
        # hack: simply hardcode the number of features
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
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
    # NOTE: even if we are using 2D gaussians, we still have 3D scale, but with z-scale = 0
    def save_full_ply(self, save_path: str) -> None:
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
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_path)
    
    @torch.no_grad()
    def build_acc(self, gaussians, rebuild=True):
        # Note: gaussians need to be activated
        if self.gaussian_2d:
            # TODO: only rebuild after densification. If gs do not densify, just use update_bvh
            self.tracer.build_acc_dataclass(gaussians)
        else:
            self.tracer.build_acc_dataclass(gaussians, rebuild=rebuild)
    

    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        isosurface_render: bool = False,
        update_info: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        if isosurface_render:
            # set all opacity of gs to 1
            gs._opacities = torch.ones_like(gs._opacities)
        
        def render_fn(opaticy_mask=None, return_info=False, new_H=None, new_W=None):
            
            # gsplat
            
            # concat extras to colors to N-D feature
            features = [gs.rgbs]
            for _, v in gs.extras.items():
                features.append(v)
            features = torch.cat(features, dim=-1)
            
            # TODO: support 2DGS
            if self.gaussian_2d:
                renders, alphas, rendered_normal, surface_normal, render_distort, median_depth, info = rasterization_2dgs(
                    means=gs.means,
                    quats=gs.quats,
                    scales=gs.scales,
                    opacities=gs.opacities.squeeze(-1)*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(-1),
                    colors=features[None, ...],  # [C, N, D]
                    viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                    Ks=cam.Ks[None, ...], # [C, 3, 3]
                    width=cam.W if new_W is None else new_W,
                    height=cam.H if new_H is None else new_H,
                    packed=self.render_cfg.packed,
                    absgrad=self.render_cfg.absgrad,
                    sparse_grad=self.render_cfg.sparse_grad,
                    distloss=True, 
                    **kwargs,
                )
                info["2dgs_rendered_normal"] = rendered_normal.squeeze(0)
                info["2dgs_surface_normal"] = surface_normal.squeeze(0)
            else:              
                renders, alphas, info = rasterization(
                    means=gs.means,
                    quats=gs.quats,
                    scales=gs.scales,
                    opacities=gs.opacities.squeeze(-1)*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(-1),
                    # colors=gs.rgbs,
                    colors=features,
                    viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                    Ks=cam.Ks[None, ...],  # [C, 3, 3]
                    width=cam.W if new_W is None else new_W,
                    height=cam.H if new_H is None else new_H,
                    packed=self.render_cfg.packed,
                    absgrad=self.render_cfg.absgrad,
                    sparse_grad=self.render_cfg.sparse_grad,
                    rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                    # distloss=True, # this only supported by modified version of gsplat. We don't use it.
                    **kwargs,
                )
                
            renders = renders[0]
            alphas = alphas[0]
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"
            
            # assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
            # rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
                        
            assert renders.shape[-1] == features.shape[-1] + 1, f"Must render rgb, depth and alpha"
            extra_lengths = [v.shape[-1] for k, v in gs.extras.items()]
            rendered = torch.split(renders, [3] + extra_lengths + [1], dim=-1)
            rendered_rgb = rendered[0]
            rendered_depth = rendered[-1]
            rendered_extras = {}
            for i, k in enumerate(gs.extras.keys()):
                rendered_extras[k] = rendered[i+1]
            
            # if radii is 2D (in 2DGS), take max
            if info["radii"].shape[-1] == 2:
                info["radii"] = torch.max(info["radii"], dim=-1).values
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas, rendered_extras
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas, rendered_extras, info
        
        # render rgb and opacity
        rgb, depth, opacity, extras, info = render_fn(return_info=True)
        if update_info:
            self.info = info
        
        # rotate normal to world space and negate the G-axis and the B-axis to match the prediction of monocular normal estimation
        c2w = cam.camtoworlds
        w2c = torch.inverse(c2w)
        
        # # alpha-normalize the blended normal and the materials
        # # Not sure if this is necessary, but this is done in R3DG and IRGS
        # extras["_normals"] = extras["_normals"] / opacity.clamp_min(1e-10)
        # extras["_normals_minscale"] = extras["_normals_minscale"] / opacity.clamp_min(1e-10)
        # extras["_albedos"] = extras["_albedos"] / opacity.clamp_min(1e-10)
        # extras["_roughnesses"] = extras["_roughnesses"] / opacity.clamp_min(1e-10)
        # extras["_metallics"] = extras["_metallics"] / opacity.clamp_min(1e-10)
        
        # caution: normal is not normalized
        normals = extras.get("_normals", None)
        normals = F.normalize(normals, dim=-1)
        extras["_world_normals"] = normals
        normals = normals @ w2c[:3, :3].T
        normals[..., 1] = -normals[..., 1]
        normals[..., 2] = -normals[..., 2]
        normals = F.normalize(normals, dim=-1)
        extras["_normals"] = normals
        
        normals_minscale = extras.get("_normals_minscale", None)
        normals_minscale = F.normalize(normals_minscale, dim=-1)
        normals_minscale = normals_minscale @ w2c[:3, :3].T
        normals_minscale[..., 1] = -normals_minscale[..., 1]
        normals_minscale[..., 2] = -normals_minscale[..., 2]
        normals_minscale = F.normalize(normals_minscale, dim=-1)
        extras["_normals_minscale"] = normals_minscale
        
        distortion_map = None
        if "distortion" in self.info:
            distortion_map = self.info["distortion"]
            distortion_map = distortion_map.squeeze(0)
        elif "render_distloss" in self.info:
            distortion_map = self.info["render_distloss"]
            distortion_map = distortion_map.squeeze((0, -1))
        
        # if not self.training:
        #     # filter the intrinsic with opacity mask to remove floaters in the sky
        #     opacity_threshold = self.render_cfg.get("opacity_threshold", 0.1)
        #     opacity_mask = (opacity > opacity_threshold).float() # [H, W, 1]
        #     for k in ["_normals", "_world_normals", "_albedos", "_roughnesses", "_metallics", "_normals_minscale"]:
        #         extras[k] = extras[k] * opacity_mask
                
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity,
            "normal": extras.get("_normals", None),
            "world_normal": extras.get("_world_normals", None),
            "albedo": extras.get("_albedos", None),
            "roughness": extras.get("_roughnesses", None),
            "metallic": extras.get("_metallics", None),
            "normal_minscale": extras.get("_normals_minscale", None),
            "distortion_map": distortion_map,
        }
        
        
        if self.training and update_info:
            self.info["means2d"].retain_grad()
        
        return results, render_fn

    def affine_transformation(
        self,
        rgb_blended: torch.Tensor,
        image_infos: Dict[str, torch.Tensor]
        ):
        if "Affine" in self.models:
            affine_trs = self.models['Affine'](image_infos)
            rgb_transformed = (affine_trs[..., :3, :3] @ rgb_blended[..., None] + affine_trs[..., :3, 3:])[..., 0]
            
            return rgb_transformed
        else:       
            return rgb_blended
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
            novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )

        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        
        return outputs
    
    def backward(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        # ----------------- backward ----------------
        total_loss = sum(loss for loss in loss_dict.values())
        self.grad_scaler.scale(total_loss).backward()

        # # Debug: envmap gradient/parameter deltas
        # debug_envmap_grad = self.optim_general.get("debug_envmap_grad", False)
        # debug_envmap_grad_freq = self.optim_general.get("debug_envmap_grad_freq", 1)
        # do_debug = debug_envmap_grad and (self.step % max(int(debug_envmap_grad_freq), 1) == 0)
        # envmap_pre = None
        # if do_debug and "Sky" in self.models and hasattr(self.models["Sky"], "base"):
        #     envmap_base = self.models["Sky"].base
        #     grad = envmap_base.grad
        #     if grad is None:
        #         print(f"[envmap-grad] step={self.step} grad=None")
        #     else:
        #         with torch.no_grad():
        #             grad_stats = {
        #                 "min": grad.min().item(),
        #                 "max": grad.max().item(),
        #                 "mean": grad.mean().item(),
        #                 "abs_mean": grad.abs().mean().item(),
        #                 "norm": grad.norm().item(),
        #             }
        #             print(
        #                 "[envmap-grad] step={step} min={min:.6e} max={max:.6e} "
        #                 "mean={mean:.6e} abs_mean={abs_mean:.6e} norm={norm:.6e}".format(
        #                     step=self.step, **grad_stats
        #                 )
        #             )
        #             envmap_pre = envmap_base.detach().clone()

        self.optimizer_step()

        # if do_debug and envmap_pre is not None:
        #     envmap_post = self.models["Sky"].base.detach()
        #     delta = envmap_post - envmap_pre
        #     with torch.no_grad():
        #         delta_stats = {
        #             "min": delta.min().item(),
        #             "max": delta.max().item(),
        #             "mean": delta.mean().item(),
        #             "abs_mean": delta.abs().mean().item(),
        #             "norm": delta.norm().item(),
        #         }
        #         print(
        #             "[envmap-delta] step={step} min={min:.6e} max={max:.6e} "
        #             "mean={mean:.6e} abs_mean={abs_mean:.6e} norm={norm:.6e}".format(
        #                 step=self.step, **delta_stats
        #             )
        #         )
        
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            for group in self.optimizer.param_groups:
                if group["name"] in self.lr_schedulers:
                    new_lr = self.lr_schedulers[group["name"]](self.step)
                    group["lr"] = new_lr
                    # print(f"Step {self.step}: {group['name']} learning rate: {new_lr:.6f}")
                
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # calculate loss
        loss_dict = {}
        loss_stat_dict = {}
        
        if "egocar_masks" in image_infos:
            # in the case of egocar, we need to mask out the egocar region
            valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
            
        gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
        predicted_rgb = outputs["rgb"] * valid_loss_mask[..., None]
        
        gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
        pred_occupied_mask = outputs["opacity"].squeeze() * valid_loss_mask
        
        gt_normal = image_infos["normals"]
        pred_normal = outputs["normal"]
        pred_normal_minscale = outputs["normal_minscale"]
        
        # rgb loss
        Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
        simloss = 1 - self.ssim(gt_rgb.permute(2, 0, 1)[None, ...], predicted_rgb.permute(2, 0, 1)[None, ...])
        loss_dict.update({
            "rgb_loss": self.losses_dict.rgb.w * Ll1,
            "ssim_loss": self.losses_dict.ssim.w * simloss,
        })
        
        
        # mask loss
        if self.sky_opacity_loss_fn is not None:
            w = self.losses_dict.mask.w
            w_final = self.losses_dict.mask.get("w_final", None)
            end_step = self.losses_dict.mask.get("schedule_end_step", None)
            if w_final is not None and end_step is not None:
                if step >= end_step:
                    w = w_final
                else:
                    w = w + (w_final - w) * (step / end_step)
                
            sky_loss_opacity = self.sky_opacity_loss_fn(pred_occupied_mask, gt_occupied_mask) * w
            loss_dict.update({"sky_loss_opacity": sky_loss_opacity})
        
        # depth loss
        if self.depth_loss_fn is not None and step > self.losses_dict.depth.start_step:
            if "lidar_depth_map" in image_infos:
                gt_depth = image_infos["lidar_depth_map"]
                lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
                if lidar_hit_mask.sum() == 0:
                    gt_depth = None
                pred_depth = outputs["depth"]
                if gt_depth is not None:
                    depth_loss = self.depth_loss_fn(pred_depth, gt_depth, lidar_hit_mask)
                else:
                    depth_loss = None
            else:
                depth_loss = None
            
            lidar_w_decay = self.losses_dict.depth.get("lidar_w_decay", -1)
            if lidar_w_decay > 0:
                decay_weight = np.exp(-self.step / 8000 * lidar_w_decay)
            else:
                decay_weight = 1
            if depth_loss is not None:
                depth_loss = depth_loss * self.losses_dict.depth.w * decay_weight
                loss_dict.update({"depth_loss": depth_loss})
        
        # mono depth loss
        if self.mono_depth_loss_fn is not None and step > self.losses_dict.mono_depth.start_step:
            gt_mono_depth = image_infos["mono_depths"]
            pred_depth = outputs["depth"]
            if "lidar_depth_map" in image_infos:
                gt_depth = image_infos["lidar_depth_map"]
                lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
                if lidar_hit_mask.sum() == 0:
                    lidar_hit_mask = None
            else:
                lidar_hit_mask = None
            
            # dilate the lidar hit mask to cover more area
            if lidar_hit_mask is not None:
                lidar_hit_mask = torch.nn.functional.max_pool2d(
                    lidar_hit_mask.unsqueeze(0).unsqueeze(0), 
                    kernel_size=15, stride=1, padding=7
                ).squeeze()
                non_lidar_mask = (1.0 - lidar_hit_mask).clamp(min=0.0, max=1.0)
            else:
                non_lidar_mask = None
            
            # mono_depth_loss = self.mono_depth_loss_fn(pred_depth, gt_mono_depth, 
            #                                           sky_mask=image_infos["sky_masks"])
            mono_depth_loss = self.mono_depth_loss_fn(
                pred_depth,
                gt_mono_depth,
                sky_mask=image_infos["sky_masks"],
                extra_loss_mask=non_lidar_mask,
            )
            # mono_depth_loss = self.mono_depth_loss_fn(
            #     pred_depth,
            #     gt_mono_depth,
            #     lidar_depth=image_infos.get("lidar_depth_map", None),
            # )
            mono_depth_loss = mono_depth_loss * self.losses_dict.mono_depth.w
            loss_dict.update({"mono_depth_loss": mono_depth_loss})
        
        # decrepated
        # rigid mono depth
        if self.rigid_mono_depth_loss_fn is not None and step > self.losses_dict.rigid_mono_depth.start_step and "RigidNodes_opacity" in outputs:
            gt_mono_depth = image_infos["mono_depths"]
            pred_depth = outputs["depth"]
            
            # compute loss in rigid region
            pred_occupied_mask_rigid = outputs["RigidNodes_opacity"].squeeze() * valid_loss_mask
            pred_occupied_mask_rigid = pred_occupied_mask_rigid.clamp(0.0, 1.0)
            
            rigid_mono_depth_loss = self.rigid_mono_depth_loss_fn(
                pred_depth, gt_mono_depth,
                sky_mask=image_infos["sky_masks"],
                extra_loss_mask=pred_occupied_mask_rigid
            )
            rigid_mono_depth_loss = rigid_mono_depth_loss * self.losses_dict.rigid_mono_depth.w
            loss_dict.update({"rigid_mono_depth_loss": rigid_mono_depth_loss})
            
        if self.depth_tv_loss_fn is not None:
            depth_tv_loss = self.depth_tv_loss_fn(outputs["depth"], gt_rgb=gt_rgb, mask=gt_occupied_mask)
            loss_dict.update({"depth_tv_loss": self.losses_dict.depth_tv.w * depth_tv_loss})

        # normal loss
        if self.normal_loss_fn is not None \
            and (self.losses_dict.normal.get("end_step", None) is None or step < self.losses_dict.normal.end_step):
            normal_loss = self.normal_loss_fn(pred_normal, gt_normal, mask=gt_occupied_mask)
            loss_dict.update({"normal_loss": self.losses_dict.normal.w * normal_loss})

        # normal cos loss
        if self.normal_cos_loss_fn is not None \
            and (self.losses_dict.normal_cos.get("end_step", None) is None or step < self.losses_dict.normal_cos.end_step):
            normal_cos_loss = self.normal_cos_loss_fn(pred_normal, gt_normal, mask=gt_occupied_mask)
            loss_dict.update({"normal_cos_loss": self.losses_dict.normal_cos.w * normal_cos_loss})
        
        # normal tv loss
        if self.normal_tv_loss_fn is not None:
            normal_tv_loss = self.normal_tv_loss_fn(pred_normal, gt_rgb=gt_rgb, mask=gt_occupied_mask)
            loss_dict.update({"normal_tv_loss": self.losses_dict.normal_tv.w * normal_tv_loss})
        
        # normal minscale loss
        if self.normal_minscale_loss_fn is not None and (self.losses_dict.normal_minscale.get("end_step", None) is None or step < self.losses_dict.normal_minscale.end_step):
            normal_minscale_loss = self.normal_minscale_loss_fn(pred_normal_minscale, gt_normal, mask=gt_occupied_mask)
            loss_dict.update({"normal_minscale_loss": self.losses_dict.normal_minscale.w * normal_minscale_loss})

        # normal minscale cos loss
        if self.normal_minscale_cos_loss_fn is not None:
            normal_minscale_cos_loss = self.normal_minscale_cos_loss_fn(pred_normal_minscale, gt_normal, mask=gt_occupied_mask)
            loss_dict.update({"normal_minscale_cos_loss": self.losses_dict.normal_minscale_cos.w * normal_minscale_cos_loss})
        
        # albedo loss
        if self.albedo_loss_fn is not None and self.losses_dict.albedo.w > 0 \
            and (self.losses_dict.albedo.get("end_step", None) is None or step < self.losses_dict.albedo.end_step):
            gt_albedo = image_infos["albedos"]
            pred_albedo = outputs["albedo"]
            albedo_loss = self.albedo_loss_fn(pred_albedo, gt_albedo, mask=gt_occupied_mask)
            
            w = self.losses_dict.albedo.w
            decay_start_step = self.losses_dict.albedo.get("decay_start_step", None)
            decay_end_step = self.losses_dict.albedo.get("decay_end_step", None)
            w_final = self.losses_dict.albedo.get("w_final", None)
            
            if decay_end_step is not None and decay_start_step is not None \
                    and step >= decay_start_step and w_final is not None:
                if step >= decay_end_step:
                    w = w_final
                else:
                    w = w + (w_final - w) * (step - decay_start_step) / (decay_end_step - decay_start_step)
            
            loss_stat_dict["albedo_w"] = w
            loss_dict.update({"albedo_loss": w * albedo_loss})
        
        # metallic loss
        if self.metallic_loss_fn is not None and self.losses_dict.metallic.w > 0 \
            and (self.losses_dict.metallic.get("end_step", None) is None or step < self.losses_dict.metallic.get("end_step", None)):
            gt_metallic = image_infos["metallics"]
            pred_metallic = outputs["metallic"].squeeze(-1)
            metallic_loss = self.metallic_loss_fn(pred_metallic, gt_metallic, mask=gt_occupied_mask)
            
            w = self.losses_dict.metallic.w
            decay_start_step = self.losses_dict.metallic.get("decay_start_step", None)
            decay_end_step = self.losses_dict.metallic.get("decay_end_step", None)
            w_final = self.losses_dict.metallic.get("w_final", None)
            
            if decay_end_step is not None and decay_start_step is not None \
                    and step >= decay_start_step and w_final is not None:
                if step >= decay_end_step:
                    w = w_final
                else:
                    w = w + (w_final - w) * (step - decay_start_step) / (decay_end_step - decay_start_step)
            
            loss_stat_dict["metallic_w"] = w
            loss_dict.update({"metallic_loss": w * metallic_loss})
            
        # roughness loss
        if self.roughness_loss_fn is not None and self.losses_dict.roughness.w > 0 \
            and (self.losses_dict.roughness.get("end_step", None) is None or step < self.losses_dict.roughness.get("end_step", None)):
            gt_roughness = image_infos["roughnesses"]
            pred_roughness = outputs["roughness"].squeeze(-1)
            roughness_loss = self.roughness_loss_fn(pred_roughness, gt_roughness, mask=gt_occupied_mask)
        
            w = self.losses_dict.roughness.w
            decay_start_step = self.losses_dict.roughness.get("decay_start_step", None)
            decay_end_step = self.losses_dict.roughness.get("decay_end_step", None)
            w_final = self.losses_dict.roughness.get("w_final", None)
            
            if decay_end_step is not None and decay_start_step is not None \
                    and step >= decay_start_step and w_final is not None:
                if step >= decay_end_step:
                    w = w_final
                else:
                    w = w + (w_final - w) * (step - decay_start_step) / (decay_end_step - decay_start_step)

            loss_stat_dict["roughness_w"] = w
            loss_dict.update({"roughness_loss": w * roughness_loss})
        
        # ----- reg loss -----
        opacity_entropy_reg = self.losses_dict.get("opacity_entropy", None)
        if opacity_entropy_reg is not None:
            pred_opacity = torch.clamp(outputs["opacity"].squeeze(), 1e-6, 1 - 1e-6)
            loss_dict.update({
                "opacity_entropy_loss": opacity_entropy_reg.w * (-pred_opacity * torch.log(pred_opacity)).mean()
            })
            
        # from pvg: https://github.com/fudan-zvg/PVG/blob/b4162a9135282e0f3c929054f16be1b3fbacd77a/train.py#L161
        inverse_depth_smoothness_reg = self.losses_dict.get("inverse_depth_smoothness", None)
        if inverse_depth_smoothness_reg is not None:
            inverse_depth = 1 / (outputs["depth"] + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
                inverse_depth[None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                image_infos["pixels"][None].permute(0, 3, 1, 2)
            )
            loss_dict.update({
                "inverse_depth_smoothness_loss": inverse_depth_smoothness_reg.w * loss_inv_depth
            })
            
        # affine reg loss
        affine_reg = self.losses_dict.get("affine", None)
        if affine_reg is not None and "Affine" in self.models:
            affine_trs = self.models['Affine']({"img_idx": image_infos["img_idx"].flatten()[0]})
            reg_mat = torch.eye(3, device=self.device)
            reg_shift = torch.zeros(3, device=self.device)
            loss_affine = torch.abs(affine_trs[..., :3, :3] - reg_mat).mean() + torch.abs(affine_trs[..., :3, 3:] - reg_shift).mean()
            loss_dict.update({
                "affine_loss": affine_reg.w * loss_affine
            })

        # dynamic region loss
        dynamic_region_weighted_losses = self.losses_dict.get("dynamic_region", None)
        if dynamic_region_weighted_losses is not None:
            weight_factor = dynamic_region_weighted_losses.get("w", 1.0)
            start_from = dynamic_region_weighted_losses.get("start_from", 0)
            if self.step == start_from:
                self.render_dynamic_mask = True
            if self.step > start_from and "Dynamic_opacity" in outputs:
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                
                if dynamic_pred_mask.sum() > 0:
                    Ll1 = torch.abs(gt_rgb[dynamic_pred_mask] - predicted_rgb[dynamic_pred_mask]).mean()
                    loss_dict.update({
                        "vehicle_region_rgb_loss": weight_factor * Ll1,
                    })
        
        # distortion map loss
        distortion_reg = self.losses_dict.get("distortion", None)
        if distortion_reg is not None and outputs["distortion_map"] is not None \
            and (self.losses_dict.distortion.get("start_step", None) is None or step > self.losses_dict.distortion.start_step):
            distortion_map = outputs["distortion_map"]
            
            # ignore the sky
            mask = (1.0 - image_infos["sky_masks"]).float()
            # only regularize the region with depth less than 10
            depth_mask = (outputs["depth"] < 10.0).float().squeeze(-1)
            distortion_map = distortion_map * mask * depth_mask

            distortion_loss = distortion_map.mean() * distortion_reg.w
            loss_dict.update({
                "distortion_loss": distortion_loss
            })
        
        depth_normal = None
        rendered_normal = None
        depth_normal_reg = self.losses_dict.get("depth_normal", None)
        if depth_normal_reg is not None and (self.losses_dict.depth_normal.get("start_step", None) is None or step > self.losses_dict.depth_normal.start_step):
            if self.gaussian_2d:
                rendered_normal_2dgs = self.info["2dgs_rendered_normal"]
                normals_from_depth = self.info["2dgs_surface_normal"]
                normals_from_depth *= outputs["opacity"].detach()
                normal_error = (1 - (rendered_normal_2dgs * normals_from_depth).sum(dim=-1))
                depth_normal_loss = normal_error.mean() * depth_normal_reg.w
                loss_dict.update({
                    "depth_normal_loss": depth_normal_loss
                })
            else:
                rendered_normal = outputs["world_normal"].detach()
                depth = outputs["depth"]
                depth_normal = depth_to_normal(
                    c2w=cam_infos["camera_to_world"].reshape(4, 4),
                    K=cam_infos["intrinsics"].reshape(3, 3),
                    depth_map=depth.squeeze(-1),
                    img_width=cam_infos["width"].item(),
                    img_height=cam_infos["height"].item(),
                )
                
                image_weight = (1.0 - get_img_grad_weight(image_infos["pixels"])) # (H, W)
                image_weight = (image_weight).clamp(0,1).detach() ** 2
                depth_normal_loss = depth_normal_reg.w * (gt_occupied_mask * image_weight * (((depth_normal - rendered_normal)).abs().sum(-1))).mean()
                depth_normal_cos_loss = depth_normal_reg.w * (gt_occupied_mask * image_weight * (1 - (depth_normal * rendered_normal).sum(-1))).mean()
                loss_dict.update({
                    "depth_normal_loss": depth_normal_loss,
                    "depth_normal_cos_loss": depth_normal_cos_loss,
                })
        
        # deprecated
        if self.road_normal_loss_fn is not None:
            if "road_masks" in image_infos:
                road_mask = image_infos["road_masks"]
                
                if depth_normal is None:
                    depth = outputs["depth"]
                    depth_normal = depth_to_normal(
                        c2w=cam_infos["camera_to_world"].reshape(4, 4),
                        K=cam_infos["intrinsics"].reshape(3, 3),
                        depth_map=depth.squeeze(-1),
                        img_width=cam_infos["width"].item(),
                        img_height=cam_infos["height"].item(),
                    )
                if rendered_normal is None:
                    rendered_normal = outputs["world_normal"].detach()
                
                # road_normal_loss = self.road_normal_loss_fn(
                #     depth_normal,
                #     road_mask=road_mask,
                # )
                
                road_normal_loss = (road_mask.float() * (((depth_normal - rendered_normal)).abs().sum(-1))).mean()
                road_normal_cos_loss = (road_mask.float() * (1 - (depth_normal * rendered_normal).sum(-1))).mean()
                
                loss_dict.update({"road_normal_loss": self.losses_dict.road_normal.w * road_normal_loss})
                loss_dict.update({"road_normal_cos_loss": self.losses_dict.road_normal.w * road_normal_cos_loss})
            elif self.losses_dict.road_normal.w > 0:
                print("Warning: road normal loss weight > 0 but no road mask provided!")
        
    
        # compute gaussian reg loss
        for class_name in self.gaussian_classes.keys():
            class_reg_loss = self.models[class_name].compute_reg_loss()
            for k, v in class_reg_loss.items():
                loss_dict[f"{class_name}_{k}"] = v
        
        # check NaN
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                logger.warning(f"Loss {k} is NaN")
                import pdb; pdb.set_trace()
        
        
        return loss_dict, loss_stat_dict
    
    def compute_multiview_depth_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
        nearest_image_infos: Dict[str, torch.Tensor],
        nearest_cam_infos: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        multiview_depth_reg = self.losses_dict.get("multiview_depth", None)
        if multiview_depth_reg is None \
            or (multiview_depth_reg is not None and self.step < multiview_depth_reg.start_step and multiview_depth_reg.w > 0):
            return torch.tensor(0.0, device=self.device)
    
        pixel_noise_th = multiview_depth_reg.get("pixel_noise_th", 1.0)
        H, W = outputs["depth"].shape[:2]
        ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        pixels = (
            torch.stack([ix, iy], dim=-1).float().to(outputs["depth"].device)
        )
        
        processed_cam = self.process_camera(
            camera_infos=cam_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=False
        )
        
        nearest_processed_cam = self.process_camera(
            camera_infos=nearest_cam_infos,
            image_ids=nearest_image_infos["img_idx"].flatten()[0],
            novel_view=False
        )
        gs = self.collect_gaussians(
            cam=nearest_processed_cam,
            image_ids=nearest_image_infos["img_idx"].flatten()[0]
        )

        # render gaussians. Note that even if we call it second time here, the render_fn obtained in the forward function still use the viewpoint camera.
        nearest_outputs, _ = self.render_gaussians(
            gs=gs,
            cam=nearest_processed_cam,
            update_info=False, # just render, no need to update info
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.),            
        )
        
        depth = outputs["depth"].squeeze(-1)
        K = processed_cam.Ks
        c2w = processed_cam.camtoworlds
        
        pts = deproject_depth(
            depth_map=depth,
            K=K,
            c2w=c2w,
            img_width=W,
            img_height=H
        ) # (N, 3)
        
        map_z, d_mask = sample_depth_at_world_points(
            depth_map=nearest_outputs["depth"].squeeze(-1),
            K=nearest_processed_cam.Ks,
            c2w=nearest_processed_cam.camtoworlds,
            points_world=pts,
            img_width=W,
            img_height=H
        ) # (1, N), (N, )
        
        pts_projections, _, _, d_mask_ = reproject_world_points_via_depth(
            points_world=pts,
            map_z=map_z,
            K_near=nearest_processed_cam.Ks,
            c2w_near=nearest_processed_cam.camtoworlds,
            K_view=processed_cam.Ks,
            c2w_view=processed_cam.camtoworlds,
        )
        d_mask &= d_mask_
        
        pixel_noise = torch.norm(
            pts_projections - pixels.reshape(*pts_projections.shape), dim=-1
        )
        
        valid_mask = (1-image_infos["sky_masks"]).bool() & (outputs["opacity"].squeeze() > 1e-6)
        if "dynamic_masks" in image_infos:
            valid_mask &= (1 - image_infos["dynamic_masks"]).bool()
        if "human_masks" in image_infos:
            valid_mask &= (1 - image_infos["human_masks"]).bool()
        if "vehicle_masks" in image_infos:
            valid_mask &= (1 - image_infos["vehicle_masks"]).bool()
            
        d_mask = d_mask & (pixel_noise < pixel_noise_th) &  valid_mask.reshape(-1)
        weights = (1.0 / torch.exp(pixel_noise)).detach()
        weights[~d_mask] = 0
        
        multiview_depth_loss = torch.tensor(0.0, device=self.device)
        if d_mask.sum() > 0:
            multiview_depth_loss = multiview_depth_reg.w * ((weights * pixel_noise)[d_mask]).mean()
        
        
        return multiview_depth_loss
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        psnr = self.psnr(outputs["rgb"], image_infos["pixels"])
        metric_dict.update({"psnr": psnr})
        return metric_dict
    
    def get_gaussian_count(self):
        num_dict = {}
        for class_name in self.gaussian_classes.keys():
            num_dict[class_name] = self.models[class_name].num_points
        return num_dict
    
    def state_dict(self, only_model: bool = True):
        state_dict = super().state_dict()
        state_dict.update({
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "step": self.step,
        })
        if not only_model:
            state_dict.update({
                "optimizer": {k: v.state_dict() for k, v in self.optimizer.items()},
                # "lr_schedulers": {k: v.state_dict() for k, v in self.lr_schedulers.items()},
                # "grad_scaler": self.grad_scaler.state_dict(),
            })
        return state_dict

    def load_state_dict(self, state_dict: dict, load_only_model: bool =True, strict: bool = True):
        step = state_dict.pop("step")
        self.step = step
        logger.info(f"Loading checkpoint at step {step}")

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)
        
        # load model
        model_state_dict = state_dict.pop("models")
        
        nomatch_classes = []
        for class_name in self.models.keys():
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                logger.warning(f"Cannot find {class_name} in the checkpoint")
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                    logger.warning(f"Remove {class_name} from the model")
                
                nomatch_classes.append(class_name)
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
            logger.info(f"{class_name}: {msg}")
        
        for class_name in nomatch_classes:
            del self.models[class_name]
            if class_name in self.model_config.keys():
                del self.model_config[class_name]
        
        msg = super().load_state_dict(state_dict, strict)
        logger.info(f"BasicTrainer: {msg}")
    
    def freeze_geometry(self) -> None:
        for model in self.models.values():
            if hasattr(model, "freeze_geometry") and callable(model.freeze_geometry):
                logger.info(f"Freezing geometry for {model.__class__.__name__}")
                model.freeze_geometry(self.optimizer)
    
    def freeze_material(self) -> None:
        for model in self.models.values():
            if hasattr(model, "freeze_material") and callable(model.freeze_material):
                logger.info(f"Freezing material for {model.__class__.__name__}")
                model.freeze_material(self.optimizer)

    def freeze_envmap(self) -> None:
        logger.info("Freezing environment map")
        sky_model = self.models.get('Sky', None)
        if sky_model is None:
            logger.warning("Sky model not found; skipping envmap freeze.")
            return
        if hasattr(sky_model, "freeze_envmap") and callable(sky_model.freeze_envmap):
            sky_model.freeze_envmap(self.optimizer)
        else:
            logger.warning("Sky model does not support freeze_envmap method.")
    
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path)
        # self.load_state_dict(state_dict, load_only_model=load_only_model, strict=True)
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=False) # since we have new features
        
    def save_checkpoint(
        self,
        log_dir: str,
        save_only_model: bool=True,
        is_final: bool=False
    ) -> None:
        """
        Save model to checkpoint.
        """
        if is_final:
            ckpt_path = os.path.join(log_dir, f"checkpoint_final.pth")
        else:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{self.step:05d}.pth")
        torch.save(self.state_dict(only_model=save_only_model), ckpt_path)
        logger.info(f"Saved a checkpoint to {ckpt_path}")
        
    def init_viewer(self, port: int = 8080):
        # a simple viewer for background ONLY visualization
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    # TBD: if needed render extra feature
    # TBD: 2DGS rendering
    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        
        cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w,
            Ks=K,
            H=H,
            W=W
        )
        
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
        }
        for class_name in ["Background"]:
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue

            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)

        gs = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],
            extras=None
        )
        
        render_colors, _, _ = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities.squeeze(-1),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
            Ks=cam.Ks[None, ...],  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            radius_clip=4.0,  # skip GSs that have small image radius (in pixels)
        )
        return render_colors[0].cpu().numpy()
