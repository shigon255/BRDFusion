from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter

from models.gaussians.basics import *

logger = logging.getLogger()

class VanillaGaussians(nn.Module):

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0
        self.freeze_geometry_flag = False
        self.freeze_material_flag = False
        
        self.device = device
        self.ball_gaussians=self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        
        # for evaluation
        self.in_test_set = False
        
        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)
        
        self._normals = torch.zeros(1, 3, device=self.device)
        self._albedos = torch.zeros(1, 3, device=self.device)
        self._roughnesses = torch.zeros(1, 1, device=self.device)
        self._metallics = torch.zeros(1, 1, device=self.device)
        
        self.geometry_keys = [
            # basic gaussian
            "xyz",
            "opacity",
            "scaling",
            "rotation",
            "normal",
            # rigid
            "ins_rotation",
            "ins_translation",
            # deformable
            "embedding",
            "deform_network",
            # smpl
            "smpl_rotation"
        ]
        
        self.material_keys = [
            "albedo",
            "roughness",
            "metallic",
        ]
        
        # print("Debugging: also freeze material")
        # self.geometry_keys = [
        #     # basic gaussian
        #     "xyz",
        #     "opacity",
        #     "scaling",
        #     "rotation",
        #     "normal",
        #     "albedo",
        #     "roughness",
        #     "metallic",
        #     # rigid
        #     "ins_rotation",
        #     "ins_translation",
        #     # deformable
        #     "embedding",
        #     "deform_network",
        #     # smpl
        #     "smpl_rotation"
        # ]
        
    @property
    def sh_degree(self):
        return self.ctrl_cfg.sh_degree

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        self._means = Parameter(init_means)
        
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        if self.ball_gaussians:
            self._scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))
        
        self._normals = Parameter(torch.ones(self.num_points, 3, device=self.device))
        self._albedos = Parameter(torch.ones(self.num_points, 3, device=self.device))
        self._roughnesses = Parameter(torch.ones(self.num_points, 1, device=self.device))
        self._metallics = Parameter(torch.ones(self.num_points, 1, device=self.device))
        
        
    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)
    @property
    def shs_0(self):
        return self._features_dc
    @property
    def shs_rest(self):
        return self._features_rest
    @property
    def num_points(self):
        return self._means.shape[0]
    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                # pad scale 0 for 3DGS consistency
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales)
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)
    @property
    def get_quats(self):
        return self.quat_act(self._quats)
    
    @property
    def get_normals(self):
        return self.normal_act(self._normals)
    
    def get_normals_no_grad(self):
        with torch.no_grad():
            return self.normal_act(self._normals)
    
    @property
    def get_albedos(self):
        return self.albedo_act(self._albedos)

    @property
    def get_roughnesses(self):
        return self.roughness_act(self._roughnesses)
    
    @property
    def get_metallics(self):
        return self.metallic_act(self._metallics)
    
    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)
    
    def normal_act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
    
    def albedo_act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def roughness_act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def metallic_act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
    def preprocess_per_train_step(self, step: int):
        self.step = step
        
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        self.after_train(radii, xys_grad, last_size)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)

    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
            self.class_prefix+"normal": [self._normals],
            self.class_prefix+"albedo": [self._albedos],
            self.class_prefix+"roughness": [self._roughnesses],
            self.class_prefix+"metallic": [self._metallics],
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    
    def freeze_geometry(self, optimizer: torch.optim.Optimizer = None) -> None:
        self.freeze_geometry_flag = True
        self.freeze_params(self.geometry_keys, optimizer=optimizer)
    
    def freeze_material(self, optimizer: torch.optim.Optimizer = None) -> None:
        self.freeze_material_flag = True
        self.freeze_params(self.material_keys, optimizer=optimizer)

    # set the parameters to not require gradients and zeroing out their optimizer states
    def freeze_params(self, param_keys: List[str], optimizer: torch.optim.Optimizer = None) -> None:
        frozen_count = 0
        for param_key in param_keys:
            for name, param_group in self.get_param_groups().items():
                # component name is after #
                component_name = name.split("#")[1]
                if param_key == component_name:
                    for param in param_group:
                        if isinstance(param, Parameter):
                            param.requires_grad_(False)
                            param.grad = None
                            frozen_count += 1
                            try:
                                if optimizer is not None:
                                    for group in optimizer.param_groups:
                                        if any(param is p for p in group['params']):
                                            if param in optimizer.state:
                                                # NOTE: deleting the whole state is safe because optimizer.state is a defaultdict. When dup_in_optim acess this state, it will obtain a new empty state.
                                                del optimizer.state[param]
                                                print(f"Deleted optimizer state for {name}")
                                                # state = optimizer.state[param]
                                                # for k in state:
                                                #     if isinstance(state[k], torch.Tensor):
                                                #         # print(f"Zeroing out {k} in optimizer state for {name}")
                                                #         # state[k].zero_()
                            except Exception as e:
                                print(f"Error while freezing {name} parameters: {e}")
                                import pdb; pdb.set_trace()
                        elif isinstance(param, list):
                            for p in param:
                                if isinstance(p, Parameter):
                                    p.requires_grad_(False)
                                    p.grad = None
                                    frozen_count += 1
                                    if optimizer is not None:
                                        for group in optimizer.param_groups:
                                            if any(p is pp for pp in group['params']):
                                                if p in optimizer.state:
                                                    del optimizer.state[p]
                                                    print(f"Deleted optimizer state for {name}")
                                                    # state = optimizer.state[p]
                                                    # for k in state:
                                                    #     if isinstance(state[k], torch.Tensor):
                                                    #         print(f"Zeroing out {k} in optimizer state for {name}")
                                                    #         state[k].zero_()
                        else:
                            print(f"Unrecognised parameter type {type(param)} for {param_key} in {name}")
                            import pdb; pdb.set_trace()
                        print(f"Frozen {name} parameters (requires_grad={param.requires_grad if isinstance(param, Parameter) else 'N/A'})")
        print(f"Total frozen parameters: {frozen_count}")
    
    # debugging
    def mask_gs(self, mask_func):
        print("Debugging: masking gaussians...")
        xyz = self._means.data
        mask = mask_func(xyz)
        # mask out all gs parameter
        self._means = Parameter(self._means.data[mask].clone())
        self._scales = Parameter(self._scales.data[mask].clone())
        self._quats = Parameter(self._quats.data[mask].clone())
        self._opacities = Parameter(self._opacities.data[mask].clone())
        self._features_dc = Parameter(self._features_dc.data[mask].clone())
        self._features_rest = Parameter(self._features_rest.data[mask].clone())
        self._normals = Parameter(self._normals.data[mask].clone())
        self._albedos = Parameter(self._albedos.data[mask].clone())
        self._roughnesses = Parameter(self._roughnesses.data[mask].clone())
        self._metallics = Parameter(self._metallics.data[mask].clone())
        print(f"Masking out {xyz.shape[0] - self.num_points} gaussians, left {self.num_points} gaussians")

    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # when freezing geometry, skip split/cull
            if not self.freeze_geometry_flag:
                # only split/cull if we've seen every image since opacity reset
                reset_interval = self.ctrl_cfg.reset_alpha_interval
                do_densification = (
                    self.step < self.ctrl_cfg.stop_split_at
                    and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
                )
                # split & duplicate
                print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
                if do_densification:
                    assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                    
                    avg_grad_norm = self.xys_grad_norm / self.vis_counts
                    print(f"Class {self.class_prefix} avg grad norm stats: min {avg_grad_norm.min().item():.6f}, max {avg_grad_norm.max().item():.6f}, mean {avg_grad_norm.mean().item():.6f}")
                    high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                    
                    # print(f"Class {self.class_prefix} avg grad norm stats: min {avg_grad_norm.min().item():.6f}, max {avg_grad_norm.max().item():.6f}, mean {avg_grad_norm.mean().item():.6f}")
                    
                    # split the gaussians that are large 
                    splits = (
                        self.get_scaling.max(dim=-1).values > \
                            self.ctrl_cfg.densify_size_thresh * self.scene_scale
                    ).squeeze()
                    
                    # print(f"Class {self.class_prefix} scale stats: min {self.get_scaling.min().item():.6f}, max {self.get_scaling.max().item():.6f}, mean {self.get_scaling.mean().item():.6f}")
                    
                    # split the gaussians that are large in screen space
                    if self.step < self.ctrl_cfg.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                    
                    # split the gaussians that also have high grads
                    splits &= high_grads
                    nsamps = self.ctrl_cfg.n_split_samples
                    (
                        split_means,
                        split_feature_dc,
                        split_feature_rest,
                        split_opacities,
                        split_scales,
                        split_quats,
                        split_normals,
                        split_albedos,
                        split_roughnesses,
                        split_metallics,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (
                        self.get_scaling.max(dim=-1).values <= \
                            self.ctrl_cfg.densify_size_thresh * self.scene_scale
                    ).squeeze()
                    dups &= high_grads
                    (
                        dup_means,
                        dup_feature_dc,
                        dup_feature_rest,
                        dup_opacities,
                        dup_scales,
                        dup_quats,
                        dup_normals,
                        dup_albedos,
                        dup_roughnesses,
                        dup_metallics,
                    ) = self.dup_gaussians(dups)
                    
                    self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                    # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                    self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                    self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                    self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                    self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                    self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                    self._normals = Parameter(torch.cat([self._normals.detach(), split_normals, dup_normals], dim=0))
                    self._albedos = Parameter(torch.cat([self._albedos.detach(), split_albedos, dup_albedos], dim=0))
                    self._roughnesses = Parameter(torch.cat([self._roughnesses.detach(), split_roughnesses, dup_roughnesses], dim=0))
                    self._metallics = Parameter(torch.cat([self._metallics.detach(), split_metallics, dup_metallics], dim=0))
                    
                    # TODO: write things to be put on the slide
                    # if freezing material, set requires_grad to be False for new gaussians
                    if self.freeze_material_flag:
                        self._albedos.requires_grad_(False)
                        self._roughnesses.requires_grad_(False)
                        self._metallics.requires_grad_(False)
                    
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                        dim=0,
                    )
                    
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_gaussian_param_groups()
                    dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                    dup_idcs = torch.where(dups)[0]
                    param_groups = self.get_gaussian_param_groups()
                    dup_in_optim(optimizer, dup_idcs, param_groups, 1)

                # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                    # save checkpoints right when the opacity is reset (saves every 2k)
                if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    remove_from_optim(optimizer, deleted_mask, param_groups)
                print(f"Class {self.class_prefix} left points: {self.num_points}")
                        
                # reset opacity
                if self.step % reset_interval == self.ctrl_cfg.refine_interval \
                    and self.step < self.ctrl_cfg.stop_reset_alpha_at:
                    # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                        # we align to original repo of gaussians spalting
                    reset_value = torch.min(self.get_opacity.data,
                                            torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                    self._opacities.data = torch.logit(reset_value)
                    # reset the exp of optimizer
                    for group in optimizer.param_groups:
                        if group["name"] == self.class_prefix+"opacity":
                            old_params = group["params"][0]
                            param_state = optimizer.state[old_params]
                            param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                            param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self._normals = Parameter(self._normals[~culls].detach())
        self._albedos = Parameter(self._albedos[~culls].detach())
        self._roughnesses = Parameter(self._roughnesses[~culls].detach())
        self._metallics = Parameter(self._metallics[~culls].detach())
        print(f"     Cull: {n_bef - self.num_points}")
        
        assert not self.freeze_geometry_flag, "Should not be culling when geometry is frozen"
        
        if self.freeze_material_flag:
            self._albedos.requires_grad_(False)
            self._roughnesses.requires_grad_(False)
            self._metallics.requires_grad_(False)
        
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        
        new_normals = self._normals[split_mask].repeat(samps, 1)
        new_albedos = self._albedos[split_mask].repeat(samps, 1)
        new_roughnesses = self._roughnesses[split_mask].repeat(samps, 1)
        new_metallics = self._metallics[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_normals, new_albedos, new_roughnesses, new_metallics

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_normals = self._normals[dup_mask]
        dup_albedos = self._albedos[dup_mask]
        dup_roughnesses = self._roughnesses[dup_mask]
        dup_metallics = self._metallics[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_normals, dup_albedos, dup_roughnesses, dup_metallics

    def get_normals_minscale(self, scales, quats):
        if scales.numel() == 0:
            return torch.empty((0, 3), dtype=scales.dtype, device=scales.device)
        rotations_mat = quat_to_rotmat(self.quat_act(quats))
        min_scales = torch.argmin(scales, dim=-1)
        indices = torch.arange(min_scales.shape[0], device=min_scales.device)
        normals = rotations_mat[indices, :, min_scales]
        return self.normal_act(normals)

    def get_normals_maxscale(self, scales, quats):
        if scales.numel() == 0:
            return torch.empty((0, 3), dtype=scales.dtype, device=scales.device)
        rotations_mat = quat_to_rotmat(self.quat_act(quats))
        max_scales = torch.argmax(scales, dim=-1)
        indices = torch.arange(max_scales.shape[0], device=max_scales.device)
        normals = rotations_mat[indices, :, max_scales]
        return self.normal_act(normals)

    def get_min_max_scaling(self, scales):
        min_scales = torch.min(scales, dim=-1).values
        max_scales = torch.max(scales, dim=-1).values
        return min_scales, max_scales

    def get_min_max_axis_dir(self, scales, quats):
        if scales.numel() == 0:
            empty = torch.empty((0, 3), dtype=scales.dtype, device=scales.device)
            return empty, empty
        rotations_mat = quat_to_rotmat(self.quat_act(quats))
        min_scales = torch.argmin(scales, dim=-1)
        max_scales = torch.argmax(scales, dim=-1)
        min_indices = torch.arange(min_scales.shape[0], device=min_scales.device)
        max_indices = torch.arange(max_scales.shape[0], device=max_scales.device)
        min_axis = rotations_mat[min_indices, :, min_scales]
        max_axis = rotations_mat[max_indices, :, max_scales]
        return self.normal_act(min_axis), self.normal_act(max_axis)

    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        activated_colors = rgbs
        activated_normals = self.get_normals
        activated_albedos = self.get_albedos
        activated_roughnesses = self.get_roughnesses
        activated_metallics = self.get_metallics
        
        # show the shortest axis of transformed gaussians
        normals_minscale = self.get_normals_minscale(activated_scales, activated_rotations).detach()
        dotprod = torch.sum(-viewdirs * normals_minscale, dim=1, keepdim=True)
        normals_minscale = torch.where(dotprod > 0, normals_minscale, -normals_minscale)

        # collect gaussians information
        gs_dict = dict(
            _means=self._means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=activated_colors[filter_mask],
            _features_dc=self._features_dc[filter_mask],
            _features_rest=self._features_rest[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
            _normals=activated_normals[filter_mask],
            _albedos=activated_albedos[filter_mask],
            _roughnesses=activated_roughnesses[filter_mask],
            _metallics=activated_metallics[filter_mask],
            _normals_minscale=normals_minscale[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                print(f"NaN detected in gaussian {k} at step {self.step}")
                import pdb; pdb.set_trace()
                # raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                print(f"Inf detected in gaussian {k} at step {self.step}")
                import pdb; pdb.set_trace()
                # raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
    
    # get gaussian independent of camera for exporting ply
    # transform gaussians to world space
    # collect xyz, normals, f_dc, f_rest, opacities, scales, quats, albedos, roughnesses, metallics
    def get_gaussian_indep(self):
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
            
        # don't activate to prevent double activation (sigmoid)
        means = self._means
        opacities = self._opacities
        scales = self._scales
        if self.gaussian_2d:
            # pad scale 0 for 3DGS consistency
            # but now in log scale, pad -inf
            scales = torch.cat([scales[..., :2], torch.zeros_like(scales[..., :1]) - 1e6], dim=-1)
        quats = self._quats
        f_dcs = self._features_dc
        f_rests = self._features_rest
        normals = self._normals
        albedos = self._albedos
        roughnesses = self._roughnesses
        metallics = self._metallics
        normals_minscale = self.get_normals_minscale(torch.exp(scales), self.quat_act(quats)).detach()
        
        min_scale, max_scale = self.get_min_max_scaling(torch.exp(scales))
        
        # collect gaussians information
        gs_dict = dict(
            _means=means[filter_mask],
            _opacities=opacities[filter_mask],
            _features_dc=f_dcs[filter_mask],
            _features_rest=f_rests[filter_mask],
            _scales=scales[filter_mask],
            _quats=quats[filter_mask],
            _normals=normals[filter_mask],
            _albedos=albedos[filter_mask],
            _roughnesses=roughnesses[filter_mask],
            _metallics=metallics[filter_mask],
            _normals_minscale=normals_minscale[filter_mask],
            _min_scales=min_scale[filter_mask],
            _max_scales=max_scale[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
    
    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            if self.gaussian_2d:
                # sharp shape regularization is not compatible with 2D Gaussians, skipping...
                pass
            else:
                w = sharp_shape_reg_cfg.w
                max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
                step_interval = sharp_shape_reg_cfg.step_interval
                if self.step % step_interval == 0:
                    # scale regularization
                    scale_exp = self.get_scaling
                    scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                    scale_reg = scale_reg.mean() * w
                    loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            if self.gaussian_2d:
                print("Warning: using 2DGS, which don't need flattent reg (but still computed)")
            if (self.cur_radii > 0).sum():
                sclaings = self.get_scaling
                min_scale, _ = torch.min(sclaings, dim=1)
                min_scale = torch.clamp(min_scale, 0, 30)
                flatten_loss = torch.abs(min_scale)[self.cur_radii > 0].mean()
                loss_dict["flatten"] = flatten_loss * flatten_reg.w
        
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        
        # from DeSiRe-GS
        maxscale_reg = self.reg_cfg.get("maxscale_reg", None)
        if maxscale_reg is not None:
            if (self.cur_radii > 0).sum():
                w = maxscale_reg.w
                max_allowed_scale = maxscale_reg.max_allowed_scale
                maxscale, _ = torch.max(self.get_scaling, dim=1)
                maxscale_loss = torch.relu(maxscale - max_allowed_scale)
                valid_maxscale_mask = maxscale_loss > 0 & (self.cur_radii > 0).squeeze()
                maxscale_loss = w * maxscale_loss[valid_maxscale_mask].mean()
                loss_dict["maxscale_reg"] = maxscale_loss
                
        
        normal_align_reg = self.reg_cfg.get("normal_align_reg", None)
        if normal_align_reg is not None:
            w = normal_align_reg.w
            normals = self.get_normals_no_grad()
            # normals = self.get_normals
            normals_minscale = self.get_normals_minscale(self.get_scaling, self.get_quats)
            # align the minscale and the normals with L1 loss
            normal_align_loss = torch.mean(torch.abs(normals - normals_minscale)) * w
            loss_dict["normal_align_reg"] = normal_align_loss
        
        normal_align_cos_reg = self.reg_cfg.get("normal_align_cos_reg", None)
        if normal_align_cos_reg is not None:
            w = normal_align_cos_reg.w
            normals = self.get_normals_no_grad()
            # normals = self.get_normals
            normals_minscale = self.get_normals_minscale(self.get_scaling, self.get_quats)
            # align the minscale and the normals with cosine similarity
            normal_align_cos_loss = torch.mean(1 - torch.sum(normals * normals_minscale, dim=1, keepdim=True)) * w
            loss_dict["normal_align_cos_reg"] = normal_align_cos_loss
        
        sum_opacity_reg = self.reg_cfg.get("sum_opacity_reg", None)
        if sum_opacity_reg is not None:
            w = sum_opacity_reg.w
            sum_opacity_loss = torch.sum(self.get_opacity) * w
            loss_dict["sum_opacity_reg"] = sum_opacity_loss
        
        return loss_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        if self.gaussian_2d and state_dict["_scales"].shape[1] == 3:
            print("Warning: loading 3D scales into 2D Gaussians, setting the scale z to zero")
            state_dict["_scales"] = state_dict["_scales"][:, :2]
        
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        self._normals = Parameter(torch.zeros((N,) + self._normals.shape[1:], device=self.device))
        self._albedos = Parameter(torch.zeros((N,) + self._albedos.shape[1:], device=self.device))
        self._roughnesses = Parameter(torch.zeros((N,) + self._roughnesses.shape[1:], device=self.device))
        self._metallics = Parameter(torch.zeros((N,) + self._metallics.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    
    def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
        means = self._means
        direct_color = self.colors
        
        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }