from typing import Dict, List, Tuple
import logging

import torch
from torch.nn import Parameter

from models.modules import ConditionalDeformNetwork
from models.gaussians.basics import *
from models.nodes.rigid import RigidNodes

logger = logging.getLogger()


class DeformableNodes(RigidNodes):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.instances_embedding = torch.zeros(1, self.networks_cfg.embed_dim, device=self.device)
        self.deform_network = ConditionalDeformNetwork(
            input_ch=3, **self.networks_cfg
        ).to(self.device)

    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        super().create_from_pcd(instance_pts_dict=instance_pts_dict)
        init_embedding = torch.rand(self.num_instances, self.networks_cfg.embed_dim, device=self.device)
        self.instances_embedding = Parameter(init_embedding) # overrided the previous one
        
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"embedding"] = [self.instances_embedding]
        param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        return param_groups

    def get_deformation_rotation_matrix(self, u1, u2, v1, v2):
        """
        u1, u2: min/max axis of the gaussian before deformation
        v1, v2: min/max axis of the gaussian after deformation
        """
        u1 = u1 / u1.norm(dim=1, keepdim=True)
        u2 = u2 / u2.norm(dim=1, keepdim=True)
        v1 = v1 / v1.norm(dim=1, keepdim=True)
        v2 = v2 / v2.norm(dim=1, keepdim=True)
        # assert not torch.any(is_parallel(u1, v1)), "u1 contains vectors parallel to vectors in v1"
        # assert not torch.any(is_parallel(u2, v2)), "u2 contains vectors parallel to vectors in v2"
        # assert not torch.any(torch.allclose(u1, v1)), "u1 contains vectors parallel to vectors in v1"
        # assert not torch.any(torch.allclose(u2, v2)), "u2 contains vectors parallel to vectors in v2"
        
        u3 = torch.cross(u1, u2, dim=1)
        v3 = torch.cross(v1, v2, dim=1)

        U = torch.stack([u1, u2, u3], dim=2)
        V = torch.stack([v1, v2, v3], dim=2)

        R = torch.matmul(V, U.transpose(1, 2))
        return R

    def get_deformation(self, local_means) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """
        assert local_means.shape[0] == self.point_ids.shape[0], \
            "its a bug here, we need to pass the mask for points_ids"
        nonrigid_embed = self.instances_embedding[self.point_ids[..., 0]]
        ins_height = self.instances_size[self.point_ids[..., 0]][..., 2]
        x = local_means.data / ins_height[:, None] * 2
        t = self.get_current_normed_time()
        t = t.unsqueeze(0).repeat(self.point_ids.shape[0], 1)
        delta_xyz, delta_quat, delta_scale = self.deform_network(x, t, nonrigid_embed)
        if self.gaussian_2d:
            delta_scale[..., 2] = 0.0
        return delta_xyz, delta_quat, delta_scale
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.ctrl_cfg.use_deformgs_for_nonrigid and self.step > self.ctrl_cfg.use_deformgs_after:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(local_means=self._means)
        
        if delta_xyz is not None:
            if self.ctrl_cfg.stop_optimizing_canonical_xyz:
                means = self._means.data + delta_xyz
            else:
                means = self._means + delta_xyz
            world_means = self.transform_means(means)
        else:
            world_means = self.transform_means(self._means)
        
        quats = None
        if delta_quat is not None:
            quats = self.get_quats + delta_quat
            world_quats = self.transform_quats(quats)
        else:
            world_quats = self.transform_quats(self._quats)
        
        if delta_scale is not None:
            activated_scales = self.get_scaling + delta_scale
        else:
            activated_scales = self.get_scaling
        
        # TBD: deform the normal beform transforming
        before_scale = self.get_scaling.detach()
        before_quat = self.get_quats.detach()
        after_scale = activated_scales.detach()
        if quats is not None:
            after_quat = self.normal_act(quats.detach())
        else:
            after_quat = self.normal_act(world_quats.detach())
        before_min_axis, before_max_axis = self.get_min_max_axis_dir(
            before_scale, before_quat
        )
        after_min_axis, after_max_axis = self.get_min_max_axis_dir(
            after_scale, after_quat
        )
        
        dotprod_min_axis = torch.sum(before_min_axis * after_min_axis, dim=1, keepdim=True)
        dotprod_max_axis = torch.sum(before_max_axis * after_max_axis, dim=1, keepdim=True)
        flip_mask_min_axis = dotprod_min_axis <= 0
        flip_mask_max_axis = dotprod_max_axis <= 0
        after_min_axis = torch.where(flip_mask_min_axis, -after_min_axis, after_min_axis)
        after_max_axis = torch.where(flip_mask_max_axis, -after_max_axis, after_max_axis)
        
        deformation_R = self.get_deformation_rotation_matrix(
            u1=before_min_axis, u2=before_max_axis, v1=after_min_axis, v2=after_max_axis
        )
        world_normals = torch.bmm(deformation_R, self.normal_act(self.get_normals).unsqueeze(-1)).squeeze(-1)
        world_normals = self.transform_normals(world_normals)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        valid_mask = self.get_pts_valid_mask()
            
        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        activated_normals = self.normal_act(world_normals)
        activated_albedos = self.get_albedos
        activated_roughnesses = self.get_roughnesses
        activated_metallics = self.get_metallics
        
        # get shortest axis of transformed gaussians
        normals_minscale = self.get_normals_minscale(activated_scales, activated_rotations).detach()
        dotprod = torch.sum(-viewdirs * normals_minscale, dim=1, keepdim=True)
        normals_minscale = torch.where(dotprod > 0, normals_minscale, -normals_minscale)
        
        # for debugging transformation
        # normals_local_minscale = self.get_normals_minscale(self.get_scaling, self.get_quats).detach()
        # normals_local_minscale = torch.bmm(deformation_R, normals_local_minscale.unsqueeze(-1)).squeeze(-1)
        # normals_local_minscale = self.transform_normals(self.normal_act(normals_local_minscale))
        # dotprod = torch.sum(-viewdirs * normals_local_minscale, dim=1, keepdim=True)
        # normals_local_minscale = torch.where(dotprod > 0, normals_local_minscale, -normals_local_minscale)
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
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

        # check nan in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
            "local_xyz_deformed": means[filter_mask] if delta_xyz is not None else None,
        }
        return gs_dict

    # for exporting ply
    def get_gaussian_indep(self) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.ctrl_cfg.use_deformgs_for_nonrigid and self.step > self.ctrl_cfg.use_deformgs_after:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(local_means=self._means)
        
        if delta_xyz is not None:
            if self.ctrl_cfg.stop_optimizing_canonical_xyz:
                means = self._means.data + delta_xyz
            else:
                means = self._means + delta_xyz
            world_means = self.transform_means(means)
        else:
            world_means = self.transform_means(self._means)
        
        quats = None
        if delta_quat is not None:
            quats = self.get_quats + delta_quat
            world_quats = self.transform_quats(quats)
        else:
            world_quats = self.transform_quats(self._quats)
        
        if delta_scale is not None:
            activated_scales = self.get_scaling + delta_scale
        else:
            activated_scales = self.get_scaling
        
        # TBD: deform the normal beform transforming
        before_scale = self.get_scaling.detach()
        before_quat = self.get_quats.detach()
        after_scale = activated_scales.detach()
        if quats is not None:
            after_quat = self.normal_act(quats.detach())
        else:
            after_quat = self.normal_act(world_quats.detach())
        before_min_axis, before_max_axis = self.get_min_max_axis_dir(
            before_scale, before_quat
        )
        after_min_axis, after_max_axis = self.get_min_max_axis_dir(
            after_scale, after_quat
        )
        
        dotprod_min_axis = torch.sum(before_min_axis * after_min_axis, dim=1, keepdim=True)
        dotprod_max_axis = torch.sum(before_max_axis * after_max_axis, dim=1, keepdim=True)
        flip_mask_min_axis = dotprod_min_axis <= 0
        flip_mask_max_axis = dotprod_max_axis <= 0
        after_min_axis = torch.where(flip_mask_min_axis, -after_min_axis, after_min_axis)
        after_max_axis = torch.where(flip_mask_max_axis, -after_max_axis, after_max_axis)
        
        deformation_R = self.get_deformation_rotation_matrix(
            u1=before_min_axis, u2=before_max_axis, v1=after_min_axis, v2=after_max_axis
        )
        world_normals = torch.bmm(deformation_R, self.normal_act(self.get_normals).unsqueeze(-1)).squeeze(-1)
        world_normals = self.transform_normals(world_normals)
        
        valid_mask = self.get_pts_valid_mask()
        
        means = world_means    
        opacities = self._opacities * valid_mask.float().unsqueeze(-1)
        def safe_log(x):
            # clip x to avoid nan
            return torch.log(torch.clamp(x, min=1e-6))
        scales = safe_log(activated_scales)
        if self.gaussian_2d:
            # pad scale 0 for 3DGS consistency
            # but now in log scale, pad -inf
            scales = torch.cat([scales[..., :2], torch.zeros_like(scales[..., :1]) - 1e6], dim=-1)
        quats = self.quat_act(world_quats)
        f_dcs = self._features_dc
        f_rests = self._features_rest
        normals = self.normal_act(world_normals)
        albedos = self._albedos
        roughnesses = self._roughnesses
        metallics = self._metallics
        
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
        )

        # check nan in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
        
        # Since we use this function for exporting ply, we don't need to cache scale
        # self._gs_cache = {
        #     "_scales": activated_scales[filter_mask],
        #     "local_xyz_deformed": means[filter_mask] if delta_xyz is not None else None,
        # }
        return gs_dict

    def compute_reg_loss(self):
        loss_dict = super().compute_reg_loss()
        out_of_bound_losscfg = self.reg_cfg.get("out_of_bound_loss", None)
        if out_of_bound_losscfg is not None:
            w = out_of_bound_losscfg.w
            local_xyz_deformed = self._gs_cache["local_xyz_deformed"]
            if w > 0 and local_xyz_deformed is not None:
                local_xyz_deformed = self._gs_cache["local_xyz_deformed"]
                per_pts_size = self.instances_size[self.point_ids[..., 0]]
                loss_dict["out_of_bound_loss"] = torch.relu(local_xyz_deformed.abs() - per_pts_size / 2).mean() * w
        return loss_dict

    def deform_gaussian_points(
        self, gaussian_dict: Dict[str, torch.Tensor], cur_normalized_time: float,
    ) -> Dict[str, torch.Tensor]:
        """
        deform the points
        """
        means = gaussian_dict["means"]
        nonrigid_embed = self.instances_embedding[gaussian_dict["ids"].squeeze()]
        cur_normalized_time = torch.tensor(cur_normalized_time, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(means.shape[0], 1)
        delta_xyz, delta_quat, delta_scale = self.deform_network(means, cur_normalized_time, nonrigid_embed)
        if self.gaussian_2d:
            delta_scale[..., 2] = 0.0
        gaussian_dict["means"] = means + delta_xyz
        if delta_scale is not None:
            gaussian_dict["scales"] = gaussian_dict["scales"] + delta_scale
        if delta_quat is not None:
            gaussian_dict["quats"] = gaussian_dict["quats"] + delta_quat
        # TBD: deform the normal
        return gaussian_dict

    def load_state_dict(self, dict: Dict, **kwargs) -> str:
        instances_num = dict["instances_fv"].shape[1]
        self.instances_embedding = Parameter(
            torch.rand([instances_num, self.networks_cfg.embed_dim], device=self.device)
        )
        # NOTE: keep it, maybe used in the future
        # del self.instances_embedding
        # self.instances_embedding = nn.Embedding(
        #     instances_num, self.networks_cfg.embed_dim
        # ).to(self.device)
        msg = super().load_state_dict(dict, **kwargs)
        return msg
    
    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = super().collect_gaussians_from_ids(ids)
        # collect embeddings
        for id in ids:
            instance_embedding = self.instances_embedding[id]
            gaussian_dict[id]["embedding"] = instance_embedding
        return gaussian_dict

    def replace_instances(self, replace_dict: Dict[int, int]) -> None:
        """
        replace instances from the model
        
        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(replace_dict.values())
        for ins_id, new_id in replace_dict.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians_dict[new_id]
            self._means = Parameter(torch.cat([self._means, new_gaussian["_means"]], dim=0))
            self._scales = Parameter(torch.cat([self._scales, new_gaussian["_scales"]], dim=0))
            self._quats = Parameter(torch.cat([self._quats, new_gaussian["_quats"]], dim=0))
            self._features_dc = Parameter(torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0))
            self._features_rest = Parameter(torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0))
            self._opacities = Parameter(torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0))
            self._normals = Parameter(torch.cat([self._normals, new_gaussian["_normals"]], dim=0))
            self._albedos = Parameter(torch.cat([self._albedos, new_gaussian["_albedos"]], dim=0))
            self._roughnesses = Parameter(torch.cat([self._roughnesses, new_gaussian["_roughnesses"]], dim=0))
            self._metallics = Parameter(torch.cat([self._metallics, new_gaussian["_metallics"]], dim=0))
            # keeps original point ids
            self.point_ids = torch.cat([self.point_ids, torch.full_like(new_gaussian["point_ids"], ins_id)], dim=0)
            # replace embeddings
            # NOTE: modify data in nn.Parameter directly
            self.instances_embedding.data[ins_id] = new_gaussian["embedding"]

    def export_gaussians_to_ply(
        self, alpha_thresh: float, instance_id: List[int] = None, specific_frame: int = 0,
    ) -> Dict[str, torch.Tensor]:
        self.cur_frame = specific_frame
        pts_mask = self.point_ids[..., 0] == instance_id

        if self.ctrl_cfg.use_deformgs_for_nonrigid and self.step > self.ctrl_cfg.use_deformgs_after:
            delta_xyz, _, _ = self.get_deformation(local_means=self._means)
            means = self._means + delta_xyz
        else:
            means = self._means
        means = means[pts_mask]
        direct_color = self.colors[pts_mask]
        
        activated_opacities = self.get_opacity[pts_mask]
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }