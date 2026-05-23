import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData
from torch.nn import Parameter

from models.nodes.rigid import RigidNodes
from models.trainers.base import GSModelType


POINT_ID_KEYS = ("points_ids", "point_ids")
TIME_INSTANCE_KEYS = {
    "instances_fv",
    "instances_trans",
    "instances_quats",
    "smpl_qauts",
}
SUPPORTED_DYNAMIC_CLASSES = ("RigidNodes", "SMPLNodes", "DeformableNodes")
INSTANCE_MAJOR_TEMPLATE_KEYS = {
    "template.init_beta",
    "template.A0_inv",
    "template.J_canonical",
    "template.W",
    "template.j0_t",
    "template.bbox",
    "template.voxel_deformer.lbs_voxel_base",
    "template.voxel_deformer.voxel_w_correction",
    "template.voxel_deformer.offset",
    "template.voxel_deformer.scale",
    "template.voxel_deformer.grid_denorm",
}


def _safe_logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def _quat_from_axis_angle(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    axis = axis / (axis.norm() + 1e-12)
    half = 0.5 * angle_rad
    s = torch.sin(torch.tensor(half, dtype=torch.float32, device=axis.device))
    c = torch.cos(torch.tensor(half, dtype=torch.float32, device=axis.device))
    return torch.cat([c.view(1), axis * s], dim=0)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # quaternion format: [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=0,
    )


def _build_camera_frame_rotation_quat(
    cam_right: torch.Tensor,
    cam_up: torch.Tensor,
    cam_forward: torch.Tensor,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> torch.Tensor:
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=cam_up.device)
    if abs(yaw_deg) > 1e-12:
        q_yaw = _quat_from_axis_angle(cam_up, np.deg2rad(yaw_deg))
        q = _quat_mul(q, q_yaw)
    if abs(pitch_deg) > 1e-12:
        q_pitch = _quat_from_axis_angle(cam_right, np.deg2rad(pitch_deg))
        q = _quat_mul(q, q_pitch)
    if abs(roll_deg) > 1e-12:
        q_roll = _quat_from_axis_angle(cam_forward, np.deg2rad(roll_deg))
        q = _quat_mul(q, q_roll)
    q = q / (q.norm() + 1e-12)
    return q


def _collect_prefixed_fields(names: List[str], prefix: str) -> List[str]:
    selected = [n for n in names if n.startswith(prefix)]
    if len(selected) == 0:
        return []
    return sorted(selected, key=lambda n: int(n.split("_")[-1]))


def load_rigid_ply_as_model_tensors(ply_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ply = PlyData.read(ply_path)
    vtx = ply["vertex"].data
    names = list(vtx.dtype.names)

    def req(name: str) -> np.ndarray:
        if name not in names:
            raise KeyError(f"Required field '{name}' not found in {ply_path}")
        return np.asarray(vtx[name])

    xyz = np.stack([req("x"), req("y"), req("z")], axis=1).astype(np.float32)
    scales = np.stack([req("scale_0"), req("scale_1"), req("scale_2")], axis=1).astype(np.float32)
    quats = np.stack([req("rot_0"), req("rot_1"), req("rot_2"), req("rot_3")], axis=1).astype(np.float32)
    opacities = req("opacity").astype(np.float32).reshape(-1, 1)
    normals = np.stack([req("nx"), req("ny"), req("nz")], axis=1).astype(np.float32)
    albedos = np.stack([req("albedo_0"), req("albedo_1"), req("albedo_2")], axis=1).astype(np.float32)
    roughnesses = req("roughness").astype(np.float32).reshape(-1, 1)
    metallics = req("metallic").astype(np.float32).reshape(-1, 1)

    f_dc_fields = _collect_prefixed_fields(names, "f_dc_")
    if len(f_dc_fields) == 0:
        raise KeyError(f"No f_dc_* fields found in {ply_path}")
    f_dc = np.stack([np.asarray(vtx[n]) for n in f_dc_fields], axis=1).astype(np.float32)

    f_rest_fields = _collect_prefixed_fields(names, "f_rest_")
    if len(f_rest_fields) == 0:
        raise KeyError(f"No f_rest_* fields found in {ply_path}")
    f_rest_flat = np.stack([np.asarray(vtx[n]) for n in f_rest_fields], axis=1).astype(np.float32)
    if f_rest_flat.shape[1] % 3 != 0:
        raise ValueError(f"f_rest field count must be divisible by 3, got {f_rest_flat.shape[1]}")
    f_rest = f_rest_flat.reshape(f_rest_flat.shape[0], 3, -1).transpose(0, 2, 1)

    out = {
        "_means": torch.from_numpy(xyz).to(device),
        "_scales": torch.log(torch.from_numpy(np.clip(scales, 1e-8, None)).to(device)),
        "_quats": torch.from_numpy(quats).to(device),
        "_opacities": _safe_logit(torch.from_numpy(opacities).to(device)),
        "_normals": torch.from_numpy(normals).to(device),
        "_albedos": _safe_logit(torch.from_numpy(np.clip(albedos, 1e-6, 1 - 1e-6)).to(device)),
        "_roughnesses": _safe_logit(torch.from_numpy(np.clip(roughnesses, 1e-6, 1 - 1e-6)).to(device)),
        "_metallics": _safe_logit(torch.from_numpy(np.clip(metallics, 1e-6, 1 - 1e-6)).to(device)),
        "_features_dc": torch.from_numpy(f_dc).to(device),
        "_features_rest": torch.from_numpy(f_rest).to(device),
    }
    # drop non-finite points to avoid propagating NaN/Inf into model params
    finite_mask = torch.isfinite(out["_means"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_scales"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_quats"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_opacities"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_normals"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_albedos"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_roughnesses"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_metallics"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_features_dc"]).all(dim=1)
    finite_mask &= torch.isfinite(out["_features_rest"]).reshape(out["_features_rest"].shape[0], -1).all(dim=1)
    for k, v in out.items():
        out[k] = v[finite_mask]
    return out


def insert_rigid_object_for_eval(
    trainer,
    dataset,
    ply_path: str,
    anchor_timestep: int = 0,
    anchor_cam_id: int = 0,
    forward_m: float = 8.0,
    right_m: float = 0.0,
    up_m: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_up_deg: float = 0.0,
    roll_left_deg: float = 0.0,
    scale_mode: str = "ego",
    manual_scale: float = 1.0,
    visible_first_frame_only: bool = True,
) -> Dict[str, int]:
    if "RigidNodes" not in trainer.models:
        ctrl_cfg = OmegaConf.create(
            {
                "sh_degree": 3,
                "sh_degree_interval": 1000,
                "gaussian_2d": False,
                "ball_gaussians": False,
                "cull_out_of_bound": False,
                "refine_interval": 1000000000,
                "warmup_steps": 0,
                "reset_alpha_interval": 1000000000,
                "stop_split_at": 0,
                "stop_screen_size_at": 0,
                "densify_grad_thresh": 1e9,
                "densify_size_thresh": 1e9,
                "n_split_samples": 2,
                "cull_alpha_thresh": 0.0,
                "cull_scale_thresh": 1e9,
                "cull_screen_size": 1e9,
                "split_screen_size": 1e9,
                "reset_alpha_value": 0.01,
            }
        )
        model = RigidNodes(
            class_name="RigidNodes",
            ctrl=ctrl_cfg,
            reg=None,
            networks=None,
            scene_scale=trainer.scene_radius,
            scene_origin=trainer.scene_origin,
            num_train_images=trainer.num_train_images,
            device=trainer.device,
        ).to(trainer.device)
        T = dataset.num_img_timesteps
        device = trainer.device
        model._means = Parameter(torch.empty((0, 3), device=device))
        model._scales = Parameter(torch.empty((0, 3), device=device))
        model._quats = Parameter(torch.empty((0, 4), device=device))
        model._features_dc = Parameter(torch.empty((0, 3), device=device))
        model._features_rest = Parameter(torch.empty((0, 15, 3), device=device))
        model._opacities = Parameter(torch.empty((0, 1), device=device))
        model._normals = Parameter(torch.empty((0, 3), device=device))
        model._albedos = Parameter(torch.empty((0, 3), device=device))
        model._roughnesses = Parameter(torch.empty((0, 1), device=device))
        model._metallics = Parameter(torch.empty((0, 1), device=device))
        model.point_ids = torch.empty((0, 1), dtype=torch.long, device=device)
        model.instances_size = torch.empty((0, 3), device=device)
        model.instances_fv = torch.zeros((T, 0), dtype=torch.bool, device=device)
        model.instances_trans = Parameter(torch.zeros((T, 0, 3), device=device))
        model.instances_quats = Parameter(torch.zeros((T, 0, 4), device=device))

        trainer.models["RigidNodes"] = model
        trainer.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
    else:
        model = trainer.models["RigidNodes"]
    device = trainer.device

    src = load_rigid_ply_as_model_tensors(ply_path, device=device)
    n_pts = src["_means"].shape[0]
    if n_pts == 0:
        raise ValueError("Loaded PLY has zero points.")

    target_rest_dim = model._features_rest.shape[1]
    src_rest_dim = src["_features_rest"].shape[1]
    if src_rest_dim < target_rest_dim:
        pad = torch.zeros(n_pts, target_rest_dim - src_rest_dim, 3, device=device, dtype=src["_features_rest"].dtype)
        src["_features_rest"] = torch.cat([src["_features_rest"], pad], dim=1)
    elif src_rest_dim > target_rest_dim:
        src["_features_rest"] = src["_features_rest"][:, :target_rest_dim, :]

    # recenter canonical means so instance translation controls final placement directly
    src["_means"] = src["_means"] - src["_means"].mean(dim=0, keepdim=True)

    # object scale estimation
    src_extent = (src["_means"].max(dim=0).values - src["_means"].min(dim=0).values).norm().item()
    if scale_mode == "ego":
        ref_sizes = model.instances_size
        ref_scale = None
        if ref_sizes.numel() > 0:
            cand = ref_sizes.norm(dim=1)
            cand = cand[torch.isfinite(cand) & (cand > 1e-6)]
            if cand.numel() > 0:
                ref_scale = cand.median().item()
        if ref_scale is None or not np.isfinite(ref_scale):
            ref_scale = max(src_extent, 1.0)
        scale = ref_scale / max(src_extent, 1e-6)
    elif scale_mode == "manual":
        scale = manual_scale
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")
    scale = float(scale)

    src["_means"] = src["_means"] * scale
    src["_scales"] = src["_scales"] + np.log(max(scale, 1e-8))
    if not torch.isfinite(src["_means"]).all():
        raise ValueError("Inserted means became non-finite after scaling. Check PLY values and scale arguments.")

    num_cams = dataset.pixel_source.num_cams
    anchor_timestep = int(np.clip(anchor_timestep, 0, dataset.num_img_timesteps - 1))
    anchor_cam_id = int(np.clip(anchor_cam_id, 0, num_cams - 1))
    anchor_idx = anchor_timestep * num_cams + anchor_cam_id
    _, cam_infos = dataset.full_image_set.get_image(anchor_idx, 1.0)
    c2w = cam_infos["camera_to_world"].to(device)

    cam_pos = c2w[:3, 3]
    cam_right = c2w[:3, 0] / (c2w[:3, 0].norm() + 1e-12)
    cam_up = c2w[:3, 1] / (c2w[:3, 1].norm() + 1e-12)
    cam_forward = c2w[:3, 2] / (c2w[:3, 2].norm() + 1e-12)
    target_pos = cam_pos + cam_forward * forward_m + cam_right * right_m + cam_up * up_m

    print(f"Object position: {target_pos}")
    print(f"Camera position: {cam_pos}, forward: {cam_forward}, right: {cam_right}, up: {cam_up}")

    rotation_quat = _build_camera_frame_rotation_quat(
        cam_right=cam_right,
        cam_up=cam_up,
        cam_forward=cam_forward,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_up_deg,
        roll_deg=roll_left_deg,
    ).to(device)

    if not hasattr(model, "point_ids"):
        model.point_ids = torch.empty((0, 1), dtype=torch.long, device=device)
    if not hasattr(model, "instances_size"):
        model.instances_size = torch.empty((0, 3), device=device)
    if not hasattr(model, "instances_fv"):
        model.instances_fv = torch.zeros((dataset.num_img_timesteps, 0), dtype=torch.bool, device=device)
    if not hasattr(model, "instances_trans"):
        model.instances_trans = Parameter(torch.zeros((dataset.num_img_timesteps, 0, 3), device=device))
    if not hasattr(model, "instances_quats"):
        model.instances_quats = Parameter(torch.zeros((dataset.num_img_timesteps, 0, 4), device=device))

    old_num_instances = model.instances_fv.shape[1]
    new_id = old_num_instances
    T = model.instances_fv.shape[0]

    # append per-point tensors
    model._means = Parameter(torch.cat([model._means.data, src["_means"]], dim=0))
    model._scales = Parameter(torch.cat([model._scales.data, src["_scales"]], dim=0))
    model._quats = Parameter(torch.cat([model._quats.data, src["_quats"]], dim=0))
    model._features_dc = Parameter(torch.cat([model._features_dc.data, src["_features_dc"]], dim=0))
    model._features_rest = Parameter(torch.cat([model._features_rest.data, src["_features_rest"]], dim=0))
    model._opacities = Parameter(torch.cat([model._opacities.data, src["_opacities"]], dim=0))
    model._normals = Parameter(torch.cat([model._normals.data, src["_normals"]], dim=0))
    model._albedos = Parameter(torch.cat([model._albedos.data, src["_albedos"]], dim=0))
    model._roughnesses = Parameter(torch.cat([model._roughnesses.data, src["_roughnesses"]], dim=0))
    model._metallics = Parameter(torch.cat([model._metallics.data, src["_metallics"]], dim=0))
    model.point_ids = torch.cat(
        [model.point_ids, torch.full((n_pts, 1), new_id, dtype=model.point_ids.dtype, device=device)],
        dim=0,
    )

    # append per-instance tensors
    src_size = (src["_means"].max(dim=0).values - src["_means"].min(dim=0).values).detach()
    model.instances_size = torch.cat([model.instances_size, src_size.unsqueeze(0)], dim=0)

    fv_new_col = torch.zeros((T, 1), dtype=model.instances_fv.dtype, device=device)
    if visible_first_frame_only:
        fv_new_col[0, 0] = True
    else:
        fv_new_col[:, 0] = True
    model.instances_fv = torch.cat([model.instances_fv, fv_new_col], dim=1)

    trans_new = torch.zeros((T, 1, 3), dtype=model.instances_trans.dtype, device=device)
    trans_new[:, 0, :] = target_pos
    model.instances_trans = Parameter(torch.cat([model.instances_trans.data, trans_new], dim=1))

    quat_new = torch.zeros((T, 1, 4), dtype=model.instances_quats.dtype, device=device)
    quat_new[:, 0, :] = rotation_quat
    model.instances_quats = Parameter(torch.cat([model.instances_quats.data, quat_new], dim=1))

    return {
        "inserted_instance_id": int(new_id),
        "inserted_points": int(n_pts),
    }


def set_inserted_rigid_rotation_for_eval(
    trainer,
    dataset,
    instance_id: int,
    yaw_deg: float,
    pitch_up_deg: float = 0.0,
    roll_left_deg: float = 0.0,
    anchor_timestep: int = 0,
    anchor_cam_id: int = 0,
) -> None:
    if "RigidNodes" not in trainer.models:
        raise ValueError("RigidNodes not found in current trainer model.")
    model = trainer.models["RigidNodes"]
    device = trainer.device

    num_cams = dataset.pixel_source.num_cams
    anchor_timestep = int(np.clip(anchor_timestep, 0, dataset.num_img_timesteps - 1))
    anchor_cam_id = int(np.clip(anchor_cam_id, 0, num_cams - 1))
    anchor_idx = anchor_timestep * num_cams + anchor_cam_id
    _, cam_infos = dataset.full_image_set.get_image(anchor_idx, 1.0)
    c2w = cam_infos["camera_to_world"].to(device)
    cam_right = c2w[:3, 0] / (c2w[:3, 0].norm() + 1e-12)
    cam_up = c2w[:3, 1] / (c2w[:3, 1].norm() + 1e-12)
    cam_forward = c2w[:3, 2] / (c2w[:3, 2].norm() + 1e-12)
    rot_quat = _build_camera_frame_rotation_quat(
        cam_right=cam_right,
        cam_up=cam_up,
        cam_forward=cam_forward,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_up_deg,
        roll_deg=roll_left_deg,
    ).to(device)

    model.instances_quats.data[:, instance_id, :] = rot_quat.view(1, 4).repeat(model.instances_quats.shape[0], 1)


def _parse_dynamic_classes(raw: str) -> List[str]:
    classes = [x.strip() for x in raw.split(",") if x.strip()]
    if not classes:
        raise ValueError("No classes provided for dynamic insertion.")
    bad = [c for c in classes if c not in SUPPORTED_DYNAMIC_CLASSES]
    if bad:
        raise ValueError(
            f"Unsupported classes: {bad}. Supported classes: {SUPPORTED_DYNAMIC_CLASSES}"
        )
    return classes


def _get_point_id_key(state: Dict[str, torch.Tensor]) -> str:
    for key in POINT_ID_KEYS:
        if key in state:
            return key
    raise KeyError(f"State dict missing point id key: expected one of {POINT_ID_KEYS}")


def _get_num_instances(state: Dict[str, torch.Tensor]) -> int:
    if "instances_fv" in state:
        return int(state["instances_fv"].shape[1])
    if "instances_size" in state:
        return int(state["instances_size"].shape[0])
    pid_key = _get_point_id_key(state)
    pids = state[pid_key].reshape(-1).long()
    return int(torch.max(pids).item()) + 1 if pids.numel() > 0 else 0


def _get_num_timesteps(state: Dict[str, torch.Tensor]) -> Optional[int]:
    for key in ("instances_trans", "instances_fv", "instances_quats", "smpl_qauts"):
        val = state.get(key, None)
        if torch.is_tensor(val) and val.ndim >= 1:
            return int(val.shape[0])
    return None


def _clone_state_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


def _normalize_instance_major_state(
    state: Dict[str, torch.Tensor],
    instance_ids: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    out = _clone_state_to_cpu(state)
    expected_instances = _get_num_instances(out)
    if expected_instances < 0:
        return out

    selected_ids: Optional[List[int]] = None
    if instance_ids is not None:
        selected_ids = [int(x) for x in instance_ids]
        if len(selected_ids) != expected_instances:
            selected_ids = None

    for key, val in list(out.items()):
        if not torch.is_tensor(val) or val.ndim < 1:
            continue

        if key in TIME_INSTANCE_KEYS and val.ndim >= 2:
            cur_instances = int(val.shape[1])
            if cur_instances == expected_instances:
                continue
            if selected_ids is not None and cur_instances > max(selected_ids, default=-1):
                out[key] = val[:, selected_ids].clone()
            elif cur_instances >= expected_instances:
                out[key] = val[:, :expected_instances].clone()
            continue

        if key in INSTANCE_MAJOR_TEMPLATE_KEYS:
            cur_instances = int(val.shape[0])
            if cur_instances == expected_instances:
                continue
            if selected_ids is not None and cur_instances > max(selected_ids, default=-1):
                out[key] = val[selected_ids].clone()
            elif cur_instances >= expected_instances:
                out[key] = val[:expected_instances].clone()
            continue

    return out


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)


def _quat_mul_batched(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return _quat_normalize(out)


def _quat_rotate_batched(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = _quat_normalize(q)
    q_xyz = q[..., 1:4]
    qw = q[..., 0:1]
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (qw * uv + uuv)


def _build_camera_frame_rotation_quat_batched(
    cam_right: torch.Tensor,
    cam_up: torch.Tensor,
    cam_forward: torch.Tensor,
    yaw_deg: float = 0.0,
    pitch_up_deg: float = 0.0,
    roll_left_deg: float = 0.0,
) -> torch.Tensor:
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=cam_up.device)
    if abs(yaw_deg) > 1e-12:
        q = _quat_mul_batched(q, _quat_from_axis_angle(cam_up, np.deg2rad(yaw_deg)))
    if abs(pitch_up_deg) > 1e-12:
        q = _quat_mul_batched(q, _quat_from_axis_angle(cam_right, np.deg2rad(pitch_up_deg)))
    if abs(roll_left_deg) > 1e-12:
        q = _quat_mul_batched(q, _quat_from_axis_angle(cam_forward, np.deg2rad(roll_left_deg)))
    return _quat_normalize(q)


def _build_world_yaw_rotation_quat(
    yaw_deg: float,
    device: torch.device,
) -> torch.Tensor:
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    return _quat_from_axis_angle(world_up, np.deg2rad(float(yaw_deg)))


def _get_anchor_camera_frame(dataset, anchor_timestep: int, anchor_cam_id: int):
    num_cams = dataset.pixel_source.num_cams
    t_idx = int(np.clip(anchor_timestep, 0, dataset.num_img_timesteps - 1))
    c_idx = int(np.clip(anchor_cam_id, 0, num_cams - 1))
    idx = t_idx * num_cams + c_idx
    _, cam_infos = dataset.full_image_set.get_image(idx, 1.0)
    c2w = cam_infos["camera_to_world"].cpu().float()

    cam_pos = c2w[:3, 3]
    cam_right = c2w[:3, 0] / (c2w[:3, 0].norm() + 1e-12)
    cam_up = c2w[:3, 1] / (c2w[:3, 1].norm() + 1e-12)
    cam_forward = c2w[:3, 2] / (c2w[:3, 2].norm() + 1e-12)
    return cam_pos, cam_right, cam_up, cam_forward


def _get_root_quats_at_frame(state: Dict[str, torch.Tensor], frame_idx: int) -> Optional[torch.Tensor]:
    if "instances_quats" not in state:
        return None
    quats = state["instances_quats"][frame_idx]
    if quats.ndim == 3 and quats.shape[1] == 1:
        quats = quats[:, 0, :]
    if quats.ndim != 2 or quats.shape[-1] != 4:
        raise ValueError(f"Unsupported instances_quats shape at frame: {tuple(quats.shape)}")
    return _quat_normalize(quats)


def _compute_instance_centers_at_frame(
    state: Dict[str, torch.Tensor],
    frame_idx: int,
    use_visibility_mask: bool,
) -> torch.Tensor:
    pid_key = _get_point_id_key(state)
    pids = state[pid_key].reshape(-1).long()
    means = state["_means"].float()
    num_instances = _get_num_instances(state)

    trans = state.get("instances_trans", None)
    trans_frame = trans[frame_idx].float() if trans is not None else None

    quats_frame = _get_root_quats_at_frame(state, frame_idx)
    vis = state.get("instances_fv", None)
    vis_frame = vis[frame_idx].bool() if vis is not None else None

    centers: List[torch.Tensor] = []
    for ins_id in range(num_instances):
        if use_visibility_mask and vis_frame is not None and not bool(vis_frame[ins_id]):
            continue

        pts_mask = pids == ins_id
        if torch.any(pts_mask):
            local_center = means[pts_mask].mean(dim=0)
        else:
            local_center = torch.zeros(3, dtype=torch.float32, device=means.device)

        world_center = local_center
        if quats_frame is not None:
            world_center = _quat_rotate_batched(quats_frame[ins_id], world_center)
        if trans_frame is not None:
            world_center = world_center + trans_frame[ins_id]
        centers.append(world_center)

    if not centers:
        return torch.empty((0, 3), dtype=torch.float32, device=means.device)
    return torch.stack(centers, dim=0)


def _choose_target_position(
    cam_pos: torch.Tensor,
    cam_right: torch.Tensor,
    cam_up: torch.Tensor,
    cam_forward: torch.Tensor,
    base_forward_m: float,
    base_right_m: float,
    up_m: float,
    existing_centers: torch.Tensor,
    enable_search: bool,
    search_radius_m: float,
    search_step_m: float,
    placement_mode: str = "visible",
) -> torch.Tensor:
    base_pos = cam_pos + cam_forward * base_forward_m + cam_right * base_right_m + cam_up * up_m

    if (not enable_search) or existing_centers.numel() == 0:
        return base_pos

    right_offsets = np.arange(-search_radius_m, search_radius_m + 1e-6, search_step_m, dtype=np.float32)
    forward_offsets = np.array([-2.0, 0.0, 2.0, 4.0, 6.0], dtype=np.float32)

    best_score = -1e9
    best_pos = base_pos

    def _score_candidate(cand: torch.Tensor) -> float:
        dists = torch.norm(existing_centers - cand.view(1, 3), dim=1)
        min_dist = float(torch.min(dists).item()) if dists.numel() > 0 else 1e6
        pullback = float(torch.norm(cand - base_pos).item())

        if placement_mode == "clearance":
            return min_dist - 0.05 * pullback

        rel = cand - cam_pos
        forward_dist = float(torch.dot(rel, cam_forward).item())
        right_dist = float(torch.dot(rel, cam_right).item())
        up_dist = float(torch.dot(rel, cam_up).item())
        if forward_dist <= 1e-3:
            return -1e9

        lateral_ratio = abs(right_dist) / max(forward_dist, 1e-6)
        vertical_ratio = abs(up_dist) / max(forward_dist, 1e-6)
        centeredness = lateral_ratio + 0.5 * vertical_ratio
        forward_bias = abs(forward_dist - base_forward_m)

        return (
            1.5 * min_dist
            - 3.5 * centeredness
            - 0.08 * pullback
            - 0.12 * forward_bias
        )

    for d_f in forward_offsets:
        f_m = max(1.0, base_forward_m + float(d_f))
        for d_r in right_offsets:
            r_m = base_right_m + float(d_r)
            cand = cam_pos + cam_forward * f_m + cam_right * r_m + cam_up * up_m
            score = _score_candidate(cand)
            if score > best_score:
                best_score = score
                best_pos = cand

    return best_pos


def _parse_optional_world_xyz(raw: Optional[Sequence[float]]) -> Optional[torch.Tensor]:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None
        values = [x.strip() for x in raw.split(",")]
    else:
        values = list(raw)
    if len(values) != 3:
        raise ValueError(f"target_world_xyz must contain exactly 3 values, got: {raw!r}")
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


def _align_source_time_dims(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, val in source_state.items():
        if (
            key in TIME_INSTANCE_KEYS
            and torch.is_tensor(val)
            and key in target_state
            and torch.is_tensor(target_state[key])
            and val.ndim >= 1
            and target_state[key].ndim >= 1
        ):
            src_t = int(val.shape[0])
            tgt_t = int(target_state[key].shape[0])
            if src_t > tgt_t:
                out[key] = val[:tgt_t]
            elif src_t < tgt_t:
                pad_count = tgt_t - src_t
                pad = val[-1:].repeat((pad_count,) + (1,) * (val.ndim - 1))
                out[key] = torch.cat([val, pad], dim=0)
            else:
                out[key] = val
        else:
            out[key] = val
    return out


def _maybe_scale_source_states_for_eval(
    source_states: Dict[str, Dict[str, torch.Tensor]],
    target_states: Dict[str, Dict[str, torch.Tensor]],
    scale_mode: str,
    manual_scale: float,
) -> float:
    def _scale_state_spatial_fields(state: Dict[str, torch.Tensor], scale: float) -> None:
        if "_means" in state:
            state["_means"] = state["_means"] * scale
        if "_scales" in state:
            state["_scales"] = state["_scales"] + np.log(max(scale, 1e-8))
        if "instances_trans" in state and torch.is_tensor(state["instances_trans"]):
            state["instances_trans"] = state["instances_trans"] * scale
        if "instances_size" in state and torch.is_tensor(state["instances_size"]):
            state["instances_size"] = state["instances_size"] * scale

        # SMPL / voxel-deformer spatial frame also needs scaling; otherwise the
        # canonical Gaussian positions and the deformation frame go out of sync.
        for key in (
            "template.J_canonical",
            "template.j0_t",
            "template.bbox",
            "template.voxel_deformer.offset",
            "template.voxel_deformer.scale",
            "template.voxel_deformer.grid_denorm",
        ):
            if key in state and torch.is_tensor(state[key]):
                state[key] = state[key] * scale

        if "template.A0_inv" in state and torch.is_tensor(state["template.A0_inv"]):
            a0_inv = state["template.A0_inv"].clone()
            a0_inv[..., :3, 3] = a0_inv[..., :3, 3] * scale
            state["template.A0_inv"] = a0_inv

    if scale_mode == "manual":
        scale = float(manual_scale)
    elif scale_mode == "ego":
        ref_sizes: List[torch.Tensor] = []
        for class_name, state in target_states.items():
            if class_name not in source_states:
                continue
            if "instances_size" in state and torch.is_tensor(state["instances_size"]) and state["instances_size"].numel() > 0:
                norms = torch.norm(state["instances_size"].float(), dim=-1)
                norms = norms[torch.isfinite(norms) & (norms > 1e-6)]
                if norms.numel() > 0:
                    ref_sizes.append(norms.cpu())
        ref_scale = float(torch.cat(ref_sizes, dim=0).median().item()) if ref_sizes else 1.0

        all_means: List[torch.Tensor] = []
        for state in source_states.values():
            if "_means" in state and state["_means"].numel() > 0:
                all_means.append(state["_means"].float())
        if all_means:
            means = torch.cat(all_means, dim=0)
            src_extent = torch.norm(means.max(dim=0).values - means.min(dim=0).values).item()
        else:
            src_extent = 1.0
        scale = ref_scale / max(src_extent, 1e-6)
    else:
        raise ValueError(f"Unsupported scale mode: {scale_mode}")

    for state in source_states.values():
        _scale_state_spatial_fields(state, scale)
    return float(scale)


def _compute_group_center(class_states: Dict[str, Dict[str, torch.Tensor]], frame_idx: int) -> torch.Tensor:
    centers_all: List[torch.Tensor] = []
    for state in class_states.values():
        if "instances_trans" not in state:
            continue
        centers = _compute_instance_centers_at_frame(state, frame_idx=frame_idx, use_visibility_mask=False)
        if centers.numel() > 0:
            centers_all.append(centers)
    if not centers_all:
        return torch.zeros(3, dtype=torch.float32)
    return torch.cat(centers_all, dim=0).mean(dim=0)


def _apply_group_transform_to_state(
    state: Dict[str, torch.Tensor],
    q_global: torch.Tensor,
    source_center: torch.Tensor,
    target_center: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[k] = v.clone() if torch.is_tensor(v) else v

    device = None
    if "instances_trans" in out and torch.is_tensor(out["instances_trans"]):
        device = out["instances_trans"].device
    elif "instances_quats" in out and torch.is_tensor(out["instances_quats"]):
        device = out["instances_quats"].device
    if device is None:
        return out

    q_global_d = q_global.to(device)
    source_center_d = source_center.to(device)
    target_center_d = target_center.to(device)

    if "instances_trans" in out:
        trans = out["instances_trans"].float()
        trans = _quat_rotate_batched(
            q_global_d.view(1, 1, 4),
            trans - source_center_d.view(1, 1, 3),
        ) + target_center_d.view(1, 1, 3)
        out["instances_trans"] = trans

    if "instances_quats" in out:
        q_old = out["instances_quats"].float()
        if q_old.ndim == 3:
            q_new = _quat_mul_batched(q_global_d.view(1, 1, 4), q_old)
        elif q_old.ndim == 4:
            q_new = _quat_mul_batched(q_global_d.view(1, 1, 1, 4), q_old)
        else:
            raise ValueError(f"Unsupported instances_quats shape: {tuple(q_old.shape)}")
        out["instances_quats"] = _quat_normalize(q_new)

    return out


def _append_class_state_for_eval(
    target_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    t_pid_key = _get_point_id_key(target_state)
    s_pid_key = _get_point_id_key(src_state)

    t_num_points = int(target_state[t_pid_key].shape[0])
    s_num_points = int(src_state[s_pid_key].shape[0])
    t_num_instances = _get_num_instances(target_state)
    s_num_instances = _get_num_instances(src_state)

    all_keys = sorted(set(target_state.keys()).union(set(src_state.keys())))
    for key in all_keys:
        t = target_state.get(key, None)
        s = src_state.get(key, None)

        if t is None:
            continue
        if s is None:
            out[key] = t
            continue
        if (not torch.is_tensor(t)) or (not torch.is_tensor(s)):
            out[key] = t
            continue

        s = s.to(device=t.device, dtype=t.dtype)

        if key == t_pid_key:
            shifted = (src_state[s_pid_key].long() + t_num_instances).to(device=t.device, dtype=t.dtype)
            out[key] = torch.cat([t, shifted], dim=0)
            continue

        if key == s_pid_key and s_pid_key != t_pid_key:
            continue

        if t.ndim >= 1 and s.ndim >= 1 and t.shape[0] == t_num_points and s.shape[0] == s_num_points:
            if t.shape[1:] != s.shape[1:]:
                raise ValueError(
                    "Incompatible per-point tensor shape for key "
                    f"'{key}': target={tuple(t.shape[1:])}, source={tuple(s.shape[1:])}. "
                    "Please export/insert with matching model config."
                )
            else:
                out[key] = torch.cat([t, s], dim=0)
            continue

        if (
            t.ndim >= 2
            and s.ndim >= 2
            and t.shape[0] == s.shape[0]
            and t.shape[1] == t_num_instances
            and s.shape[1] == s_num_instances
            and t.shape[2:] == s.shape[2:]
        ):
            out[key] = torch.cat([t, s], dim=1)
            continue

        if (
            t.ndim >= 1
            and s.ndim >= 1
            and t.shape[0] == t_num_instances
            and s.shape[0] == s_num_instances
            and t.shape[1:] == s.shape[1:]
        ):
            out[key] = torch.cat([t, s], dim=0)
            continue

        if key in INSTANCE_MAJOR_TEMPLATE_KEYS and t.ndim >= 1 and s.ndim >= 1 and t.shape[0] == t_num_instances and s.shape[0] == s_num_instances:
            if t.shape[1:] == s.shape[1:]:
                out[key] = torch.cat([t, s], dim=0)
            else:
                out[key] = t
            continue

        out[key] = t

    return out


def _format_state_shape_mismatches(
    model,
    state_dict: Dict[str, torch.Tensor],
) -> List[str]:
    current_state = model.state_dict()
    lines: List[str] = []
    for key, val in state_dict.items():
        cur = current_state.get(key, None)
        if not torch.is_tensor(val) or not torch.is_tensor(cur):
            continue
        if tuple(val.shape) != tuple(cur.shape):
            lines.append(
                f"{key}: merged={tuple(val.shape)} current={tuple(cur.shape)}"
            )
    return lines


def insert_dynamic_asset_for_eval(
    trainer,
    dataset,
    asset_path: str,
    classes: str = "RigidNodes,SMPLNodes",
    anchor_timestep: int = 0,
    anchor_cam_id: int = 0,
    source_anchor_timestep: Optional[int] = None,
    forward_m: float = 8.0,
    right_m: float = 0.0,
    up_m: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_up_deg: float = 0.0,
    roll_left_deg: float = 0.0,
    scale_mode: str = "ego",
    manual_scale: float = 1.0,
    disable_place_search: bool = False,
    search_radius_m: float = 8.0,
    search_step_m: float = 2.0,
    placement_mode: str = "visible",
    target_world_xyz: Optional[Sequence[float]] = None,
    target_world_timestep: Optional[int] = None,
    target_world_yaw_deg: Optional[float] = None,
) -> Dict[str, object]:
    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Dynamic asset path not found: {asset_path}")

    class_list = _parse_dynamic_classes(classes)
    package = torch.load(asset_path, map_location="cpu")
    if "classes" not in package:
        raise KeyError("Dynamic asset package missing 'classes' field.")
    package_classes = package["classes"]

    missing_in_asset = [c for c in class_list if c not in package_classes]
    if missing_in_asset:
        raise KeyError(f"Requested classes missing in asset: {missing_in_asset}")

    missing_in_model = [c for c in class_list if c not in trainer.models]
    if missing_in_model:
        raise KeyError(
            "Requested classes missing in current checkpoint model: "
            f"{missing_in_model}."
        )

    source_states: Dict[str, Dict[str, torch.Tensor]] = {}
    target_states: Dict[str, Dict[str, torch.Tensor]] = {}
    for class_name in class_list:
        source_states[class_name] = _normalize_instance_major_state(
            package_classes[class_name]["state_dict"],
            instance_ids=package_classes[class_name].get("instance_ids"),
        )
        target_states[class_name] = _normalize_instance_major_state(
            {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in trainer.models[class_name].state_dict().items()
            }
        )

    applied_scale = _maybe_scale_source_states_for_eval(
        source_states=source_states,
        target_states=target_states,
        scale_mode=scale_mode,
        manual_scale=manual_scale,
    )

    direct_target_pos = _parse_optional_world_xyz(target_world_xyz)
    if direct_target_pos is not None and target_world_timestep is not None:
        target_anchor_timestep = int(np.clip(target_world_timestep, 0, dataset.num_img_timesteps - 1))
    else:
        target_anchor_timestep = int(np.clip(anchor_timestep, 0, dataset.num_img_timesteps - 1))
    requested_source_anchor_timestep = (
        target_anchor_timestep if source_anchor_timestep is None else source_anchor_timestep
    )
    source_time_lengths = [
        t for t in ( _get_num_timesteps(state) for state in source_states.values()) if t is not None
    ]
    if source_time_lengths:
        source_max_t = max(0, min(source_time_lengths) - 1)
        source_anchor_timestep = int(np.clip(requested_source_anchor_timestep, 0, source_max_t))
    else:
        source_anchor_timestep = 0

    source_group_center = _compute_group_center(source_states, frame_idx=source_anchor_timestep)

    cam_pos, cam_right, cam_up, cam_forward = _get_anchor_camera_frame(
        dataset=dataset,
        anchor_timestep=target_anchor_timestep,
        anchor_cam_id=anchor_cam_id,
    )
    base_target_pos = cam_pos + cam_forward * forward_m + cam_right * right_m + cam_up * up_m

    if direct_target_pos is not None:
        target_anchor_pos = direct_target_pos.to(device=cam_pos.device, dtype=cam_pos.dtype)
        search_enabled = False
        placement_label = "target_world_xyz"
    else:
        existing_centers_list: List[torch.Tensor] = []
        for class_name in SUPPORTED_DYNAMIC_CLASSES:
            if class_name not in trainer.models:
                continue
            state = trainer.models[class_name].state_dict()
            if "instances_trans" not in state:
                continue
            centers = _compute_instance_centers_at_frame(
                state,
                frame_idx=target_anchor_timestep,
                use_visibility_mask=True,
            )
            if centers.numel() > 0:
                existing_centers_list.append(centers.cpu())
        if existing_centers_list:
            existing_centers = torch.cat(existing_centers_list, dim=0)
        else:
            existing_centers = torch.empty((0, 3), dtype=torch.float32)

        search_enabled = not disable_place_search
        placement_label = placement_mode
        target_anchor_pos = _choose_target_position(
            cam_pos=cam_pos,
            cam_right=cam_right,
            cam_up=cam_up,
            cam_forward=cam_forward,
            base_forward_m=forward_m,
            base_right_m=right_m,
            up_m=up_m,
            existing_centers=existing_centers,
            enable_search=search_enabled,
            search_radius_m=search_radius_m,
            search_step_m=search_step_m,
            placement_mode=placement_mode,
        )
    print(
        "[insert_dynamic] "
        f"asset={os.path.abspath(asset_path)} "
        f"target_anchor_timestep={target_anchor_timestep} "
        f"source_anchor_timestep={source_anchor_timestep} "
        f"offsets=(forward={float(forward_m):.4f}, right={float(right_m):.4f}, up={float(up_m):.4f}) "
        f"placement={placement_label} "
        f"search_enabled={search_enabled} "
        f"base_xyz=({base_target_pos[0].item():.4f}, "
        f"{base_target_pos[1].item():.4f}, {base_target_pos[2].item():.4f}) "
        f"chosen_xyz=({target_anchor_pos[0].item():.4f}, "
        f"{target_anchor_pos[1].item():.4f}, {target_anchor_pos[2].item():.4f}) "
        f"camera_xyz=({cam_pos[0].item():.4f}, {cam_pos[1].item():.4f}, {cam_pos[2].item():.4f})"
    )

    if target_world_yaw_deg is not None:
        q_global = _build_world_yaw_rotation_quat(
            yaw_deg=float(target_world_yaw_deg),
            device=cam_pos.device,
        )
        rotation_label = f"target_world_yaw_deg={float(target_world_yaw_deg):.4f}"
    else:
        q_global = _build_camera_frame_rotation_quat_batched(
            cam_right=cam_right,
            cam_up=cam_up,
            cam_forward=cam_forward,
            yaw_deg=yaw_deg,
            pitch_up_deg=pitch_up_deg,
            roll_left_deg=roll_left_deg,
        )
        rotation_label = (
            f"camera_frame yaw={float(yaw_deg):.4f} "
            f"pitch={float(pitch_up_deg):.4f} roll={float(roll_left_deg):.4f}"
        )
    print(f"[insert_dynamic] rotation={rotation_label}")

    stats: Dict[str, Dict[str, int]] = {}
    for class_name in class_list:
        model = trainer.models[class_name]
        target_state = _normalize_instance_major_state(
            {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in model.state_dict().items()
            }
        )
        src_state = _align_source_time_dims(source_states[class_name], target_state)
        transformed_src_state = _apply_group_transform_to_state(
            state=src_state,
            q_global=q_global,
            source_center=source_group_center,
            target_center=target_anchor_pos,
        )
        merged_state = _append_class_state_for_eval(
            target_state=target_state,
            src_state=transformed_src_state,
        )
        merged_state = _normalize_instance_major_state(merged_state)

        old_instances = _get_num_instances(target_state)
        new_instances = _get_num_instances(merged_state)
        stats[class_name] = {
            "old_instances": int(old_instances),
            "new_instances": int(new_instances),
            "added_instances": int(new_instances - old_instances),
        }
        if new_instances > old_instances:
            inserted_ids = list(range(int(old_instances), int(new_instances)))
            print(
                "[insert_dynamic] "
                f"class={class_name} inserted_ids={inserted_ids} "
                f"added_instances={int(new_instances - old_instances)}"
            )

        try:
            model.load_state_dict(merged_state)
        except RuntimeError as exc:
            mismatches = _format_state_shape_mismatches(model, merged_state)
            detail = "\n".join(mismatches[:20])
            if len(mismatches) > 20:
                detail += f"\n... ({len(mismatches) - 20} more)"
            raise RuntimeError(
                f"Failed to load merged state for {class_name}: {exc}\n"
                f"State shape mismatches:\n{detail}"
            ) from exc

    return {
        "asset_path": os.path.abspath(asset_path),
        "classes": class_list,
        "applied_scale": float(applied_scale),
        "placement_mode": placement_label,
        "target_anchor_timestep": int(target_anchor_timestep),
        "source_anchor_timestep": int(source_anchor_timestep),
        "target_world_xyz": None if direct_target_pos is None else [float(v) for v in direct_target_pos.tolist()],
        "target_world_yaw_deg": None if target_world_yaw_deg is None else float(target_world_yaw_deg),
        "stats": stats,
    }
