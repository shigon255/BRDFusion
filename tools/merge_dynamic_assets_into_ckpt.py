import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset


POINT_ID_KEYS = ("points_ids", "point_ids")
TIME_INSTANCE_KEYS = {
    "instances_fv",
    "instances_trans",
    "instances_quats",
    "smpl_qauts",
}
SUPPORTED_CLASSES = ("RigidNodes", "SMPLNodes", "DeformableNodes")


def parse_instance_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        out.append(int(token))
    return sorted(set(out))


def parse_classes(raw: str) -> List[str]:
    classes = [x.strip() for x in raw.split(",") if x.strip()]
    if not classes:
        raise ValueError("No classes provided.")
    bad = [c for c in classes if c not in SUPPORTED_CLASSES]
    if bad:
        raise ValueError(f"Unsupported classes: {bad}. Supported: {SUPPORTED_CLASSES}")
    return classes


def get_point_id_key(state: Dict[str, torch.Tensor]) -> str:
    for key in POINT_ID_KEYS:
        if key in state:
            return key
    raise KeyError(f"State dict missing point id key: expected one of {POINT_ID_KEYS}")


def get_num_instances(state: Dict[str, torch.Tensor]) -> int:
    if "instances_fv" in state:
        return int(state["instances_fv"].shape[1])
    if "instances_size" in state:
        return int(state["instances_size"].shape[0])
    pid_key = get_point_id_key(state)
    pids = state[pid_key].reshape(-1).long()
    return int(torch.max(pids).item()) + 1 if pids.numel() > 0 else 0


def get_num_timesteps(state: Dict[str, torch.Tensor]) -> Optional[int]:
    for key in ("instances_trans", "instances_fv", "instances_quats", "smpl_qauts"):
        val = state.get(key, None)
        if torch.is_tensor(val) and val.ndim >= 1:
            return int(val.shape[0])
    return None


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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
    return quat_normalize(out)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    q_xyz = q[..., 1:4]
    qw = q[..., 0:1]
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (qw * uv + uuv)


def quat_from_axis_angle(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    axis = axis / (axis.norm() + 1e-12)
    half = 0.5 * angle_rad
    s = torch.sin(torch.tensor(half, dtype=torch.float32, device=axis.device))
    c = torch.cos(torch.tensor(half, dtype=torch.float32, device=axis.device))
    return torch.cat([c.view(1), axis * s], dim=0)


def build_camera_frame_rotation_quat(
    cam_right: torch.Tensor,
    cam_up: torch.Tensor,
    cam_forward: torch.Tensor,
    yaw_deg: float = 0.0,
    pitch_up_deg: float = 0.0,
    roll_left_deg: float = 0.0,
) -> torch.Tensor:
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=cam_up.device)
    if abs(yaw_deg) > 1e-12:
        q = quat_mul(q, quat_from_axis_angle(cam_up, np.deg2rad(yaw_deg)))
    if abs(pitch_up_deg) > 1e-12:
        q = quat_mul(q, quat_from_axis_angle(cam_right, np.deg2rad(pitch_up_deg)))
    if abs(roll_left_deg) > 1e-12:
        q = quat_mul(q, quat_from_axis_angle(cam_forward, np.deg2rad(roll_left_deg)))
    return quat_normalize(q)


def get_anchor_camera_frame(
    dataset: DrivingDataset,
    anchor_timestep: int,
    anchor_cam_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return quat_normalize(quats)


def compute_instance_centers_at_frame(
    state: Dict[str, torch.Tensor],
    frame_idx: int,
    use_visibility_mask: bool,
) -> torch.Tensor:
    pid_key = get_point_id_key(state)
    pids = state[pid_key].reshape(-1).long()
    means = state["_means"].float()
    num_instances = get_num_instances(state)

    trans = state.get("instances_trans", None)
    trans_frame = None
    if trans is not None:
        trans_frame = trans[frame_idx].float()

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
            local_center = torch.zeros(3, dtype=torch.float32)

        world_center = local_center
        if quats_frame is not None:
            world_center = quat_rotate(quats_frame[ins_id], world_center)
        if trans_frame is not None:
            world_center = world_center + trans_frame[ins_id]
        centers.append(world_center)

    if not centers:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.stack(centers, dim=0)


def choose_target_position(
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
) -> torch.Tensor:
    base_pos = cam_pos + cam_forward * base_forward_m + cam_right * base_right_m + cam_up * up_m

    if (not enable_search) or existing_centers.numel() == 0:
        return base_pos

    right_offsets = np.arange(-search_radius_m, search_radius_m + 1e-6, search_step_m, dtype=np.float32)
    forward_offsets = np.array([-2.0, 0.0, 2.0, 4.0, 6.0], dtype=np.float32)

    best_score = -1e9
    best_pos = base_pos

    for d_f in forward_offsets:
        f_m = max(1.0, base_forward_m + float(d_f))
        for d_r in right_offsets:
            r_m = base_right_m + float(d_r)
            cand = cam_pos + cam_forward * f_m + cam_right * r_m + cam_up * up_m
            dists = torch.norm(existing_centers - cand.view(1, 3), dim=1)
            min_dist = float(torch.min(dists).item()) if dists.numel() > 0 else 1e6
            pullback = float(torch.norm(cand - base_pos).item())
            score = min_dist - 0.05 * pullback
            if score > best_score:
                best_score = score
                best_pos = cand

    return best_pos


def _maybe_scale_source_states(
    source_states: Dict[str, Dict[str, torch.Tensor]],
    target_models: Dict[str, Dict[str, torch.Tensor]],
    scale_mode: str,
    manual_scale: float,
) -> float:
    if scale_mode == "manual":
        scale = float(manual_scale)
    elif scale_mode == "ego":
        ref_sizes: List[torch.Tensor] = []
        for class_name, state in target_models.items():
            if class_name not in source_states:
                continue
            if "instances_size" in state and state["instances_size"].numel() > 0:
                norms = torch.norm(state["instances_size"].float(), dim=-1)
                norms = norms[torch.isfinite(norms) & (norms > 1e-6)]
                if norms.numel() > 0:
                    ref_sizes.append(norms)
        if ref_sizes:
            ref_scale = float(torch.cat(ref_sizes, dim=0).median().item())
        else:
            ref_scale = 1.0

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
        if "_means" in state:
            state["_means"] = state["_means"] * scale
        if "_scales" in state:
            state["_scales"] = state["_scales"] + np.log(max(scale, 1e-8))
    return scale


def compute_group_center(
    class_states: Dict[str, Dict[str, torch.Tensor]],
    frame_idx: int,
) -> torch.Tensor:
    centers_all: List[torch.Tensor] = []
    for state in class_states.values():
        if "instances_trans" not in state:
            continue
        centers = compute_instance_centers_at_frame(state, frame_idx=frame_idx, use_visibility_mask=False)
        if centers.numel() > 0:
            centers_all.append(centers)
    if not centers_all:
        return torch.zeros(3, dtype=torch.float32)
    return torch.cat(centers_all, dim=0).mean(dim=0)


def apply_group_transform_to_state(
    state: Dict[str, torch.Tensor],
    q_global: torch.Tensor,
    source_center: torch.Tensor,
    target_center: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        out[k] = v.clone() if torch.is_tensor(v) else v

    if "instances_trans" in out:
        trans = out["instances_trans"].float()
        trans = quat_rotate(
            q_global.view(1, 1, 4),
            trans - source_center.view(1, 1, 3),
        ) + target_center.view(1, 1, 3)
        out["instances_trans"] = trans

    if "instances_quats" in out:
        q_old = out["instances_quats"].float()
        if q_old.ndim == 3:
            q_new = quat_mul(q_global.view(1, 1, 4), q_old)
        elif q_old.ndim == 4:
            q_new = quat_mul(q_global.view(1, 1, 1, 4), q_old)
        else:
            raise ValueError(f"Unsupported instances_quats shape: {tuple(q_old.shape)}")
        out["instances_quats"] = quat_normalize(q_new)

    return out


def append_class_state(
    target_state: Dict[str, torch.Tensor],
    src_state: Dict[str, torch.Tensor],
    class_name: str,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    t_pid_key = get_point_id_key(target_state)
    s_pid_key = get_point_id_key(src_state)

    t_num_points = int(target_state[t_pid_key].shape[0])
    s_num_points = int(src_state[s_pid_key].shape[0])
    t_num_instances = get_num_instances(target_state)
    s_num_instances = get_num_instances(src_state)

    src_point_ids_shifted = src_state[s_pid_key].long() + t_num_instances

    all_keys = sorted(set(target_state.keys()).union(set(src_state.keys())))
    for key in all_keys:
        t = target_state.get(key, None)
        s = src_state.get(key, None)

        if t is None:
            # Keep target schema stable; only bring over keys that already exist in target.
            continue
        if s is None:
            out[key] = t
            continue
        if (not torch.is_tensor(t)) or (not torch.is_tensor(s)):
            out[key] = t
            continue

        if key == t_pid_key:
            out[key] = torch.cat([t.long(), src_point_ids_shifted], dim=0)
            continue

        if key == s_pid_key and s_pid_key != t_pid_key:
            continue

        if t.ndim >= 1 and s.ndim >= 1 and t.shape[0] == t_num_points and s.shape[0] == s_num_points:
            out[key] = torch.cat([t, s], dim=0)
            continue

        if (
            t.ndim >= 2
            and s.ndim >= 2
            and t.shape[0] == s.shape[0]
            and t.shape[1] == t_num_instances
            and s.shape[1] == s_num_instances
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

        if key.startswith("template.") and t.ndim >= 1 and s.ndim >= 1 and t.shape[0] == t_num_instances and s.shape[0] == s_num_instances:
            out[key] = torch.cat([t, s], dim=0)
            continue

        out[key] = t

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge exported dynamic assets into an existing checkpoint."
    )
    parser.add_argument("--target_ckpt", type=str, required=True, help="Path to target checkpoint .pth")
    parser.add_argument("--asset_path", type=str, required=True, help="Path to exported dynamic asset .pth")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Path to output merged checkpoint .pth")

    parser.add_argument(
        "--classes",
        type=str,
        default="RigidNodes,SMPLNodes",
        help="Comma-separated classes to merge. Supported: RigidNodes,SMPLNodes,DeformableNodes",
    )
    parser.add_argument("--anchor_timestep", type=int, default=0, help="Anchor timestep in target scene")
    parser.add_argument("--anchor_cam_id", type=int, default=0, help="Anchor camera id in target scene")
    parser.add_argument(
        "--source_anchor_timestep",
        type=int,
        default=None,
        help="Anchor timestep in source asset (default: same as anchor_timestep)",
    )

    parser.add_argument("--forward_m", type=float, default=8.0, help="Forward offset from anchor camera")
    parser.add_argument("--right_m", type=float, default=0.0, help="Right offset from anchor camera")
    parser.add_argument("--up_m", type=float, default=0.0, help="Up offset from anchor camera")
    parser.add_argument("--yaw_deg", type=float, default=0.0, help="Yaw around camera up axis")
    parser.add_argument("--pitch_up_deg", type=float, default=0.0, help="Pitch-up around camera right axis")
    parser.add_argument("--roll_left_deg", type=float, default=0.0, help="Roll-left around camera forward axis")

    parser.add_argument(
        "--scale_mode",
        type=str,
        choices=["ego", "manual"],
        default="ego",
        help="Scale mode for imported asset",
    )
    parser.add_argument("--manual_scale", type=float, default=1.0, help="Manual scale when scale_mode=manual")

    parser.add_argument(
        "--disable_place_search",
        action="store_true",
        help="Disable collision-aware position search and use the base placement directly",
    )
    parser.add_argument("--search_radius_m", type=float, default=8.0, help="Lateral search radius (meters)")
    parser.add_argument("--search_step_m", type=float, default=2.0, help="Lateral search step (meters)")

    args = parser.parse_args()

    classes = parse_classes(args.classes)

    target_ckpt = torch.load(args.target_ckpt, map_location="cpu")
    asset = torch.load(args.asset_path, map_location="cpu")

    if "models" not in target_ckpt:
        raise KeyError("Target checkpoint does not contain 'models'.")
    if "classes" not in asset:
        raise KeyError("Asset package does not contain 'classes'.")

    target_models = target_ckpt["models"]
    asset_classes = asset["classes"]

    missing_in_asset = [c for c in classes if c not in asset_classes]
    if missing_in_asset:
        raise KeyError(f"Requested classes missing in asset: {missing_in_asset}")

    missing_in_target = [c for c in classes if c not in target_models]
    if missing_in_target:
        raise KeyError(
            "Requested classes missing in target checkpoint: "
            f"{missing_in_target}. Target config/model must already include these classes."
        )

    source_states: Dict[str, Dict[str, torch.Tensor]] = {}
    for class_name in classes:
        source_states[class_name] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k, v in asset_classes[class_name]["state_dict"].items()
        }

    target_states_for_scale = {c: target_models[c] for c in classes}
    applied_scale = _maybe_scale_source_states(
        source_states=source_states,
        target_models=target_states_for_scale,
        scale_mode=args.scale_mode,
        manual_scale=args.manual_scale,
    )

    log_dir = os.path.dirname(args.target_ckpt)
    cfg_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config for target checkpoint: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    dataset = DrivingDataset(data_cfg=cfg.data)

    target_anchor_timestep = int(np.clip(args.anchor_timestep, 0, dataset.num_img_timesteps - 1))

    requested_source_anchor_timestep = (
        args.anchor_timestep if args.source_anchor_timestep is None else args.source_anchor_timestep
    )
    source_time_lengths = [
        t for t in (get_num_timesteps(state) for state in source_states.values()) if t is not None
    ]
    if source_time_lengths:
        source_max_t = max(0, min(source_time_lengths) - 1)
        source_anchor_timestep = int(np.clip(requested_source_anchor_timestep, 0, source_max_t))
    else:
        source_anchor_timestep = 0

    if source_anchor_timestep != int(requested_source_anchor_timestep):
        print(
            "[WARN] source_anchor_timestep clipped from "
            f"{requested_source_anchor_timestep} to {source_anchor_timestep} "
            "based on source asset temporal length"
        )

    source_group_center = compute_group_center(source_states, frame_idx=source_anchor_timestep)

    cam_pos, cam_right, cam_up, cam_forward = get_anchor_camera_frame(
        dataset=dataset,
        anchor_timestep=target_anchor_timestep,
        anchor_cam_id=args.anchor_cam_id,
    )

    existing_centers_list: List[torch.Tensor] = []
    for class_name, state in target_models.items():
        if class_name not in SUPPORTED_CLASSES:
            continue
        if "instances_trans" not in state:
            continue
        centers = compute_instance_centers_at_frame(
            state,
            frame_idx=target_anchor_timestep,
            use_visibility_mask=True,
        )
        if centers.numel() > 0:
            existing_centers_list.append(centers)
    if existing_centers_list:
        existing_centers = torch.cat(existing_centers_list, dim=0)
    else:
        existing_centers = torch.empty((0, 3), dtype=torch.float32)

    target_anchor_pos = choose_target_position(
        cam_pos=cam_pos,
        cam_right=cam_right,
        cam_up=cam_up,
        cam_forward=cam_forward,
        base_forward_m=args.forward_m,
        base_right_m=args.right_m,
        up_m=args.up_m,
        existing_centers=existing_centers,
        enable_search=(not args.disable_place_search),
        search_radius_m=args.search_radius_m,
        search_step_m=args.search_step_m,
    )

    q_global = build_camera_frame_rotation_quat(
        cam_right=cam_right,
        cam_up=cam_up,
        cam_forward=cam_forward,
        yaw_deg=args.yaw_deg,
        pitch_up_deg=args.pitch_up_deg,
        roll_left_deg=args.roll_left_deg,
    )

    print(f"[INFO] Applied scale: {applied_scale:.6f}")
    print(f"[INFO] Source group center (anchor frame): {source_group_center.tolist()}")
    print(f"[INFO] Target anchor position: {target_anchor_pos.tolist()}")

    transformed_source_states: Dict[str, Dict[str, torch.Tensor]] = {}
    for class_name, state in source_states.items():
        transformed_source_states[class_name] = apply_group_transform_to_state(
            state=state,
            q_global=q_global,
            source_center=source_group_center,
            target_center=target_anchor_pos,
        )

    for class_name in classes:
        merged_state = append_class_state(
            target_state=target_models[class_name],
            src_state=transformed_source_states[class_name],
            class_name=class_name,
        )
        old_instances = get_num_instances(target_models[class_name])
        new_instances = get_num_instances(merged_state)
        target_models[class_name] = merged_state
        print(
            f"[OK] {class_name}: instances {old_instances} -> {new_instances}, "
            f"added {new_instances - old_instances}"
        )

    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save(target_ckpt, args.output_ckpt)
    print(f"[OK] Saved merged checkpoint: {args.output_ckpt}")


if __name__ == "__main__":
    main()
