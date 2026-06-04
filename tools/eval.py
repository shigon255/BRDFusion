from typing import Dict, List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
import numpy as np
import imageio

import torch
import torch.nn as nn
from datasets.driving_dataset import DrivingDataset
from datasets.base.pixel_source import get_rays
from utils.misc import import_str
from utils.config import resolve_saved_config
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views,
    save_single_image,
    save_single_hdr,
)
from models.modules import EnvLight_EQ
from tools.insert_utils import (
    _build_camera_frame_rotation_quat_batched,
    _get_anchor_camera_frame,
    _quat_mul_batched,
    insert_dynamic_asset_for_eval,
    insert_rigid_object_for_eval,
    set_inserted_rigid_rotation_for_eval,
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _video_output_dir(cfg: OmegaConf, args: argparse.Namespace, post_fix: str) -> str:
    return getattr(args, "video_output_dir", None) or os.path.join(cfg.log_dir, f"videos{post_fix}")


def _metrics_output_dir(cfg: OmegaConf, args: argparse.Namespace, post_fix: str) -> str:
    return getattr(args, "metrics_output_dir", None) or os.path.join(cfg.log_dir, f"metrics{post_fix}")


def _render_video_path(cfg: OmegaConf, args: argparse.Namespace, post_fix: str, prefix: str, step: int) -> str:
    if args.render_video_postfix is None:
        filename = f"{prefix}_{step}.mp4"
    else:
        filename = f"{prefix}_{step}_{args.render_video_postfix}.mp4"
    return os.path.join(_video_output_dir(cfg, args, post_fix), filename)


def _parse_eval_cam_ids(raw_cam_ids: Optional[str], num_cams: int) -> List[int]:
    if raw_cam_ids is None or str(raw_cam_ids).strip() == "":
        return list(range(num_cams))
    tokens = str(raw_cam_ids).replace(",", " ").split()
    cam_indices = []
    for token in tokens:
        compact_cam_id = int(token)
        if compact_cam_id < 0 or compact_cam_id >= num_cams:
            raise ValueError(
                f"--eval_cam_ids contains compact camera id {compact_cam_id}, "
                f"but valid ids are [0, {num_cams - 1}]."
            )
        if compact_cam_id not in cam_indices:
            cam_indices.append(compact_cam_id)
    if not cam_indices:
        raise ValueError("--eval_cam_ids did not contain any camera ids.")
    return cam_indices


def _concat_layout(imgs: List[np.ndarray], cam_names: List[str]) -> np.ndarray:
    max_height = max(img.shape[0] for img in imgs)
    channel = imgs[0].shape[-1]
    padded = []
    for img in imgs:
        canvas = np.zeros((max_height, img.shape[1], channel), dtype=img.dtype)
        canvas[max_height - img.shape[0] :, :] = img
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def _layout_for_selected_cams(dataset_layout, selected_cam_indices: List[int], num_cams: int):
    if selected_cam_indices == list(range(num_cams)):
        return dataset_layout
    return _concat_layout


def _cfg_bool(value, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _cfg_has_value(container, key: str) -> bool:
    value = container.get(key, None)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    try:
        return len(value) > 0
    except TypeError:
        return True


def _find_self_center_camera(pixel_source):
    camera_data = getattr(pixel_source, "camera_data", None)
    if not camera_data:
        raise ValueError("Automatic point-light reanchoring requires self pixel_source.camera_data.")
    cameras = list(camera_data.values())
    for camera in cameras:
        if getattr(camera, "cam_name", None) in {"Camera_Center", "Cam_Center"}:
            return camera
    raise ValueError("Automatic point-light reanchoring requires Camera_Center in the self dataset cameras.")


def _derive_self_point_light_reanchor_transform(dataset):
    from datasets.self.self_sourceloader import _load_pose_file, _resolve_camera_named_file

    center_camera = _find_self_center_camera(dataset.pixel_source)
    active_source = getattr(center_camera, "active_source", "primary")
    source_camera = center_camera
    if active_source == "external":
        external_camera = getattr(center_camera, "external_camera", None)
        if external_camera is None:
            raise ValueError("Self dataset active_source=external but center camera has no external camera.")
        source_camera = external_camera

    pose_path = _resolve_camera_named_file(source_camera.data_path, source_camera.cam_name, "_poses.txt")
    _, poses = _load_pose_file(pose_path)
    if len(poses) == 0:
        raise ValueError(f"No valid pose row found in {pose_path}")
    transform = np.linalg.inv(poses[0]).astype(np.float32)
    return transform, pose_path, active_source


def _setup_default_point_light_reanchor(cfg: OmegaConf, dataset) -> None:
    tracer_cfg = cfg.trainer.tracer
    light_file = tracer_cfg.get("point_lights_file", None)
    if light_file is None or str(light_file).strip() == "":
        return

    raw_reanchor = tracer_cfg.get("point_lights_file_reanchor", None)
    reanchor_explicit = raw_reanchor is not None
    reanchor = _cfg_bool(raw_reanchor, default=True)
    if reanchor is False:
        logger.info("Point-light-file reanchoring disabled by config.")
        return

    has_explicit_anchor = (
        _cfg_has_value(tracer_cfg, "point_lights_file_reanchor_transform")
        or _cfg_has_value(tracer_cfg, "point_lights_file_reanchor_pose_path")
    )
    if has_explicit_anchor:
        tracer_cfg.point_lights_file_reanchor = True
        return

    if cfg.data.dataset != "self":
        if reanchor_explicit:
            raise ValueError(
                "Automatic point-light-file reanchoring is only implemented for the self dataset. "
                "For non-self datasets, provide trainer.tracer.point_lights_file_reanchor_transform "
                "or point_lights_file_reanchor_pose_path, or set point_lights_file_reanchor=false."
            )
        logger.warning("Skipping automatic point-light-file reanchoring for non-self dataset.")
        tracer_cfg.point_lights_file_reanchor = False
        return

    transform, pose_path, active_source = _derive_self_point_light_reanchor_transform(dataset)
    tracer_cfg.point_lights_file_reanchor = True
    tracer_cfg.point_lights_file_reanchor_transform = transform.tolist()
    logger.info(
        "Auto-derived point-light-file reanchor transform from %s source center pose: %s",
        active_source,
        pose_path,
    )


def _orthonormalize_rotation(rot: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(rot)
    out = u @ vh
    if torch.det(out) < 0:
        u = u.clone()
        u[:, -1] = -u[:, -1]
        out = u @ vh
    return out


def _interpolate_c2w(c2w0: torch.Tensor, c2w1: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=c2w0.dtype, device=c2w0.device)
    out = c2w0.clone()
    out[:3, :3] = _orthonormalize_rotation((1.0 - alpha_t) * c2w0[:3, :3] + alpha_t * c2w1[:3, :3])
    out[:3, 3] = (1.0 - alpha_t) * c2w0[:3, 3] + alpha_t * c2w1[:3, 3]
    out[3, :] = c2w0[3, :]
    return out


def _scalar_from_image_info(value: torch.Tensor) -> torch.Tensor:
    return value.flatten()[0]


def _look_at_c2w(anchor_c2w: torch.Tensor, position: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = target - position
    forward = forward / (forward.norm() + 1e-8)
    ref_up = anchor_c2w[:3, 1]
    right = torch.cross(ref_up, forward, dim=0)
    if right.norm() < 1e-6:
        right = anchor_c2w[:3, 0]
    right = right / (right.norm() + 1e-8)
    up = torch.cross(forward, right, dim=0)
    up = up / (up.norm() + 1e-8)

    out = anchor_c2w.clone()
    out[:3, 0] = right
    out[:3, 1] = up
    out[:3, 2] = forward
    out[:3, 3] = position
    return out


class _InterpolatedCameraRenderSet:
    split = "interpolated"

    def __init__(
        self,
        datasource,
        start_timestep: int,
        num_timesteps: int,
        interp_steps: int,
        cam_indices: Optional[List[int]] = None,
    ):
        if interp_steps < 1:
            raise ValueError("interp_steps must be >= 1 for interpolated rendering.")
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be >= 1 for interpolated rendering.")
        self.datasource = datasource
        self.cam_indices = list(cam_indices or range(datasource.num_cams))
        self.num_cams = len(self.cam_indices)
        self.camera_list = [datasource.camera_list[i] for i in self.cam_indices]
        self.interp_steps = int(interp_steps)
        self.time_entries = []

        end_timestep = start_timestep + num_timesteps - 1
        for timestep in range(start_timestep, end_timestep):
            self.time_entries.append((timestep, timestep + 1, 0.0, timestep))
            for step in range(1, self.interp_steps + 1):
                alpha = step / float(self.interp_steps + 1)
                nearest = timestep if alpha < 0.5 else timestep + 1
                self.time_entries.append((timestep, timestep + 1, alpha, nearest))
        self.time_entries.append((end_timestep, end_timestep, 0.0, end_timestep))
        self.num_timestamps = len(self.time_entries)

    def __len__(self) -> int:
        return self.num_timestamps * self.num_cams

    def _full_index(self, frame_idx: int, compact_cam_idx: int) -> int:
        return frame_idx * self.datasource.num_cams + compact_cam_idx

    def get_image(self, idx: int, camera_downscale: float):
        timestamp_idx = idx // self.num_cams
        selected_cam_idx = idx % self.num_cams
        compact_cam_idx = self.cam_indices[selected_cam_idx]
        frame0, frame1, alpha, nearest_frame = self.time_entries[timestamp_idx]

        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        try:
            img0, cam0 = self.datasource.get_image(self._full_index(frame0, compact_cam_idx))
            img1, cam1 = self.datasource.get_image(self._full_index(frame1, compact_cam_idx))
            image_infos, cam_infos = self.datasource.get_image(self._full_index(nearest_frame, compact_cam_idx))
        finally:
            self.datasource.reset_downscale_factor()

        c2w = _interpolate_c2w(cam0["camera_to_world"], cam1["camera_to_world"], alpha)
        intrinsics = (1.0 - alpha) * cam0["intrinsics"] + alpha * cam1["intrinsics"]
        intrinsics = intrinsics.clone()
        intrinsics[2, 2] = 1.0

        img_height = int(cam_infos["height"].item())
        img_width = int(cam_infos["width"].item())
        x, y = torch.meshgrid(
            torch.arange(img_width, device=c2w.device),
            torch.arange(img_height, device=c2w.device),
            indexing="xy",
        )
        origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, intrinsics)
        image_infos["origins"] = origins.reshape(img_height, img_width, 3)
        image_infos["viewdirs"] = viewdirs.reshape(img_height, img_width, 3)
        image_infos["direction_norm"] = direction_norm.reshape(img_height, img_width, 1)

        if "normed_time" in image_infos and "normed_time" in img0 and "normed_time" in img1:
            t0 = _scalar_from_image_info(img0["normed_time"])
            t1 = _scalar_from_image_info(img1["normed_time"])
            normed_time = (1.0 - alpha) * t0 + alpha * t1
            image_infos["normed_time"] = torch.ones_like(image_infos["normed_time"]) * normed_time

        image_infos["frame_idx"] = torch.full_like(image_infos["frame_idx"], int(nearest_frame))
        image_infos.pop("pixels", None)
        image_infos.pop("relighted_pixels", None)
        image_infos.pop("blender_normal", None)
        image_infos.pop("blender_albedo", None)
        image_infos.pop("blender_metallic", None)
        image_infos.pop("blender_roughness", None)

        cam_infos["camera_to_world"] = c2w
        cam_infos["intrinsics"] = intrinsics
        return image_infos, cam_infos


class _SpiralCameraRenderSet:
    split = "spiral"

    def __init__(
        self,
        datasource,
        timestep: int,
        cam_id: int,
        frames: int,
        loops: float,
        radius_m: float,
        vertical_amplitude_m: float,
        target_distance_m: float,
    ):
        if datasource.num_cams != 1:
            raise ValueError("Spiral view rendering currently supports only 1-camera checkpoints.")
        if frames < 2:
            raise ValueError("spiral_frames must be >= 2.")
        if loops <= 0:
            raise ValueError("spiral_loops must be > 0.")
        if target_distance_m <= 0:
            raise ValueError("spiral_target_distance_m must be > 0.")
        if cam_id < 0 or cam_id >= datasource.num_cams:
            raise ValueError(f"spiral_cam_id={cam_id} is out of range [0, {datasource.num_cams - 1}].")
        if timestep < 0 or timestep >= datasource.num_frames:
            raise ValueError(f"spiral_timestep={timestep} is out of range [0, {datasource.num_frames - 1}].")

        self.datasource = datasource
        self.num_cams = 1
        self.camera_list = [datasource.camera_list[cam_id]]
        self.timestep = int(timestep)
        self.cam_id = int(cam_id)
        self.frames = int(frames)
        self.num_timestamps = self.frames
        self.loops = float(loops)
        self.radius_m = float(radius_m)
        self.vertical_amplitude_m = float(vertical_amplitude_m)
        self.target_distance_m = float(target_distance_m)

    def __len__(self) -> int:
        return self.frames

    def _full_index(self) -> int:
        return self.timestep * self.datasource.num_cams + self.cam_id

    def _spiral_c2w(self, anchor_c2w: torch.Tensor, frame_idx: int) -> torch.Tensor:
        if frame_idx == 0 or frame_idx == self.frames - 1:
            return anchor_c2w.clone()

        s = frame_idx / float(self.frames - 1)
        theta = 2.0 * np.pi * self.loops * s
        envelope = np.sin(np.pi * s)
        right = anchor_c2w[:3, 0]
        up = anchor_c2w[:3, 1]
        forward = anchor_c2w[:3, 2]
        position = anchor_c2w[:3, 3]
        offset = envelope * (
            self.radius_m * np.cos(theta) * right
            + self.vertical_amplitude_m * np.sin(theta) * up
        )
        target = position + self.target_distance_m * forward
        return _look_at_c2w(anchor_c2w, position + offset, target)

    def get_image(self, idx: int, camera_downscale: float):
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        try:
            image_infos, cam_infos = self.datasource.get_image(self._full_index())
        finally:
            self.datasource.reset_downscale_factor()

        c2w = self._spiral_c2w(cam_infos["camera_to_world"], idx)
        intrinsics = cam_infos["intrinsics"].clone()
        img_height = int(cam_infos["height"].item())
        img_width = int(cam_infos["width"].item())
        x, y = torch.meshgrid(
            torch.arange(img_width, device=c2w.device),
            torch.arange(img_height, device=c2w.device),
            indexing="xy",
        )
        origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, intrinsics)
        image_infos["origins"] = origins.reshape(img_height, img_width, 3)
        image_infos["viewdirs"] = viewdirs.reshape(img_height, img_width, 3)
        image_infos["direction_norm"] = direction_norm.reshape(img_height, img_width, 1)
        image_infos["frame_idx"] = torch.full_like(image_infos["frame_idx"], self.timestep)
        image_infos.pop("pixels", None)
        image_infos.pop("relighted_pixels", None)
        image_infos.pop("blender_normal", None)
        image_infos.pop("blender_albedo", None)
        image_infos.pop("blender_metallic", None)
        image_infos.pop("blender_roughness", None)

        cam_infos["camera_to_world"] = c2w
        cam_infos["intrinsics"] = intrinsics
        return image_infos, cam_infos


class _FixedCameraOffset(nn.Module):
    def __init__(
        self,
        forward_m: float = 0.0,
        left_m: float = 0.0,
        up_m: float = 0.0,
        yaw_left_deg: float = 0.0,
        pitch_up_deg: float = 0.0,
        roll_left_deg: float = 0.0,
    ):
        super().__init__()
        self.forward_m = float(forward_m)
        self.left_m = float(left_m)
        self.up_m = float(up_m)
        self.yaw_left_deg = float(yaw_left_deg)
        self.pitch_up_deg = float(pitch_up_deg)
        self.roll_left_deg = float(roll_left_deg)

    @staticmethod
    def _axis_angle_to_rotmat(axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
        axis = axis / (axis.norm() + 1e-12)
        x, y, z = axis[0], axis[1], axis[2]
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)
        C = 1.0 - c
        return torch.stack(
            [
                torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s]),
                torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s]),
                torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C]),
            ],
            dim=0,
        )

    def _apply_single(self, c2w: torch.Tensor) -> torch.Tensor:
        out = c2w.clone()
        R = out[:3, :3]

        if (
            abs(self.yaw_left_deg) > 1e-12
            or abs(self.pitch_up_deg) > 1e-12
            or abs(self.roll_left_deg) > 1e-12
        ):
            right = R[:, 0]
            up = R[:, 1]
            forward = R[:, 2]
            device = c2w.device
            dtype = c2w.dtype
            R_local = torch.eye(3, device=device, dtype=dtype)
            if abs(self.yaw_left_deg) > 1e-12:
                yaw_rad = torch.tensor(np.deg2rad(self.yaw_left_deg), device=device, dtype=dtype)
                R_local = R_local @ self._axis_angle_to_rotmat(up, yaw_rad)
            if abs(self.pitch_up_deg) > 1e-12:
                pitch_rad = torch.tensor(np.deg2rad(self.pitch_up_deg), device=device, dtype=dtype)
                R_local = R_local @ self._axis_angle_to_rotmat(right, pitch_rad)
            if abs(self.roll_left_deg) > 1e-12:
                roll_rad = torch.tensor(np.deg2rad(self.roll_left_deg), device=device, dtype=dtype)
                R_local = R_local @ self._axis_angle_to_rotmat(forward, roll_rad)

            out[:3, :3] = R @ R_local

        right = out[:3, 0]
        up = out[:3, 1]
        forward = out[:3, 2]
        delta = self.forward_m * forward + self.left_m * (-right) + self.up_m * up
        out[:3, 3] = out[:3, 3] + delta
        return out

    def forward(self, camtoworlds: torch.Tensor, image_ids: torch.Tensor = None) -> torch.Tensor:
        c2w = camtoworlds
        if c2w.ndim == 2:
            return self._apply_single(c2w)
        shape = c2w.shape
        flat = c2w.reshape(-1, 4, 4)
        out = torch.stack([self._apply_single(m) for m in flat], dim=0)
        return out.reshape(shape)


class _ComposedCamPosePerturb(nn.Module):
    def __init__(self, base_module: nn.Module, offset_module: _FixedCameraOffset):
        super().__init__()
        self.base_module = base_module
        self.offset_module = offset_module

    def forward(self, camtoworlds: torch.Tensor, image_ids: torch.Tensor = None) -> torch.Tensor:
        out = self.base_module(camtoworlds, image_ids)
        out = self.offset_module(out, image_ids)
        return out


def _to8b(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def _parse_instance_ids_csv(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        tokens = raw
    else:
        raw = str(raw).strip()
        if raw == "":
            return []
        tokens = raw.split(",")

    out: List[int] = []
    for token in tokens:
        if token is None:
            continue
        token = str(token).strip()
        if token == "":
            continue
        out.append(int(token))
    return sorted(set(out))


def _apply_instance_deletions(
    trainer: BasicTrainer,
    delete_spec: Dict[str, Optional[str]],
) -> None:
    for class_name, raw_ids in delete_spec.items():
        ids = _parse_instance_ids_csv(raw_ids)
        if not ids:
            continue

        if class_name not in trainer.models:
            logger.warning("Skip deletion: %s not found in trainer models.", class_name)
            continue

        model = trainer.models[class_name]
        if not hasattr(model, "remove_instances"):
            logger.warning("Skip deletion: %s does not implement remove_instances().", class_name)
            continue

        num_instances = getattr(model, "num_instances", None)
        valid_ids: List[int] = []
        for ins_id in ids:
            if num_instances is None:
                valid_ids.append(ins_id)
            elif 0 <= ins_id < int(num_instances):
                valid_ids.append(ins_id)
            else:
                logger.warning(
                    "Skip invalid %s instance id %d (valid range: [0, %d]).",
                    class_name,
                    ins_id,
                    int(num_instances) - 1,
                )

        if not valid_ids:
            continue

        # Delete in descending order to avoid index shift issues.
        valid_ids = sorted(set(valid_ids), reverse=True)
        model.remove_instances(valid_ids)
        logger.info("Deleted %s instances: %s", class_name, valid_ids)


def _apply_rigid_instance_translation(
    trainer: BasicTrainer,
    raw_ids: Optional[str],
    offset_xyz: List[float],
) -> None:
    ids = _parse_instance_ids_csv(raw_ids)
    if not ids:
        if any(abs(v) > 1e-12 for v in offset_xyz):
            logger.warning("Ignoring rigid translation offset because --move_rigid_ids is empty.")
        return

    if "RigidNodes" not in trainer.models:
        raise ValueError("--move_rigid_ids was provided, but RigidNodes is not present in this trainer.")

    model = trainer.models["RigidNodes"]
    if not hasattr(model, "instances_trans"):
        raise ValueError("RigidNodes does not expose instances_trans; cannot apply rigid translation.")

    num_instances = getattr(model, "num_instances", None)
    if num_instances is None:
        raise ValueError("RigidNodes does not expose num_instances; cannot validate --move_rigid_ids.")

    bad_ids = [ins_id for ins_id in ids if ins_id < 0 or ins_id >= int(num_instances)]
    if bad_ids:
        raise ValueError(
            f"Invalid RigidNodes instance IDs for translation: {bad_ids}. "
            f"Valid range is [0, {int(num_instances) - 1}]."
        )

    device = model.instances_trans.device
    dtype = model.instances_trans.dtype
    offset = torch.tensor(offset_xyz, device=device, dtype=dtype)
    if torch.all(torch.abs(offset) <= 1e-12):
        logger.info("Rigid translation for IDs %s is zero; leaving trajectories unchanged.", ids)
        return

    trans_before = model.instances_trans.data[:, ids, :].detach().clone()
    model.instances_trans.data[:, ids, :] = model.instances_trans.data[:, ids, :] + offset.view(1, 1, 3)
    trans_after = model.instances_trans.data[:, ids, :].detach()
    sample_frames = sorted(set([0, trans_after.shape[0] // 2, trans_after.shape[0] - 1]))
    for local_idx, ins_id in enumerate(ids):
        for frame_idx in sample_frames:
            before_xyz = trans_before[frame_idx, local_idx].detach().cpu().tolist()
            after_xyz = trans_after[frame_idx, local_idx].detach().cpu().tolist()
            print(
                "[move_rigid] "
                f"id={ins_id} frame={frame_idx} "
                f"before_xyz=({before_xyz[0]:.4f}, {before_xyz[1]:.4f}, {before_xyz[2]:.4f}) "
                f"after_xyz=({after_xyz[0]:.4f}, {after_xyz[1]:.4f}, {after_xyz[2]:.4f})"
            )
    logger.info(
        "Applied eval-only world translation to RigidNodes IDs %s: "
        "dx=%.4f dy=%.4f dz=%.4f meters",
        ids,
        float(offset[0].item()),
        float(offset[1].item()),
        float(offset[2].item()),
    )


def _load_rigid_move_specs(specs_json_path: str) -> List[Dict[str, object]]:
    specs_path = os.path.abspath(specs_json_path)
    with open(specs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        specs = data.get("items", None)
        if specs is None:
            raise ValueError("Rigid move specs JSON must be a list or a dict with an 'items' list.")
    elif isinstance(data, list):
        specs = data
    else:
        raise ValueError("Rigid move specs JSON must be a list or a dict with an 'items' list.")

    if not specs:
        raise ValueError(f"No rigid move specs found in: {specs_path}")

    defaults = {
        "anchor_timestep": 0,
        "anchor_cam_id": 0,
        "forward_m": 0.0,
        "right_m": 0.0,
        "up_m": 0.0,
        "yaw_deg": 0.0,
        "scale": 1.0,
    }
    out: List[Dict[str, object]] = []
    for idx, raw in enumerate(specs):
        if not isinstance(raw, dict):
            raise ValueError(f"Rigid move spec #{idx} is not an object: {raw!r}")
        if raw.get("enabled", True) is False:
            continue
        raw_ids = raw.get("rigid_ids", raw.get("ids", None))
        ids = _parse_instance_ids_csv(raw_ids)
        if not ids:
            raise ValueError(f"Rigid move spec #{idx} missing required non-empty 'rigid_ids'.")
        spec = dict(defaults)
        spec.update(raw)
        spec["rigid_ids"] = ids
        spec["anchor_timestep"] = int(spec["anchor_timestep"])
        spec["anchor_cam_id"] = int(spec["anchor_cam_id"])
        spec["forward_m"] = float(spec["forward_m"])
        spec["right_m"] = float(spec["right_m"])
        spec["up_m"] = float(spec["up_m"])
        spec["yaw_deg"] = float(spec["yaw_deg"])
        spec["scale"] = float(spec["scale"])
        if spec["scale"] <= 0.0:
            raise ValueError(f"Rigid move spec #{idx} has non-positive scale: {spec['scale']}")
        out.append(spec)

    if not out:
        raise ValueError(f"All rigid move specs were disabled or invalid in: {specs_path}")
    return out


def _apply_rigid_instance_camera_relative_move(
    trainer: BasicTrainer,
    dataset: DrivingDataset,
    raw_ids,
    anchor_timestep: int,
    anchor_cam_id: int,
    forward_m: float,
    right_m: float,
    up_m: float,
    yaw_deg: float,
    scale: float,
) -> None:
    ids = _parse_instance_ids_csv(raw_ids)
    if not ids:
        raise ValueError("Camera-relative rigid move requires at least one RigidNodes id.")
    if scale <= 0.0:
        raise ValueError(f"Rigid move scale must be positive, got {scale}.")
    if anchor_timestep < 0:
        raise ValueError(f"Rigid move anchor_timestep must be >= 0, got {anchor_timestep}.")
    if anchor_cam_id < 0:
        raise ValueError(f"Rigid move anchor_cam_id must be >= 0, got {anchor_cam_id}.")

    if "RigidNodes" not in trainer.models:
        raise ValueError("Rigid move was requested, but RigidNodes is not present in this trainer.")

    model = trainer.models["RigidNodes"]
    if not hasattr(model, "instances_trans"):
        raise ValueError("RigidNodes does not expose instances_trans; cannot apply rigid move.")
    if not hasattr(model, "instances_quats"):
        raise ValueError("RigidNodes does not expose instances_quats; cannot apply rigid yaw.")

    num_instances = getattr(model, "num_instances", None)
    if num_instances is None:
        raise ValueError("RigidNodes does not expose num_instances; cannot validate moved IDs.")
    bad_ids = [ins_id for ins_id in ids if ins_id < 0 or ins_id >= int(num_instances)]
    if bad_ids:
        raise ValueError(
            f"Invalid RigidNodes instance IDs for camera-relative move: {bad_ids}. "
            f"Valid range is [0, {int(num_instances) - 1}]."
        )

    cam_pos, cam_right, cam_down, cam_forward = _get_anchor_camera_frame(
        dataset=dataset,
        anchor_timestep=anchor_timestep,
        anchor_cam_id=anchor_cam_id,
    )
    cam_up = -cam_down
    device = model.instances_trans.device
    dtype = model.instances_trans.dtype
    offset = (
        cam_forward.to(device=device, dtype=dtype) * float(forward_m)
        + cam_right.to(device=device, dtype=dtype) * float(right_m)
        + cam_up.to(device=device, dtype=dtype) * float(up_m)
    )

    q_delta = _build_camera_frame_rotation_quat_batched(
        cam_right=cam_right.to(device=device, dtype=dtype),
        cam_up=cam_up.to(device=device, dtype=dtype),
        cam_forward=cam_forward.to(device=device, dtype=dtype),
        yaw_deg=float(yaw_deg),
        pitch_up_deg=0.0,
        roll_left_deg=0.0,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        trans_before = model.instances_trans.data[:, ids, :].detach().clone()
        if torch.any(torch.abs(offset) > 1e-12):
            model.instances_trans.data[:, ids, :] = model.instances_trans.data[:, ids, :] + offset.view(1, 1, 3)
        trans_after = model.instances_trans.data[:, ids, :].detach()

        if abs(float(yaw_deg)) > 1e-12:
            old_quats = model.instances_quats.data[:, ids, :]
            model.instances_quats.data[:, ids, :] = _quat_mul_batched(q_delta, old_quats)

        if abs(float(scale) - 1.0) > 1e-12:
            point_ids = getattr(model, "point_ids", None)
            if point_ids is None:
                raise ValueError("RigidNodes does not expose point_ids; cannot apply rigid scale.")
            point_ids_flat = point_ids[..., 0].long()
            scale_t = torch.tensor(float(scale), device=model._means.device, dtype=model._means.dtype)
            for ins_id in ids:
                pts_mask = point_ids_flat == ins_id
                if not bool(torch.any(pts_mask)):
                    logger.warning("Rigid move scale skipped id=%d because it has no Gaussian points.", ins_id)
                    continue
                center = model._means.data[pts_mask].mean(dim=0, keepdim=True)
                model._means.data[pts_mask] = center + (model._means.data[pts_mask] - center) * scale_t
                if hasattr(model, "_scales"):
                    model._scales.data[pts_mask] = model._scales.data[pts_mask] + torch.log(scale_t)
                if hasattr(model, "instances_size"):
                    model.instances_size.data[ins_id] = model.instances_size.data[ins_id] * scale_t.to(
                        device=model.instances_size.device,
                        dtype=model.instances_size.dtype,
                    )

    sample_frames = sorted(set([0, trans_after.shape[0] // 2, trans_after.shape[0] - 1]))
    for local_idx, ins_id in enumerate(ids):
        for frame_idx in sample_frames:
            before_xyz = trans_before[frame_idx, local_idx].detach().cpu().tolist()
            after_xyz = trans_after[frame_idx, local_idx].detach().cpu().tolist()
            print(
                "[move_rigid] "
                f"id={ins_id} frame={frame_idx} "
                f"before_xyz=({before_xyz[0]:.4f}, {before_xyz[1]:.4f}, {before_xyz[2]:.4f}) "
                f"after_xyz=({after_xyz[0]:.4f}, {after_xyz[1]:.4f}, {after_xyz[2]:.4f})"
            )
    logger.info(
        "Applied eval-only camera-relative rigid move to IDs %s: "
        "anchor_timestep=%d anchor_cam_id=%d forward=%.4f right=%.4f up=%.4f "
        "yaw=%.4f scale=%.6f world_offset=(%.4f, %.4f, %.4f)",
        ids,
        int(anchor_timestep),
        int(anchor_cam_id),
        float(forward_m),
        float(right_m),
        float(up_m),
        float(yaw_deg),
        float(scale),
        float(offset[0].item()),
        float(offset[1].item()),
        float(offset[2].item()),
    )


def _material_value_to_logit(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Material override values must be in [0, 1], got {value}.")
    eps = 1e-6
    clamped = min(max(float(value), eps), 1.0 - eps)
    return torch.logit(torch.tensor(clamped, device=device, dtype=dtype))


def _apply_gaussian_material_overrides(
    trainer: BasicTrainer,
    metallic: Optional[float] = None,
    roughness: Optional[float] = None,
) -> None:
    if metallic is None and roughness is None:
        return

    updated = []
    with torch.no_grad():
        for class_name, model in trainer.models.items():
            changed = []
            if metallic is not None and hasattr(model, "_metallics"):
                param = model._metallics
                if torch.is_tensor(param) and param.numel() > 0:
                    param.data.fill_(_material_value_to_logit(metallic, param.device, param.dtype).item())
                    changed.append(f"metallic={float(metallic):.4f}")
            if roughness is not None and hasattr(model, "_roughnesses"):
                param = model._roughnesses
                if torch.is_tensor(param) and param.numel() > 0:
                    param.data.fill_(_material_value_to_logit(roughness, param.device, param.dtype).item())
                    changed.append(f"roughness={float(roughness):.4f}")
            if changed:
                num_points = getattr(model, "num_points", None)
                updated.append(f"{class_name}(points={num_points}, {', '.join(changed)})")

    if not updated:
        logger.warning("Requested material overrides, but no Gaussian material tensors were found.")
        return

    print("[material_override] " + " | ".join(updated))
    logger.info("Applied eval-only Gaussian material overrides: %s", " | ".join(updated))


def _build_inserted_only_render_keys(classes_csv: str) -> List[str]:
    class_names = [x.strip() for x in classes_csv.split(",") if x.strip()]
    keys: List[str] = []
    for class_name in class_names:
        keys.append(f"{class_name}_rgbs")
        keys.append(f"{class_name}_depths")
        keys.append(f"{class_name}_opacities")
    if len(class_names) > 1:
        keys.extend(["Dynamic_rgbs", "Dynamic_depths", "Dynamic_opacities"])
    return keys


def _extend_unique(base: List[str], extra: List[str]) -> List[str]:
    out = list(base)
    seen = set(out)
    for key in extra:
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _resolve_asset_classes_from_package(asset_path: str) -> str:
    package = torch.load(asset_path, map_location="cpu")
    package_classes = package.get("classes", {})
    if not package_classes:
        raise KeyError(f"Dynamic asset package missing classes: {asset_path}")
    return ",".join(package_classes.keys())


def _parse_optional_world_xyz(raw) -> Optional[List[float]]:
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
        raise ValueError(f"target_world_xyz must have exactly 3 values, got: {raw!r}")
    return [float(v) for v in values]


def _load_insert_dynamic_specs(
    specs_json_path: str,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    specs_path = os.path.abspath(specs_json_path)
    with open(specs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        specs = data.get("items", None)
        if specs is None:
            raise ValueError("Specs JSON must be a list or a dict with an 'items' list.")
    elif isinstance(data, list):
        specs = data
    else:
        raise ValueError("Specs JSON must be a list or a dict with an 'items' list.")

    if not specs:
        raise ValueError(f"No insertion specs found in: {specs_path}")

    simplified_specs = bool(getattr(args, "insert_specs_simple", False))
    if simplified_specs:
        defaults = {
            "anchor_timestep": 0,
            "anchor_cam_id": 0,
            "source_anchor_timestep": None,
            "forward_m": 8.0,
            "right_m": 0.0,
            "up_m": 0.0,
            "yaw_deg": 0.0,
            "pitch_up_deg": 0.0,
            "roll_left_deg": 0.0,
            "scale_mode": "manual",
            "manual_scale": 1.0,
            "disable_place_search": True,
            "search_radius_m": args.insert_dynamic_search_radius_m,
            "search_step_m": args.insert_dynamic_search_step_m,
            "placement_mode": args.insert_dynamic_placement_mode,
            "target_world_xyz": None,
            "target_world_timestep": None,
            "target_world_yaw_deg": None,
        }
    else:
        defaults = {
            "anchor_timestep": args.insert_anchor_timestep,
            "anchor_cam_id": args.insert_cam_id,
            "source_anchor_timestep": args.insert_dynamic_source_anchor_timestep,
            "forward_m": args.insert_forward_m,
            "right_m": args.insert_right_m,
            "up_m": args.insert_up_m,
            "yaw_deg": args.insert_yaw_deg,
            "pitch_up_deg": args.insert_pitch_up_deg,
            "roll_left_deg": args.insert_roll_left_deg,
            "scale_mode": args.insert_scale_mode,
            "manual_scale": args.insert_manual_scale,
            "disable_place_search": args.insert_dynamic_disable_place_search,
            "search_radius_m": args.insert_dynamic_search_radius_m,
            "search_step_m": args.insert_dynamic_search_step_m,
            "placement_mode": args.insert_dynamic_placement_mode,
            "target_world_xyz": _parse_optional_world_xyz(args.insert_target_world_xyz),
            "target_world_timestep": args.insert_target_world_timestep,
            "target_world_yaw_deg": args.insert_target_world_yaw_deg,
        }

    out: List[Dict[str, object]] = []
    for idx, raw in enumerate(specs):
        if not isinstance(raw, dict):
            raise ValueError(f"Spec #{idx} is not an object: {raw!r}")
        if raw.get("enabled", True) is False:
            continue

        asset_path = raw.get("asset_path", None)
        if not asset_path:
            raise ValueError(f"Spec #{idx} missing required field 'asset_path'.")
        if not os.path.isabs(asset_path):
            asset_path = os.path.abspath(os.path.join(os.path.dirname(specs_path), asset_path))

        spec = dict(defaults)
        spec.update(raw)
        spec["asset_path"] = asset_path
        if simplified_specs:
            spec["up_m"] = -float(spec.get("up_m", 0.0))
            if "scale" in raw:
                spec["scale_mode"] = "manual"
                spec["manual_scale"] = float(raw["scale"])
            if "auto_place" in raw:
                spec["disable_place_search"] = not bool(raw["auto_place"])

        classes = spec.get("classes", None)
        if isinstance(classes, (list, tuple)):
            spec["classes"] = ",".join(str(x).strip() for x in classes if str(x).strip())
        elif classes is None or str(classes).strip() == "":
            spec["classes"] = _resolve_asset_classes_from_package(asset_path)

        out.append(spec)

    if not out:
        raise ValueError(f"All insertion specs were disabled or invalid in: {specs_path}")
    return out


@torch.no_grad()
def do_insert_rotation_sweep(
    cfg: OmegaConf,
    trainer: BasicTrainer,
    dataset: DrivingDataset,
    args: argparse.Namespace,
    post_fix: str = "",
):
    if not hasattr(args, "inserted_instance_id"):
        raise ValueError("Rotation sweep requires an inserted object. Use --insert_rigid_ply first.")
    ins_id = int(args.inserted_instance_id)
    trainer.set_eval()

    num_cams = dataset.pixel_source.num_cams
    frame_idx = int(np.clip(args.insert_sweep_frame, 0, dataset.num_img_timesteps - 1))
    cam_id = int(np.clip(args.insert_sweep_cam_id, 0, num_cams - 1))
    vis_idx = frame_idx * num_cams + cam_id

    angles = np.linspace(args.insert_sweep_yaw_start, args.insert_sweep_yaw_end, args.insert_sweep_num)
    out_dir = os.path.join(_video_output_dir(cfg, args, post_fix), f"insert_sweep_frame{frame_idx:03d}_cam{cam_id}")
    os.makedirs(out_dir, exist_ok=True)
    frames = []

    logger.info(
        f"Running insertion yaw sweep: frame={frame_idx}, cam={cam_id}, "
        f"angles=[{angles[0]:.2f}, ..., {angles[-1]:.2f}] ({len(angles)} steps)"
    )

    for i, yaw in enumerate(angles):
        set_inserted_rigid_rotation_for_eval(
            trainer=trainer,
            dataset=dataset,
            instance_id=ins_id,
            yaw_deg=float(yaw),
            pitch_up_deg=args.insert_pitch_up_deg,
            roll_left_deg=args.insert_roll_left_deg,
            anchor_timestep=args.insert_anchor_timestep,
            anchor_cam_id=args.insert_cam_id,
        )
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=False,
            compute_error_map=False,
            vis_indices=[vis_idx],
            verbose=False,
            timing=args.enable_timing,
        )
        key = "pbr_colors" if "pbr_colors" in render_results else "rgbs"
        if key not in render_results or len(render_results[key]) == 0:
            raise RuntimeError(f"Failed to render key '{key}' during rotation sweep.")
        frame = _to8b(render_results[key][0])
        frames.append(frame)
        imageio.imwrite(os.path.join(out_dir, f"sweep_{i:04d}_yaw_{yaw:+07.2f}.png"), frame)

    video_path = os.path.join(out_dir, "rotation_sweep.mp4")
    imageio.mimwrite(video_path, frames, fps=args.insert_sweep_fps)
    logger.info(f"Saved insertion rotation sweep video to {video_path}")


@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True
):
    if args.insert_rotate_sweep:
        do_insert_rotation_sweep(
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            args=args,
            post_fix=post_fix,
        )
        return

    trainer.set_eval() 
    
    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            verbose=True,
            timing=args.enable_timing,
            measure_fps=args.measure_fps,
        )
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "sky_psnr",
                    "ground_psnr",
                    "ground_pbr_psnr",
                    "pbr_psnr",
                    "pbr_ssim",
                    "pbr_lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = os.path.join(_metrics_output_dir(cfg, args, post_fix), f"images_test_{current_time}.json")
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        video_output_pth = _render_video_path(cfg, args, post_fix, "test_set", step)
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_test_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=False,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
        
    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")
        tmp_path = f"{cfg.log_dir}/visualize"
        os.makedirs(tmp_path, exist_ok=True)
        
        # # debugging
        # print("debugging: filtering gaussians...")
        # # get the camera pose at timestep 38 as anchor
        # num_cams = dataset.pixel_source.num_cams
        # idx = num_cams * 38  # timestep 38
        # anchor_cam_info = dataset.full_image_set.get_image(idx, 1.0)[1]
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
        #     # xyz: (N, 3), torch tensor
        #     mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)

        #     # filter gs that is right to the anchor
        #     offset = right * 1.0
        #     rel_pos = xyz - (offset + anchor_pos).reshape(1, 3)  # (N, 3)
        #     rel_pos_vec = rel_pos / (rel_pos.norm(dim=1, keepdim=True) + 1e-8)
        #     right_dot = (rel_pos_vec * right.reshape(1, 3)).sum(dim=1)  # (N,)
        #     mask = mask & (right_dot > 0)

        #     # also filter out gs behind the anchor
        #     forward_dot = (rel_pos_vec * forward.reshape(1, 3)).sum(dim=1)  # (N,)
        #     mask = mask & (forward_dot > 0)

        #     # # also filter out gs that is too above the anchor
        #     # up_dot = (rel_pos_vec * up.reshape(1, 3)).sum(dim=1)  # (N,)
        #     # up_dis = up_dot * rel_pos.norm(dim=1)  # (N,)
        #     # mask = mask & (up_dis < 3.0)

        #     return mask
        # trainer.models["Background"].mask_gs(mask_func)
        
        # # filter out invalid samples
        # print("Debugging: filtering invalid envmap samples...")
        # # invalid_samples = [
        # #     (183, 185),
        # #     (106, 236),
        # #     (33, 236),
        # #     (183, 184),
        # #     (84, 236),
        # #     (27, 238),
        # #     (93, 187),
        # #     (18, 239)
        # # ]
        # invalid_samples = [
        #     (24, 240),
        #     (163, 187),
        #     (23, 240),
        #     (163, 188),
        #     (111, 234),
        #     (108, 192)
        # ]
        # for coord in invalid_samples:
        #     trainer.models['Sky'].base.data[coord[1], coord[0], :] = 0.001
        # trainer.models['Sky'].update_pdf()
        
        num_cams = dataset.pixel_source.num_cams
        num_total_timesteps = dataset.num_img_timesteps
        selected_cam_indices = _parse_eval_cam_ids(args.eval_cam_ids, num_cams)
        render_num_cams = len(selected_cam_indices)
        render_layout = _layout_for_selected_cams(dataset.layout, selected_cam_indices, num_cams)
        camera_interp_steps = int(args.camera_interp_steps or 0)
        spiral_timestep = args.spiral_timestep
        if spiral_timestep is not None and camera_interp_steps > 0:
            raise ValueError("--spiral_timestep cannot be combined with --camera_interp_steps.")
        
        start_timestep = int(args.eval_start_timestep or 0)
        if start_timestep < 0 or start_timestep >= num_total_timesteps:
            raise ValueError(
                f"--eval_start_timestep={start_timestep} is out of range "
                f"[0, {num_total_timesteps - 1}]."
            )
        
        num_eval_timesteps = num_total_timesteps - start_timestep
        if args.eval_num_timesteps is not None:
            if args.eval_num_timesteps < 1:
                raise ValueError(f"--eval_num_timesteps must be >= 1, got {args.eval_num_timesteps}.")
            num_eval_timesteps = min(args.eval_num_timesteps, num_total_timesteps - start_timestep)

        logger.info(
            "Full dataset loaded for rendering: timesteps=%d, cameras=%d. "
            "Selected render range: start=%d, count=%d, compact_cam_ids=%s.",
            num_total_timesteps,
            num_cams,
            start_timestep,
            num_eval_timesteps,
            selected_cam_indices,
        )

        if spiral_timestep is not None:
            render_dataset = _SpiralCameraRenderSet(
                datasource=dataset.pixel_source,
                timestep=int(spiral_timestep),
                cam_id=int(args.spiral_cam_id),
                frames=int(args.spiral_frames),
                loops=float(args.spiral_loops),
                radius_m=float(args.spiral_radius_m),
                vertical_amplitude_m=float(args.spiral_vertical_amplitude_m),
                target_distance_m=float(args.spiral_target_distance_m),
            )
            vis_indices = None
            render_num_timestamps = render_dataset.num_timestamps
            render_fps = int(args.spiral_fps or cfg.render.fps)
            render_compute_metrics = False
            print(
                f"Rendering spiral view at timestep {spiral_timestep}, compact camera "
                f"{args.spiral_cam_id}, {render_num_timestamps} frames, {args.spiral_loops} "
                f"loops, radius {args.spiral_radius_m}m, vertical amplitude "
                f"{args.spiral_vertical_amplitude_m}m, target distance "
                f"{args.spiral_target_distance_m}m. Saving at {render_fps} FPS."
            )
        elif camera_interp_steps > 0:
            render_dataset = _InterpolatedCameraRenderSet(
                datasource=dataset.pixel_source,
                start_timestep=start_timestep,
                num_timesteps=num_eval_timesteps,
                interp_steps=camera_interp_steps,
                cam_indices=selected_cam_indices,
            )
            vis_indices = None
            render_num_timestamps = render_dataset.num_timestamps
            render_fps = int(cfg.render.fps) * (camera_interp_steps + 1)
            render_compute_metrics = False
            print(
                f"Rendering {num_eval_timesteps} source timesteps from {start_timestep} to "
                f"{start_timestep + num_eval_timesteps - 1}, each with {render_num_cams} selected cameras and "
                f"{camera_interp_steps} interpolated poses between timesteps, total "
                f"{len(render_dataset)} images. Saving at {render_fps} FPS."
            )
        else:
            render_dataset = dataset.full_image_set
            render_num_timestamps = num_eval_timesteps
            render_fps = cfg.render.fps
            render_compute_metrics = True
            vis_indices = []
            for timestep in range(start_timestep, start_timestep + num_eval_timesteps):
                for cam_id in selected_cam_indices:
                    vis_indices.append(timestep * num_cams + cam_id)
            print(
                f"Rendering {num_eval_timesteps} timesteps from {start_timestep} to "
                f"{start_timestep + num_eval_timesteps - 1}, selected compact cameras "
                f"{selected_cam_indices}, total {len(vis_indices)} images. "
                f"Dataset remains full: {num_total_timesteps} timesteps, {num_cams} cameras."
            )
        
        render_results = render_images(
            trainer=trainer,
            dataset=render_dataset,
            compute_metrics=render_compute_metrics,
            compute_error_map=cfg.render.vis_error and render_compute_metrics,
            tmp_path=tmp_path,
            vis_indices=vis_indices,
            isosurface_render=args.isosurface_render,
            timing=args.enable_timing,
            measure_fps=args.measure_fps,
        )
        
        if log_metrics and render_compute_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "pbr_psnr",
                    "pbr_ssim",
                    "pbr_lpips",
                    "sky_psnr",
                    "ground_psnr",
                    "ground_pbr_psnr",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            full_metrics_file = os.path.join(_metrics_output_dir(cfg, args, post_fix), f"images_full_{current_time}.json")
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        video_output_pth = _render_video_path(cfg, args, post_fix, "full_set", step)
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=render_layout,
            # num_timestamps=dataset.num_img_timesteps,
            num_timestamps=render_num_timestamps,
            keys=render_keys,
            num_cams=render_num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=render_fps,
            verbose=True,
        )
        if "point_light_emitters_rgbs" in render_results and len(render_results["point_light_emitters_rgbs"]) > 0:
            emitter_only_path = video_output_pth.replace(".mp4", "_emitters_only.mp4")
            _ = save_videos(
                render_results,
                emitter_only_path,
                layout=render_layout,
                num_timestamps=render_num_timestamps,
                keys=["point_light_emitters_rgbs"],
                num_cams=render_num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=render_fps,
                verbose=True,
            )
            logger.info(f"Emitter-only video saved to {emitter_only_path}")
        
        if 'envmap_visualize' in render_results:
            # save envmap
            envmap_path = os.path.join(_video_output_dir(cfg, args, post_fix), "envmap.png")
            save_single_image(image=render_results['envmap_visualize'], save_pth=envmap_path)
            
        if 'hdr_envmap_visualize' in render_results:
            hdr_envmap_path = os.path.join(_video_output_dir(cfg, args, post_fix), "envmap.hdr")
            save_single_hdr(hdr_image=render_results['hdr_envmap_visualize'], save_pth=hdr_envmap_path)
        
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/full/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    
    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is not None:
        logger.info("Rendering novel views...")
        render_traj = dataset.get_novel_render_traj(
            traj_types=render_novel_cfg.traj_types,
            target_frames=render_novel_cfg.get("frames", dataset.frame_num),
        )
        video_output_dir = os.path.join(_video_output_dir(cfg, args, post_fix), f"novel_{step}")
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        
        for traj_type, traj in render_traj.items():
            # Prepare rendering data
            render_data = dataset.prepare_novel_view_render_data(traj)
            
            # Render and save video
            save_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
            render_novel_views(
                trainer, render_data, save_path,
                fps=render_novel_cfg.get("fps", cfg.render.fps),
                isosurface_render=args.isosurface_render,
            )
            logger.info(f"Saved novel view video for trajectory type: {traj_type} to {save_path}")
            
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    saved_config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(saved_config_path):
        raise FileNotFoundError(
            f"Eval requires the saved checkpoint config at {saved_config_path}. "
            "Pass --resume_from pointing to a training run checkpoint directory that contains config.yaml."
        )
    cfg = resolve_saved_config(
        saved_config_path,
        opts=args.opts,
        overlays=args.config_overlay,
    )
    args.enable_wandb = False
    os.makedirs(_video_output_dir(cfg, args, args.postfix), exist_ok=True)
    if args.log_metrics:
        os.makedirs(_metrics_output_dir(cfg, args, args.postfix), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import pdb; pdb.set_trace()
    
    # build dataset
    if args.dataset_source is not None:
        if cfg.data.dataset != "self":
            raise ValueError("--dataset_source is only supported for self dataset.")
        if "external_source" not in cfg.data.pixel_source:
            cfg.data.pixel_source.external_source = OmegaConf.create({})
        cfg.data.pixel_source.external_source.active_source = args.dataset_source
        logger.info("Using self dataset source for rendering/eval: %s", args.dataset_source)
    dataset = DrivingDataset(data_cfg=cfg.data)
    _setup_default_point_light_reanchor(cfg, dataset)

    model_config = cfg.model
    
    # setup trainer
    # see base.py
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=model_config,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
        use_pbr=(not args.no_pbr),
    )
    insert_fallback_model_config = OmegaConf.create(
        OmegaConf.to_container(trainer.model_config, resolve=True)
    )

    if args.eval_headlight:
        trainer.tracer_cfg.eval_headlight_mode = True
        logger.info("Enabled eval headlight mode (two camera-relative point lights).")
    
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

    if args.insert_rigid_ply is not None:
        if not os.path.exists(args.insert_rigid_ply):
            raise FileNotFoundError(f"insert_rigid_ply path not found: {args.insert_rigid_ply}")
        insert_info = insert_rigid_object_for_eval(
            trainer=trainer,
            dataset=dataset,
            ply_path=args.insert_rigid_ply,
            anchor_timestep=args.insert_anchor_timestep,
            anchor_cam_id=args.insert_cam_id,
            forward_m=args.insert_forward_m,
            right_m=args.insert_right_m,
            up_m=args.insert_up_m,
            yaw_deg=args.insert_yaw_deg,
            pitch_up_deg=args.insert_pitch_up_deg,
            roll_left_deg=args.insert_roll_left_deg,
            scale_mode=args.insert_scale_mode,
            manual_scale=args.insert_manual_scale,
            visible_first_frame_only=args.insert_only_first_frame,
        )
        logger.info(
            f"Inserted rigid object from {args.insert_rigid_ply}; "
            f"instance_id={insert_info['inserted_instance_id']}, "
            f"points={insert_info['inserted_points']}"
        )
        args.inserted_instance_id = insert_info["inserted_instance_id"]

    _apply_instance_deletions(
        trainer=trainer,
        delete_spec={
            "RigidNodes": args.delete_rigid_ids,
            "SMPLNodes": args.delete_smpl_ids,
            "DeformableNodes": args.delete_deformable_ids,
        },
    )

    inserted_dynamic_classes = args.insert_dynamic_classes

    if args.insert_dynamic_specs_json is not None:
        specs = _load_insert_dynamic_specs(args.insert_dynamic_specs_json, args)
        inserted_class_names: List[str] = []
        for spec in specs:
            insert_info = insert_dynamic_asset_for_eval(
                trainer=trainer,
                dataset=dataset,
                asset_path=str(spec["asset_path"]),
                classes=str(spec["classes"]),
                anchor_timestep=int(spec["anchor_timestep"]),
                anchor_cam_id=int(spec["anchor_cam_id"]),
                source_anchor_timestep=spec["source_anchor_timestep"],
                forward_m=float(spec["forward_m"]),
                right_m=float(spec["right_m"]),
                up_m=float(spec["up_m"]),
                yaw_deg=float(spec["yaw_deg"]),
                pitch_up_deg=float(spec["pitch_up_deg"]),
                roll_left_deg=float(spec["roll_left_deg"]),
                scale_mode=str(spec["scale_mode"]),
                manual_scale=float(spec["manual_scale"]),
                disable_place_search=bool(spec["disable_place_search"]),
                search_radius_m=float(spec["search_radius_m"]),
                search_step_m=float(spec["search_step_m"]),
                placement_mode=str(spec["placement_mode"]),
                target_world_xyz=_parse_optional_world_xyz(spec["target_world_xyz"]),
                target_world_timestep=spec["target_world_timestep"],
                target_world_yaw_deg=spec["target_world_yaw_deg"],
                fallback_model_config=insert_fallback_model_config,
            )
            inserted_class_names.extend(insert_info["classes"])
            logger.info(
                "Inserted dynamic asset spec from %s | classes=%s | scale=%.6f | placement=%s | stats=%s",
                insert_info["asset_path"],
                insert_info["classes"],
                insert_info["applied_scale"],
                insert_info["placement_mode"],
                insert_info["stats"],
            )
        inserted_dynamic_classes = ",".join(sorted(set(inserted_class_names)))
    elif args.insert_dynamic_asset is not None:
        insert_info = insert_dynamic_asset_for_eval(
            trainer=trainer,
            dataset=dataset,
            asset_path=args.insert_dynamic_asset,
            classes=args.insert_dynamic_classes,
            anchor_timestep=args.insert_anchor_timestep,
            anchor_cam_id=args.insert_cam_id,
            source_anchor_timestep=args.insert_dynamic_source_anchor_timestep,
            forward_m=args.insert_forward_m,
            right_m=args.insert_right_m,
            up_m=args.insert_up_m,
            yaw_deg=args.insert_yaw_deg,
            pitch_up_deg=args.insert_pitch_up_deg,
            roll_left_deg=args.insert_roll_left_deg,
            scale_mode=args.insert_scale_mode,
            manual_scale=args.insert_manual_scale,
            disable_place_search=args.insert_dynamic_disable_place_search,
            search_radius_m=args.insert_dynamic_search_radius_m,
            search_step_m=args.insert_dynamic_search_step_m,
            placement_mode=args.insert_dynamic_placement_mode,
            target_world_xyz=_parse_optional_world_xyz(args.insert_target_world_xyz),
            target_world_timestep=args.insert_target_world_timestep,
            target_world_yaw_deg=args.insert_target_world_yaw_deg,
            fallback_model_config=insert_fallback_model_config,
        )
        logger.info(
            "Inserted dynamic asset for eval-only rendering from %s | classes=%s | scale=%.6f | placement=%s | stats=%s",
            insert_info["asset_path"],
            insert_info["classes"],
            insert_info["applied_scale"],
            insert_info["placement_mode"],
            insert_info["stats"],
        )
        inserted_dynamic_classes = ",".join(insert_info["classes"])

    if args.move_rigid_specs_json is not None and args.move_rigid_ids is not None:
        raise ValueError("Use either --move_rigid_specs_json or --move_rigid_ids, not both.")
    if args.move_rigid_specs_json is not None:
        for spec in _load_rigid_move_specs(args.move_rigid_specs_json):
            _apply_rigid_instance_camera_relative_move(
                trainer=trainer,
                dataset=dataset,
                raw_ids=spec["rigid_ids"],
                anchor_timestep=int(spec["anchor_timestep"]),
                anchor_cam_id=int(spec["anchor_cam_id"]),
                forward_m=float(spec["forward_m"]),
                right_m=float(spec["right_m"]),
                up_m=float(spec["up_m"]),
                yaw_deg=float(spec["yaw_deg"]),
                scale=float(spec["scale"]),
            )
    elif args.move_rigid_ids is not None and args.move_anchor_timestep is not None:
        _apply_rigid_instance_camera_relative_move(
            trainer=trainer,
            dataset=dataset,
            raw_ids=args.move_rigid_ids,
            anchor_timestep=args.move_anchor_timestep,
            anchor_cam_id=args.move_anchor_cam_id,
            forward_m=args.move_forward_m,
            right_m=args.move_right_m,
            up_m=args.move_up_m,
            yaw_deg=args.move_yaw_deg,
            scale=args.move_scale,
        )
    else:
        _apply_rigid_instance_translation(
            trainer=trainer,
            raw_ids=args.move_rigid_ids,
            offset_xyz=[args.move_rigid_x_m, args.move_rigid_y_m, args.move_rigid_z_m],
        )
    _apply_gaussian_material_overrides(
        trainer=trainer,
        metallic=args.override_gaussian_metallic,
        roughness=args.override_gaussian_roughness,
    )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # load the envmap if specified
    # TODO: test loading envmap, and create new envmap
    if args.new_envmap_path is not None:
        if not os.path.exists(args.new_envmap_path):
            raise FileNotFoundError(f"Envmap path {args.new_envmap_path} does not exist.")
        if args.new_envmap_pbr_only:
            sky_params = cfg.model.Sky.params
            trainer.models["SkyPBR"] = EnvLight_EQ(
                class_name="SkyPBR",
                resolution=args.new_envmap_path_res,
                device=device,
                activation_name=sky_params.activation_name,
                min_res=sky_params.min_res,
                max_res=sky_params.max_res,
                min_roughness=sky_params.min_roughness,
                max_roughness=sky_params.max_roughness,
                init_value=sky_params.init_value,
                prior_path=sky_params.prior_path,
                path=args.new_envmap_path,
            )
            if trainer.models["SkyPBR"].forward_mode != "pure_env":
                trainer.models["SkyPBR"].build_mips()
            trainer.models["SkyPBR"].update_pdf()
            logger.info(
                f"Using new envmap for PBR only from {args.new_envmap_path}; "
                f"sky rendering still uses checkpoint Sky"
            )
        else:
            trainer.load_envmap(envmap_path=args.new_envmap_path, prior_path=cfg.model.Sky.params.prior_path, resolution=args.new_envmap_path_res)
            logger.info(f"Using new envmap from {args.new_envmap_path}")
    
    if args.calibrate_envmap:
        logger.info("Calibrating envmap, overriding the stored transform (if any)")
        calib_dataset = dataset
        if cfg.data.dataset == "self" and args.calibrate_envmap_source is not None:
            if "external_source" not in cfg.data.pixel_source:
                cfg.data.pixel_source.external_source = OmegaConf.create({})
            original_source = cfg.data.pixel_source.external_source.get("active_source", "primary")
            cfg.data.pixel_source.external_source.active_source = args.calibrate_envmap_source
            logger.info("Using self dataset source for envmap calibration: %s", args.calibrate_envmap_source)
            calib_dataset = DrivingDataset(data_cfg=cfg.data)
            cfg.data.pixel_source.external_source.active_source = original_source
        calib_num_cams = calib_dataset.pixel_source.num_cams
        calib_num_timesteps = calib_dataset.num_img_timesteps
        calib_timestep = int(args.calibrate_envmap_timestep)
        calib_cam_id = int(args.calibrate_envmap_cam_id)
        if calib_timestep < 0 or calib_timestep >= calib_num_timesteps:
            raise ValueError(
                f"--calibrate_envmap_timestep={calib_timestep} is out of range "
                f"[0, {calib_num_timesteps - 1}]."
            )
        if calib_cam_id < 0 or calib_cam_id >= calib_num_cams:
            raise ValueError(
                f"--calibrate_envmap_cam_id={calib_cam_id} is out of range "
                f"[0, {calib_num_cams - 1}]."
            )
        calib_idx = calib_timestep * calib_num_cams + calib_cam_id
        _, ff_cam_infos = calib_dataset.full_image_set.get_image(
            idx=calib_idx,
            camera_downscale=1.0,
        )
        c2w = ff_cam_infos["camera_to_world"]
        camera_dir = c2w[:3, 2]
        camera_dir = camera_dir / (camera_dir.norm() + 1e-8)  # normalize
        logger.info(
            "Calibrating envmap with timestep=%d, compact_cam_id=%d, full_index=%d",
            calib_timestep,
            calib_cam_id,
            calib_idx,
        )
        trainer.calibrate_envmap(camera_dir=camera_dir)

    if (
        abs(args.cam_forward_m) > 1e-12
        or abs(args.cam_left_m) > 1e-12
        or abs(args.cam_up_m) > 1e-12
        or abs(args.cam_yaw_left_deg) > 1e-12
        or abs(args.cam_pitch_up_deg) > 1e-12
        or abs(args.cam_roll_left_deg) > 1e-12
    ):
        offset_mod = _FixedCameraOffset(
            forward_m=args.cam_forward_m,
            left_m=args.cam_left_m,
            up_m=args.cam_up_m,
            yaw_left_deg=args.cam_yaw_left_deg,
            pitch_up_deg=args.cam_pitch_up_deg,
            roll_left_deg=args.cam_roll_left_deg,
        ).to(device)
        if "CamPosePerturb" in trainer.models:
            trainer.models["CamPosePerturb"] = _ComposedCamPosePerturb(
                base_module=trainer.models["CamPosePerturb"],
                offset_module=offset_mod,
            ).to(device)
            logger.info(
                f"Enabled camera offset/rotation on top of existing CamPosePerturb: "
                f"forward={args.cam_forward_m:.3f}m, left={args.cam_left_m:.3f}m, up={args.cam_up_m:.3f}m, "
                f"yaw_left={args.cam_yaw_left_deg:.3f}deg, pitch_up={args.cam_pitch_up_deg:.3f}deg, roll_left={args.cam_roll_left_deg:.3f}deg"
            )
        else:
            trainer.models["CamPosePerturb"] = offset_mod
            logger.info(
                f"Enabled camera offset/rotation via CamPosePerturb: "
                f"forward={args.cam_forward_m:.3f}m, left={args.cam_left_m:.3f}m, up={args.cam_up_m:.3f}m, "
                f"yaw_left={args.cam_yaw_left_deg:.3f}deg, pitch_up={args.cam_pitch_up_deg:.3f}deg, roll_left={args.cam_roll_left_deg:.3f}deg"
            )

    if args.envmap_rotation is not None:
        trainer.set_envmap_rotation(args.envmap_rotation, rotate_vertical=args.envmap_rotation_vertical)
        logger.info(f"Applied initial envmap rotation: {args.envmap_rotation} degrees (vertical={args.envmap_rotation_vertical})")
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "rgb_sky",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        "normalized_depths",
        "normals",
        "world_normals",
        "depth_normals",
        "albedos",
        "full_albedos",
        "full_roughnesses",
        "opacities",
        "roughnesses",
        "metallics",
        "normals_minscale",
        # "distortion_maps",
        # "raytrace_rgbs",
        # "raytrace_depths",
        # "raytrace_opacities",
        # "raytrace_normals",
        "raytrace_visibilities",
        "raytrace_visibilities_refine",
        # "raytrace_ind_lights",
        "raytrace_valid_counts",
        "pbr_colors",
        # "pbr_color_fulls",
        "pbr_diffuses",
        "pbr_speculars",
        "pbr_transports",
        # "pbr_dir_lights",
        # "pbr_n_d_is",
        # "pbr_lights",
        "Background_depths",
        "RigidNodes_depths",
        "DeformableNodes_depths",
        "SMPLNodes_depths",
        "Background_opacities",
        "RigidNodes_opacities",
        "DeformableNodes_opacities",
        "SMPLNodes_opacities",
        # "mask"
    ]
    if args.render_inserted_only:
        if args.insert_dynamic_specs_json is not None or args.insert_dynamic_asset is not None:
            render_keys = _extend_unique(
                render_keys,
                _build_inserted_only_render_keys(inserted_dynamic_classes),
            )
        elif args.insert_rigid_ply is not None:
            render_keys = _extend_unique(
                render_keys,
                ["RigidNodes_rgbs", "RigidNodes_depths", "RigidNodes_opacities"],
            )
        else:
            raise ValueError("--render_inserted_only requires --insert_dynamic_asset or --insert_rigid_ply.")
    synthetic_camera_path = int(args.camera_interp_steps or 0) > 0 or args.spiral_timestep is not None
    if cfg.render.vis_lidar and not synthetic_camera_path:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    
    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix=args.postfix,
        log_metrics=args.log_metrics,
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--video_output_dir", type=str, default=None, help="exact directory for rendered videos")
    parser.add_argument("--metrics_output_dir", type=str, default=None, help="exact directory for eval metric JSONs")
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    parser.add_argument("--postfix", type=str, default="_eval", help="an optional postfix for video")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    parser.add_argument("--enable_timing", action="store_true", help="enable per-op timing logs")
    parser.add_argument("--measure_fps", action="store_true", help="measure and report rendering FPS")
        
    # misc
    parser.add_argument(
        "--config_overlay",
        action="append",
        default=[],
        help=(
            "optional overlay config; can be repeated. Merge order is "
            "checkpoint config -> overlays -> CLI opts"
        ),
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    # optional
    parser.add_argument("--new_envmap_path", type=str, default=None, help="path to new envmap to use for rendering")
    parser.add_argument("--new_envmap_path_res", type=int, default=1024, help="spatial resolution of the new envmap to use for rendering (if applicable)")
    parser.add_argument("--new_envmap_pbr_only", action="store_true", help="use --new_envmap_path only for PBR lighting; keep sky rendering from checkpoint")
    parser.add_argument("--envmap_rotation", type=float, default=None, help="initial envmap rotation angle in degrees before rendering")
    parser.add_argument("--envmap_rotation_vertical", action="store_true", help="rotate the envmap vertically instead of horizontally")
    parser.add_argument("--calibrate_envmap", action="store_true", help="calibrate the envmap by the camera direction of the first frame")
    parser.add_argument("--dataset_source", type=str, default=None, choices=["primary", "external"], help="self dataset source used for rendering/eval")
    parser.add_argument("--calibrate_envmap_source", type=str, default='primary', choices=["primary", "external"], help="self dataset source used only for envmap calibration")
    parser.add_argument("--calibrate_envmap_timestep", type=int, default=0, help="timestep used as envmap calibration anchor")
    parser.add_argument("--calibrate_envmap_cam_id", type=int, default=0, help="compact camera id used as envmap calibration anchor")
    parser.add_argument("--eval_start_timestep", type=int, default=None, help="start timestep for rendering, default to the training setting")
    parser.add_argument("--eval_num_timesteps", type=int, default=None, help="number of timesteps to render, default to the training setting")
    parser.add_argument("--eval_cam_ids", type=str, default=None, help="space- or comma-separated compact camera ids to render; default renders all checkpoint cameras")
    parser.add_argument("--camera_interp_steps", type=int, default=0, help="number of interpolated camera poses to insert between consecutive rendered timesteps")
    parser.add_argument("--spiral_timestep", type=int, default=None, help="fixed timestep used as the spiral render anchor")
    parser.add_argument("--spiral_cam_id", type=int, default=0, help="compact camera id used as the spiral render anchor")
    parser.add_argument("--spiral_frames", type=int, default=120, help="number of frames in spiral render mode")
    parser.add_argument("--spiral_loops", type=float, default=1.0, help="number of loops in spiral render mode")
    parser.add_argument("--spiral_radius_m", type=float, default=1.0, help="horizontal spiral radius in meters")
    parser.add_argument("--spiral_vertical_amplitude_m", type=float, default=0.3, help="vertical spiral amplitude in meters")
    parser.add_argument("--spiral_target_distance_m", type=float, default=10.0, help="look-at target distance in front of the anchor camera")
    parser.add_argument("--spiral_fps", type=int, default=None, help="optional FPS override for spiral render mode")
    parser.add_argument("--isosurface_render", action="store_true", help="render using isosurface rendering")
    parser.add_argument("--no_pbr", action="store_true", help="whether to use pbr rendering")
    parser.add_argument("--log_metrics", action="store_true", help="whether to log metrics to wandb and save to json")
    parser.add_argument("--eval_headlight", action="store_true", help="enable eval-only two-point headlight mode")
    parser.add_argument("--insert_rigid_ply", type=str, default=None, help="optional rigid canonical PLY to insert at eval-time")
    parser.add_argument("--insert_dynamic_asset", type=str, default=None, help="optional dynamic asset .pth to insert at eval-time only (no checkpoint write)")
    parser.add_argument("--insert_dynamic_specs_json", type=str, default=None, help="JSON file describing multiple dynamic asset insertions with per-asset transforms")
    parser.add_argument("--insert_specs_simple", action="store_true", help="interpret --insert_dynamic_specs_json as simplified camera-relative insertion specs")
    parser.add_argument("--insert_dynamic_classes", type=str, default="RigidNodes,SMPLNodes", help="comma-separated classes to import from --insert_dynamic_asset")
    parser.add_argument("--insert_dynamic_source_anchor_timestep", type=int, default=None, help="anchor timestep in source asset (defaults to --insert_anchor_timestep)")
    parser.add_argument("--insert_dynamic_disable_place_search", action="store_true", help="disable collision-aware local search for dynamic insertion placement")
    parser.add_argument("--insert_dynamic_search_radius_m", type=float, default=8.0, help="lateral search radius in meters for dynamic insertion")
    parser.add_argument("--insert_dynamic_search_step_m", type=float, default=2.0, help="lateral search step in meters for dynamic insertion")
    parser.add_argument("--insert_dynamic_placement_mode", type=str, default="visible", choices=["visible", "clearance"], help="placement objective for dynamic insertion search")
    parser.add_argument("--insert_target_world_xyz", type=str, default=None, help="absolute world-space x,y,z target for inserted dynamic asset center at target timestep")
    parser.add_argument("--insert_target_world_timestep", type=int, default=None, help="target timestep where --insert_target_world_xyz should be enforced; defaults to --insert_anchor_timestep")
    parser.add_argument("--insert_target_world_yaw_deg", type=float, default=None, help="absolute world-up yaw angle in degrees for inserted dynamic asset; overrides camera-frame yaw/pitch/roll")
    parser.add_argument("--delete_rigid_ids", type=str, default=None, help="comma-separated rigid instance IDs to delete before rendering")
    parser.add_argument("--delete_smpl_ids", type=str, default=None, help="comma-separated SMPL instance IDs to delete before rendering")
    parser.add_argument("--delete_deformable_ids", type=str, default=None, help="comma-separated deformable instance IDs to delete before rendering")
    parser.add_argument("--move_rigid_ids", type=str, default=None, help="comma-separated RigidNodes instance IDs to translate at eval-time")
    parser.add_argument("--move_rigid_x_m", type=float, default=0.0, help="eval-only world-space X translation for --move_rigid_ids in meters")
    parser.add_argument("--move_rigid_y_m", type=float, default=0.0, help="eval-only world-space Y translation for --move_rigid_ids in meters")
    parser.add_argument("--move_rigid_z_m", type=float, default=0.0, help="eval-only world-space Z translation for --move_rigid_ids in meters")
    parser.add_argument("--move_rigid_specs_json", type=str, default=None, help="JSON file describing camera-relative RigidNodes moves")
    parser.add_argument("--move_anchor_timestep", type=int, default=None, help="anchor timestep for camera-relative rigid move")
    parser.add_argument("--move_anchor_cam_id", type=int, default=0, help="compact anchor camera id for camera-relative rigid move")
    parser.add_argument("--move_forward_m", type=float, default=0.0, help="camera-relative forward move in meters")
    parser.add_argument("--move_right_m", type=float, default=0.0, help="camera-relative right move in meters")
    parser.add_argument("--move_up_m", type=float, default=0.0, help="camera-relative upward move in meters")
    parser.add_argument("--move_yaw_deg", type=float, default=0.0, help="camera-relative yaw rotation in degrees")
    parser.add_argument("--move_scale", type=float, default=1.0, help="eval-only scale factor for selected rigid instances")
    parser.add_argument("--override_gaussian_metallic", type=float, default=None, help="eval-only override for every Gaussian metallic value in [0, 1]")
    parser.add_argument("--override_gaussian_roughness", type=float, default=None, help="eval-only override for every Gaussian roughness value in [0, 1]")
    parser.add_argument("--render_inserted_only", action="store_true", help="also output inserted object class layers in addition to the standard render outputs")
    parser.add_argument("--insert_only_first_frame", action="store_true", help="only show inserted object in frame 0 and render first timestep only")
    parser.add_argument("--insert_anchor_timestep", type=int, default=0, help="anchor timestep to compute placement from camera pose")
    parser.add_argument("--insert_cam_id", type=int, default=0, help="anchor camera id")
    parser.add_argument("--insert_forward_m", type=float, default=8.0, help="forward offset from anchor camera in meters")
    parser.add_argument("--insert_right_m", type=float, default=0.0, help="right offset from anchor camera in meters")
    parser.add_argument("--insert_up_m", type=float, default=0.0, help="up offset from anchor camera in meters")
    parser.add_argument("--insert_yaw_deg", type=float, default=0.0, help="yaw rotation (degrees) around anchor camera up axis")
    parser.add_argument("--insert_pitch_up_deg", type=float, default=0.0, help="pitch-up rotation (degrees) around anchor camera right axis")
    parser.add_argument("--insert_roll_left_deg", type=float, default=0.0, help="roll-left rotation (degrees) around anchor camera forward axis")
    parser.add_argument("--insert_scale_mode", type=str, choices=["ego", "manual"], default="ego", help="inserted object scale mode")
    parser.add_argument("--insert_manual_scale", type=float, default=1.0, help="manual scale factor when insert_scale_mode=manual")
    parser.add_argument("--insert_rotate_sweep", action="store_true", help="render fixed-frame rotation sweep video for inserted object")
    parser.add_argument("--insert_sweep_frame", type=int, default=0, help="fixed timestep index for insertion rotation sweep")
    parser.add_argument("--insert_sweep_cam_id", type=int, default=0, help="camera id for insertion rotation sweep")
    parser.add_argument("--insert_sweep_yaw_start", type=float, default=-180.0, help="start yaw angle (deg) for insertion sweep")
    parser.add_argument("--insert_sweep_yaw_end", type=float, default=180.0, help="end yaw angle (deg) for insertion sweep")
    parser.add_argument("--insert_sweep_num", type=int, default=73, help="number of yaw samples for insertion sweep")
    parser.add_argument("--insert_sweep_fps", type=int, default=12, help="fps for insertion sweep video")
    parser.add_argument("--cam_forward_m", type=float, default=0.0, help="eval-only camera translation forward in meters")
    parser.add_argument("--cam_left_m", type=float, default=0.0, help="eval-only camera translation left in meters")
    parser.add_argument("--cam_up_m", type=float, default=0.0, help="eval-only camera translation up in meters")
    parser.add_argument("--cam_yaw_left_deg", type=float, default=0.0, help="eval-only camera yaw to left in degrees")
    parser.add_argument("--cam_pitch_up_deg", type=float, default=0.0, help="eval-only camera pitch up in degrees")
    parser.add_argument("--cam_roll_left_deg", type=float, default=0.0, help="eval-only camera roll left in degrees")
    
    args = parser.parse_args()

    main(args)
