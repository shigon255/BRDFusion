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
from utils.misc import import_str
from utils.config import resolve_config
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
    insert_dynamic_asset_for_eval,
    insert_rigid_object_for_eval,
    set_inserted_rigid_rotation_for_eval,
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


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


def _parse_instance_ids_csv(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    raw = raw.strip()
    if raw == "":
        return []
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
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

        classes = spec.get("classes", None)
        if classes is None or str(classes).strip() == "":
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
    out_dir = os.path.join(cfg.log_dir, f"videos{post_fix}", f"insert_sweep_frame{frame_idx:03d}_cam{cam_id}")
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
            test_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
            )
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
        
        start_timestep = 0
        if args.eval_start_timestep is not None:
            start_timestep = min(args.eval_start_timestep, num_total_timesteps - 1)
        
        num_eval_timesteps = num_total_timesteps - start_timestep
        if args.eval_num_timesteps is not None:
            num_eval_timesteps = min(args.eval_num_timesteps, num_total_timesteps - start_timestep)
        print(f"Rendering {num_eval_timesteps} timesteps from {start_timestep} to {start_timestep + num_eval_timesteps - 1}, each with {num_cams} cameras, total {num_eval_timesteps * num_cams} images.")
        
        vis_indices = []
        for timestep in range(start_timestep, start_timestep + num_eval_timesteps):
            for cam_id in range(num_cams):
                vis_indices.append(timestep * num_cams + cam_id)
        
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            tmp_path=tmp_path,
            vis_indices=vis_indices,
            isosurface_render=args.isosurface_render,
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
            full_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            # num_timestamps=dataset.num_img_timesteps,
            num_timestamps=num_eval_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=cfg.render.fps,
            verbose=True,
        )
        if "point_light_emitters_rgbs" in render_results and len(render_results["point_light_emitters_rgbs"]) > 0:
            emitter_only_path = video_output_pth.replace(".mp4", "_emitters_only.mp4")
            _ = save_videos(
                render_results,
                emitter_only_path,
                layout=dataset.layout,
                num_timestamps=num_eval_timesteps,
                keys=["point_light_emitters_rgbs"],
                num_cams=dataset.pixel_source.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
            )
            logger.info(f"Emitter-only video saved to {emitter_only_path}")
        
        if 'envmap_visualize' in render_results:
            # save envmap
            envmap_path = os.path.join(
                cfg.log_dir, "images", f"{cfg.log_dir}/videos{post_fix}/envmap.png"
            )
            save_single_image(image=render_results['envmap_visualize'], save_pth=envmap_path)
            
        if 'hdr_envmap_visualize' in render_results:
            hdr_envmap_path = os.path.join(
                cfg.log_dir, "images", f"{cfg.log_dir}/videos{post_fix}/envmap.hdr"
            )
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
        video_output_dir = f"{cfg.log_dir}/videos{post_fix}/novel_{step}"
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
    OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
    cfg = resolve_config(
        os.path.join(log_dir, "config.yaml"),
        opts=args.opts,
        overlays=args.config_overlay,
    )
    args.enable_wandb = False
    for folder in [f"videos{args.postfix}", f"metrics{args.postfix}"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
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
        _, ff_cam_infos = calib_dataset.full_image_set.get_image(
            idx=0,
            camera_downscale=1.0,
        )
        c2w = ff_cam_infos["camera_to_world"]
        camera_dir = c2w[:3, 2]
        camera_dir = camera_dir / (camera_dir.norm() + 1e-8)  # normalize
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
    if cfg.render.vis_lidar:
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
    parser.add_argument("--eval_start_timestep", type=int, default=None, help="start timestep for rendering, default to the training setting")
    parser.add_argument("--eval_num_timesteps", type=int, default=None, help="number of timesteps to render, default to the training setting")
    parser.add_argument("--isosurface_render", action="store_true", help="render using isosurface rendering")
    parser.add_argument("--no_pbr", action="store_true", help="whether to use pbr rendering")
    parser.add_argument("--log_metrics", action="store_true", help="whether to log metrics to wandb and save to json")
    parser.add_argument("--eval_headlight", action="store_true", help="enable eval-only two-point headlight mode")
    parser.add_argument("--insert_rigid_ply", type=str, default=None, help="optional rigid canonical PLY to insert at eval-time")
    parser.add_argument("--insert_dynamic_asset", type=str, default=None, help="optional dynamic asset .pth to insert at eval-time only (no checkpoint write)")
    parser.add_argument("--insert_dynamic_specs_json", type=str, default=None, help="JSON file describing multiple dynamic asset insertions with per-asset transforms")
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
