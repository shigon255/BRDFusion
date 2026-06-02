#!/usr/bin/env python3
"""Merge DiffusionLight per-view HDR probes into one scene envmap.

The input HDR probes are produced by the vendored DiffusionLight wrappers under
`third_party/DiffusionLight-Turbo/outputs/.../hdr_seed*/`. Each probe is
estimated in the source camera frame. This tool uses BRDFusion dataset
calibrations to rotate every probe into a common anchor-camera frame before
merging.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyexr
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omegaconf import OmegaConf
from tqdm import tqdm

from datasets.driving_dataset import DrivingDataset


def warp_envmap_equirect(env: torch.Tensor, rot: torch.Tensor, out_hw=None) -> torch.Tensor:
    assert env.ndim == 3 and env.shape[2] in (3, 4), "env must be (H, W, 3|4)"
    assert rot.shape == (3, 3), "rot must be 3x3"

    device = env.device
    dtype = env.dtype
    h, w, _ = env.shape
    h_out, w_out = out_hw or (h, w)

    u = torch.linspace(0.0, 1.0, w_out, device=device, dtype=dtype)
    v = torch.linspace(0.0, 1.0, h_out, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    phi = (uu - 0.5) * (2.0 * torch.pi)
    theta = vv * torch.pi
    x = torch.sin(theta) * torch.sin(phi)
    y = torch.cos(theta)
    z = -torch.sin(theta) * torch.cos(phi)
    target_dirs = torch.stack([x, y, z], dim=-1)
    src_dirs = target_dirs @ rot.T

    u_src = torch.atan2(src_dirs[..., 0], -src_dirs[..., 2]).nan_to_num() / (2.0 * torch.pi) + 0.5
    v_src = torch.acos(src_dirs[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi

    x_src = u_src * (w - 1)
    y_src = v_src * (h - 1)
    x0 = torch.floor(x_src).long()
    y0 = torch.floor(y_src).long()
    x1 = (x0 + 1) % w
    y1 = torch.clamp(y0 + 1, 0, h - 1)
    wx = (x_src - x0.to(dtype)).unsqueeze(-1)
    wy = (y_src - y0.to(dtype)).unsqueeze(-1)

    def gather(ix, iy):
        return env[iy.transpose(0, 1), ix.transpose(0, 1)]

    c00 = gather(x0, y0)
    c10 = gather(x1, y0)
    c01 = gather(x0, y1)
    c11 = gather(x1, y1)
    c0 = c00 * (1 - wx.transpose(0, 1)) + c10 * wx.transpose(0, 1)
    c1 = c01 * (1 - wx.transpose(0, 1)) + c11 * wx.transpose(0, 1)
    return (c0 * (1 - wy.transpose(0, 1)) + c1 * wy.transpose(0, 1)).transpose(0, 1)


def get_rot(src_dir: torch.Tensor, tgt_dir: torch.Tensor) -> torch.Tensor:
    v = torch.cross(src_dir, tgt_dir, dim=0)
    s = v.norm()
    c = torch.dot(src_dir, tgt_dir)
    if s < 1e-6:
        if c > 0:
            return torch.eye(3, device=src_dir.device, dtype=src_dir.dtype)
        axis = torch.tensor([1.0, 0.0, 0.0], device=src_dir.device, dtype=src_dir.dtype)
        if torch.allclose(src_dir, axis):
            axis = torch.tensor([0.0, 1.0, 0.0], device=src_dir.device, dtype=src_dir.dtype)
        v = torch.cross(src_dir, axis, dim=0)
        v = v / (v.norm() + 1e-8)
        k = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=src_dir.device, dtype=src_dir.dtype)
        return torch.eye(3, device=src_dir.device, dtype=src_dir.dtype) + 2 * k @ k
    k = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=src_dir.device, dtype=src_dir.dtype)
    return torch.eye(3, device=src_dir.device, dtype=src_dir.dtype) + k + k @ k * ((1 - c) / (s**2 + 1e-8))


def hdr_to_display(hdr: np.ndarray) -> np.ndarray:
    return np.clip(np.power(np.clip(hdr, 0.0, None), 1.0 / 2.2), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_kind", choices=["waymo", "self"], required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--view_times", nargs="+", required=True)
    parser.add_argument("--cam_ids", type=int, nargs="+", required=True)
    parser.add_argument("--resize_methods", nargs="+", default=["crop"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 37, 71])
    parser.add_argument("--output_root", required=True, help="DiffusionLight scene output root containing hdr_seed*/")
    parser.add_argument("--mode", choices=["average", "median", "mean_in_inlier"], default="median")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input_image_root", default=None)
    parser.add_argument("--output_video", default=None)
    parser.add_argument("--frame_index_base", type=int, default=0)
    parser.add_argument("--anchor_time", type=int, default=0, help="Reference timestep for envmap orientation; original scripts use 0.")
    parser.add_argument("--anchor_cam_id", type=int, default=0, help="Reference camera id for envmap orientation; original scripts use camera 0.")
    parser.add_argument("--config_file", default="configs/omnire.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> OmegaConf:
    cfg = OmegaConf.load(args.config_file)
    cli = OmegaConf.from_cli(args.opts)
    if "dataset" in cli:
        cfg.dataset = cli.pop("dataset")
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{dataset_type}.yaml"))
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    cfg = OmegaConf.merge(cfg, cli)

    loaded_cameras = list(dict.fromkeys([args.anchor_cam_id, *args.cam_ids]))
    cfg.data.pixel_source.cameras = loaded_cameras
    first_downscale = 2
    if "downscale_when_loading" in cfg.data.pixel_source and len(cfg.data.pixel_source.downscale_when_loading) > 0:
        first_downscale = cfg.data.pixel_source.downscale_when_loading[0]
    cfg.data.pixel_source.downscale_when_loading = [first_downscale for _ in loaded_cameras]
    cfg.data.pixel_source.load_only_calibrations = True
    cfg.data.pixel_source.load_images_only = False
    cfg.data.pixel_source.load_objects = False
    cfg.data.pixel_source.load_smpl = False
    cfg.data.pixel_source.load_dynamic_mask = False
    cfg.data.pixel_source.load_sky_mask = False
    cfg.data.pixel_source.load_shadow_mask = False
    cfg.data.pixel_source.load_road_mask = False
    cfg.data.pixel_source.load_shading_map = False
    cfg.data.pixel_source.load_priors = False
    cfg.data.pixel_source.load_materials = False
    cfg.data.pixel_source.load_gt_depth = False
    cfg.data.lidar_source.load_lidar = False
    return cfg


def build_dataset(args: argparse.Namespace) -> DrivingDataset:
    OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)
    cfg = load_config(args)
    return DrivingDataset(data_cfg=cfg.data)


def camera_dir(dataset: DrivingDataset, time_idx: int, cam_id: int, device: torch.device) -> torch.Tensor:
    if cam_id not in dataset.pixel_source.camera_data:
        raise ValueError(
            f"Camera {cam_id} was not loaded. Loaded cameras: {sorted(dataset.pixel_source.camera_data)}. "
            "Camera 0 must exist because the default envmap anchor is frame 0, camera 0."
        )
    camera = dataset.pixel_source.camera_data[cam_id]
    if time_idx < 0 or time_idx >= camera.cam_to_worlds.shape[0]:
        raise ValueError(f"time_idx={time_idx} out of range for camera {cam_id} with {camera.cam_to_worlds.shape[0]} frames")
    c2w = camera.cam_to_worlds[time_idx].to(device)
    direction = c2w[:3, 2]
    return direction / (direction.norm() + 1e-8)


def parse_view_time(dataset_kind: str, view_time: str, cam_name_to_id: Dict[str, int], cam_tag_to_id: Dict[str, int], frame_index_base: int) -> Tuple[int, int]:
    if dataset_kind == "waymo":
        frame_str, cam_str = view_time.split("_")
        return int(frame_str), int(cam_str)
    cam_tag, frame_str = view_time.rsplit("_", 1)
    cam_id = cam_tag_to_id.get(cam_tag)
    if cam_id is None:
        cam_id = cam_name_to_id.get(cam_tag.replace("_", "."))
    if cam_id is None:
        raise ValueError(f"Unknown self camera tag in view_time: {view_time}")
    return int(frame_str) - frame_index_base, cam_id


def envmap_file(output_root: str, scene: str, view_time: str, resize_method: str, seed: int) -> str:
    return os.path.join(output_root, f"hdr_seed{seed}", f"{scene}_{view_time}_{resize_method}.exr")


def merge_envmaps(envmaps: List[np.ndarray], mode: str) -> np.ndarray:
    if not envmaps:
        raise ValueError("No envmaps were loaded; check view_times/seeds/output_root.")
    stack = np.stack(envmaps, axis=0)
    if mode == "average":
        return np.mean(stack, axis=0)
    if mode == "median":
        return np.median(stack, axis=0)
    if mode == "mean_in_inlier":
        median = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - median), axis=0) * 1.4826
        z_scores = np.abs(stack - median) / (mad + 1e-6)
        mask = (z_scores < 1.5).astype(np.float32)
        sum_inliers = np.sum(stack * mask, axis=0)
        count_inliers = np.sum(mask, axis=0)
        return np.where(count_inliers > 0, sum_inliers / np.maximum(count_inliers, 1), median)
    raise ValueError(f"Unknown merge mode: {mode}")


def input_image_path(args: argparse.Namespace, view_time: str, cam_id_to_tag: Dict[int, str]) -> str | None:
    if not args.input_image_root:
        return None
    if args.dataset_kind == "waymo":
        return os.path.join(args.input_image_root, f"{view_time}.jpg")
    time_idx, cam_id = parse_view_time(args.dataset_kind, view_time, {}, {v: k for k, v in cam_id_to_tag.items()}, args.frame_index_base)
    frame_id = time_idx + args.frame_index_base
    return os.path.join(args.input_image_root, f"{frame_id:03d}_{cam_id}.png")


def write_video(viz_data: Dict[int, Dict[int, np.ndarray]], args: argparse.Namespace, cam_id_to_tag: Dict[int, str]) -> None:
    if not args.output_video:
        return
    if not viz_data:
        print("No visualization data; skipping video.")
        return
    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    sorted_times = sorted(viz_data)
    ordered_cams = [cam for cam in args.cam_ids if any(cam in per_cam for per_cam in viz_data.values())]
    writer = imageio.get_writer(str(output_video), mode="I", fps=5)
    for time_idx in tqdm(sorted_times, desc="Rendering merge visualization"):
        num_cols = max(len(ordered_cams), 1)
        fig = plt.figure(figsize=(4 * num_cols, 10), constrained_layout=True)
        subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1])
        titles = ["Tone-mapped Envmap", "Luminance", "Input Image"]
        axes_rows = []
        for row, title in enumerate(titles):
            axs = subfigs[row].subplots(1, num_cols)
            if num_cols == 1:
                axs = [axs]
            subfigs[row].suptitle(title, fontsize=14)
            axes_rows.append(axs)

        for col, cam_id in enumerate(ordered_cams):
            env = viz_data[time_idx].get(cam_id)
            for row in range(3):
                axes_rows[row][col].axis("off")
                axes_rows[row][col].set_title(f"Cam {cam_id}")
            if env is not None:
                axes_rows[0][col].imshow(hdr_to_display(env))
                lum = 0.2126 * env[..., 0] + 0.7152 * env[..., 1] + 0.0722 * env[..., 2]
                im = axes_rows[1][col].imshow(lum, cmap="inferno", aspect="auto")
                plt.colorbar(im, ax=axes_rows[1][col], fraction=0.046, pad=0.04)
            else:
                axes_rows[0][col].text(0.5, 0.5, "N/A", ha="center", va="center")

            if args.dataset_kind == "waymo":
                view_time = f"{time_idx:03d}_{cam_id}"
            else:
                view_time = f"{cam_id_to_tag[cam_id]}_{time_idx + args.frame_index_base:03d}"
            img_path = input_image_path(args, view_time, cam_id_to_tag)
            if img_path and os.path.exists(img_path):
                axes_rows[2][col].imshow(imageio.imread(img_path))
            else:
                axes_rows[2][col].text(0.5, 0.5, "Image missing", ha="center", va="center")

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(canvas.get_renderer().buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        writer.append_data(frame)
        plt.close(fig)
    writer.close()
    print(f"Saved visualization video to {output_video}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args)
    to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=device)

    cam_id_to_name = {cam_id: cam.cam_name for cam_id, cam in dataset.pixel_source.camera_data.items()}
    cam_name_to_id = {name: cam_id for cam_id, name in cam_id_to_name.items()}
    cam_id_to_tag = {cam_id: name.replace(".", "_") for cam_id, name in cam_id_to_name.items()}
    cam_tag_to_id = {tag: cam_id for cam_id, tag in cam_id_to_tag.items()}

    anchor_dir = to_opengl @ camera_dir(dataset, args.anchor_time, args.anchor_cam_id, device)

    global_envmaps: List[np.ndarray] = []
    viz_data: Dict[int, Dict[int, np.ndarray]] = {}
    for view_time in tqdm(args.view_times, desc="Merging DiffusionLight envmaps"):
        time_idx, cam_id = parse_view_time(args.dataset_kind, view_time, cam_name_to_id, cam_tag_to_id, args.frame_index_base)
        src_dir = to_opengl @ camera_dir(dataset, time_idx, cam_id, device)
        rot = get_rot(src_dir, anchor_dir)
        current_view_envmaps = []
        for resize_method in args.resize_methods:
            for seed in args.seeds:
                path = envmap_file(args.output_root, args.scene, view_time, resize_method, seed)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Missing DiffusionLight envmap: {path}")
                envmap = pyexr.read(path)
                if envmap.dtype == np.uint8:
                    envmap = envmap.astype(np.float32) / 255.0
                envmap = envmap[..., :3].astype(np.float32)
                warped = warp_envmap_equirect(torch.from_numpy(envmap).to(device), rot).cpu().numpy()
                global_envmaps.append(warped)
                current_view_envmaps.append(warped)
        if current_view_envmaps:
            viz_data.setdefault(time_idx, {})[cam_id] = np.mean(np.stack(current_view_envmaps, axis=0), axis=0)

    merged = merge_envmaps(global_envmaps, args.mode).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pyexr.write(str(output), merged)
    print(f"Saved merged envmap to {output}")
    png_output = output.with_suffix(".png")
    plt.imsave(str(png_output), hdr_to_display(merged))
    print(f"Saved preview to {png_output}")
    write_video(viz_data, args, cam_id_to_tag)


if __name__ == "__main__":
    main()
