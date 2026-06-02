#!/usr/bin/env python3
"""Prepare per-camera envmaps for multi-camera DiffusionRenderer SDEdit."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyexr
import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from tools.rotate_envmap_between_sources import get_rot, warp_envmap_equirect


def read_hdr_envmap(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".hdr":
        env = iio.imread(path)
    elif ext == ".exr":
        env = pyexr.read(str(path))
    else:
        raise ValueError(f"Unsupported envmap format: {ext}")
    if env.dtype == np.uint8:
        env = env.astype(np.float32) / 255.0
    return env.astype(np.float32, copy=False)


def write_hdr_envmap(path: Path, env: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".hdr":
        iio.imwrite(path, env)
    elif ext == ".exr":
        pyexr.write(str(path), env)
    else:
        raise ValueError(f"Unsupported envmap format: {ext}")


def load_cfg(args: argparse.Namespace) -> OmegaConf:
    cfg = OmegaConf.load(args.config_file)
    cli = OmegaConf.from_cli(args.opts)
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if "dataset" in cli:
        cfg.dataset = cli.pop("dataset")
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{dataset_type}.yaml"))
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    return OmegaConf.merge(cfg, cli)


def parse_cam_ids(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        out.extend(int(item) for item in str(value).split() if item)
    return out or [0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--input", required=True, help="Camera-0/reference envmap path")
    parser.add_argument("--output_root", required=True, help="Directory that receives envmap_<cam>.<ext>")
    parser.add_argument("--cam_ids", nargs="+", default=["0"])
    parser.add_argument("--input_cam", type=int, default=0)
    parser.add_argument("--time_idx", type=int, default=0)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    cam_ids = parse_cam_ids(args.cam_ids)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrivingDataset(data_cfg=cfg.data)
    num_cams = dataset.pixel_source.num_cams
    if args.input_cam < 0 or args.input_cam >= num_cams:
        raise ValueError(f"input_cam={args.input_cam} out of range [0, {num_cams - 1}]")
    if args.time_idx < 0 or args.time_idx >= dataset.num_img_timesteps:
        raise ValueError(f"time_idx={args.time_idx} out of range [0, {dataset.num_img_timesteps - 1}]")

    env = read_hdr_envmap(input_path)
    env_t = torch.from_numpy(env).to(device)
    ext = input_path.suffix or ".hdr"

    input_c2w = dataset.pixel_source.camera_data[args.input_cam].cam_to_worlds[args.time_idx].to(device)
    input_dir = input_c2w[:3, 2]
    input_dir = input_dir / (input_dir.norm() + 1e-8)
    to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=device)
    input_dir_gl = to_opengl @ input_dir

    for cam_id in cam_ids:
        if cam_id < 0 or cam_id >= num_cams:
            raise ValueError(f"cam_id={cam_id} out of range [0, {num_cams - 1}]")
        output = output_root / f"envmap_{cam_id}{ext}"
        if cam_id == args.input_cam:
            if output.exists() or output.is_symlink():
                output.unlink()
            try:
                os.symlink(input_path, output)
            except OSError:
                shutil.copy2(input_path, output)
            continue

        target_c2w = dataset.pixel_source.camera_data[cam_id].cam_to_worlds[args.time_idx].to(device)
        target_dir = target_c2w[:3, 2]
        target_dir = target_dir / (target_dir.norm() + 1e-8)
        target_dir_gl = to_opengl @ target_dir
        rot = get_rot(input_dir_gl, target_dir_gl)
        warped = warp_envmap_equirect(env_t, rot).detach().cpu().numpy()
        write_hdr_envmap(output, warped)

    print(f"Prepared camera envmaps in {output_root}")


if __name__ == "__main__":
    main()
