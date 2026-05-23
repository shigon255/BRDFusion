import argparse
import json
import os

import imageio.v3 as iio
import numpy as np
import pyexr
import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset


def warp_envmap_equirect(env: torch.Tensor, rot: torch.Tensor, out_hw=None) -> torch.Tensor:
    assert env.ndim == 3 and env.shape[2] in (3, 4), "env must be (H, W, 3|4)"
    assert rot.shape == (3, 3), "rot must be (3, 3)"

    device = env.device
    dtype = env.dtype
    h, w, _ = env.shape
    if out_hw is None:
        h_out, w_out = h, w
    else:
        h_out, w_out = out_hw

    u = torch.linspace(0.0, 1.0, w_out, device=device, dtype=dtype)
    v = torch.linspace(0.0, 1.0, h_out, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    phi = (uu - 0.5) * (2.0 * torch.pi)
    theta = vv * torch.pi
    x = torch.sin(theta) * torch.sin(phi)
    y = torch.cos(theta)
    z = -torch.sin(theta) * torch.cos(phi)
    l_target = torch.stack([x, y, z], dim=-1)

    l_src = l_target @ rot.T

    u_src = torch.atan2(l_src[..., 0], -l_src[..., 2]).nan_to_num() / (2.0 * torch.pi) + 0.5
    v_src = torch.acos(l_src[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / torch.pi

    x_src = u_src * (w - 1)
    y_src = v_src * (h - 1)

    x0 = torch.floor(x_src).to(torch.long)
    y0 = torch.floor(y_src).to(torch.long)
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
    warped = c0 * (1 - wy.transpose(0, 1)) + c1 * wy.transpose(0, 1)
    return warped.transpose(0, 1)


def get_rot(src_dir: torch.Tensor, tgt_dir: torch.Tensor) -> torch.Tensor:
    src_dir = src_dir / (src_dir.norm() + 1e-8)
    tgt_dir = tgt_dir / (tgt_dir.norm() + 1e-8)
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
        k = torch.tensor(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            device=src_dir.device,
            dtype=src_dir.dtype,
        )
        return torch.eye(3, device=src_dir.device, dtype=src_dir.dtype) + 2 * k @ k

    k = torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        device=src_dir.device,
        dtype=src_dir.dtype,
    )
    return torch.eye(3, device=src_dir.device, dtype=src_dir.dtype) + k + k @ k * ((1 - c) / (s**2 + 1e-8))


def read_hdr_envmap(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hdr":
        env = iio.imread(path)
    elif ext == ".exr":
        env = pyexr.read(path)
    else:
        raise ValueError(f"Unsupported envmap format: {ext}")
    if env.dtype == np.uint8:
        env = env.astype(np.float32) / 255.0
    return env


def write_hdr_envmap(path: str, env: np.ndarray):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hdr":
        iio.imwrite(path, env)
    elif ext == ".exr":
        pyexr.write(path, env)
    else:
        raise ValueError(f"Unsupported envmap format: {ext}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rotate envmap from first camera direction of primary self source to external self source."
    )
    parser.add_argument("--config_file", required=True, help="Base config file (e.g., configs/omnire.yaml)")
    parser.add_argument("--input", "-i", required=True, help="Input envmap (.hdr/.exr)")
    parser.add_argument("--output", "-o", required=True, help="Output envmap (.hdr/.exr)")
    parser.add_argument("--time_idx", type=int, default=0, help="Frame index used for both sources")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera id used for both sources")
    parser.add_argument(
        "--save_meta_json",
        type=str,
        default=None,
        help="Optional path to save computed directions/rotation metadata as JSON",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="OmegaConf overrides")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config_file)
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{dataset_type}.yaml"))
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    cfg = OmegaConf.merge(cfg, args_from_cli)

    if cfg.data.dataset != "self":
        raise ValueError("This script currently supports only self dataset.")
    ext_cfg = cfg.data.pixel_source.get("external_source", None)
    if ext_cfg is None or not ext_cfg.get("enable", False):
        raise ValueError("Please set data.pixel_source.external_source.enable=true.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrivingDataset(data_cfg=cfg.data)

    if args.cam_id < 0 or args.cam_id >= dataset.pixel_source.num_cams:
        raise ValueError(f"cam_id={args.cam_id} out of range [0, {dataset.pixel_source.num_cams - 1}]")
    if args.time_idx < 0 or args.time_idx >= dataset.num_img_timesteps:
        raise ValueError(f"time_idx={args.time_idx} out of range [0, {dataset.num_img_timesteps - 1}]")

    cam = dataset.pixel_source.camera_data[args.cam_id]
    if not hasattr(cam, "external_camera") or cam.external_camera is None:
        raise ValueError("External camera is not attached. Check external_source config.")

    primary_c2w = cam.cam_to_worlds[args.time_idx].to(device)
    external_c2w = cam.external_camera.cam_to_worlds[args.time_idx].to(device)
    primary_dir = primary_c2w[:3, 2]
    external_dir = external_c2w[:3, 2]
    primary_dir = primary_dir / (primary_dir.norm() + 1e-8)
    external_dir = external_dir / (external_dir.norm() + 1e-8)

    # Keep same convention as existing rotate_hdr_envmap.py
    to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=device)
    primary_dir_gl = to_opengl @ primary_dir
    external_dir_gl = to_opengl @ external_dir
    rot = get_rot(primary_dir_gl, external_dir_gl)

    env = read_hdr_envmap(args.input)
    env_t = torch.from_numpy(env).to(device)
    warped = warp_envmap_equirect(env_t, rot).detach().cpu().numpy()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_hdr_envmap(args.output, warped)

    print("Primary dir:", primary_dir.detach().cpu().numpy())
    print("External dir:", external_dir.detach().cpu().numpy())
    print("Rotation (primary -> external):")
    print(rot.detach().cpu().numpy())
    print(f"Wrote rotated envmap to {args.output}")

    if args.save_meta_json is not None:
        meta = {
            "time_idx": int(args.time_idx),
            "cam_id": int(args.cam_id),
            "primary_dir": primary_dir.detach().cpu().numpy().tolist(),
            "external_dir": external_dir.detach().cpu().numpy().tolist(),
            "rotation_primary_to_external": rot.detach().cpu().numpy().tolist(),
            "input_envmap": args.input,
            "output_envmap": args.output,
        }
        os.makedirs(os.path.dirname(args.save_meta_json) or ".", exist_ok=True)
        with open(args.save_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote metadata json to {args.save_meta_json}")


if __name__ == "__main__":
    main()

