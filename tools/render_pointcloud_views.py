import argparse
import json
import os
from typing import Dict, List, Tuple

import imageio
import numpy as np
from tqdm import tqdm
import torch
from plyfile import PlyData
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection


def to8b(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


def load_points_from_ply(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        c0 = 0.28209479177387814
        sh0 = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
        rgb = sh0 * c0 + 0.5
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.full_like(xyz, 0.7, dtype=np.float32)

    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
    xyz = xyz[finite]
    rgb = np.clip(rgb[finite], 0.0, 1.0)
    return xyz, rgb


def load_cameras(camera_json: str) -> List[Dict]:
    with open(camera_json, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not isinstance(cameras, list) or len(cameras) == 0:
        raise ValueError(f"No camera entries found in {camera_json}")
    return cameras


def render_single_view(
    points: torch.Tensor,
    colors: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    H: int,
    W: int,
    radius: float,
    points_per_pixel: int,
    bg_color: Tuple[float, float, float],
) -> np.ndarray:
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3].unsqueeze(0)
    T = w2c[:3, 3].unsqueeze(0)
    image_size = torch.tensor([[H, W]], dtype=torch.float32, device=points.device)
    cams = cameras_from_opencv_projection(R, T, K.unsqueeze(0), image_size)

    pcd = Pointclouds(points=[points], features=[colors])
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=radius,
        points_per_pixel=points_per_pixel,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cams, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=bg_color),
    )
    rendered = renderer(pcd)[0, ..., :3].detach().cpu().numpy()
    return np.clip(rendered, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser("Render exported point cloud from camera views")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to exported PLY")
    parser.add_argument("--camera_json", type=str, required=True, help="Path to extracted cameras.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for rendered images/video")
    parser.add_argument("--fps", type=int, default=10, help="Video fps")
    parser.add_argument("--radius", type=float, default=0.002, help="Point radius in NDC units")
    parser.add_argument("--points_per_pixel", type=int, default=16, help="Points per pixel for rasterizer")
    parser.add_argument("--max_points", type=int, default=500000, help="Randomly subsample points to this count")
    parser.add_argument("--bg_white", action="store_true", help="Use white background (default: black)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(image_dir, exist_ok=True)

    xyz_np, rgb_np = load_points_from_ply(args.ply_path)
    if xyz_np.shape[0] == 0:
        raise ValueError("No valid points loaded from PLY.")

    if args.max_points > 0 and xyz_np.shape[0] > args.max_points:
        sel = np.random.permutation(xyz_np.shape[0])[: args.max_points]
        xyz_np = xyz_np[sel]
        rgb_np = rgb_np[sel]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.from_numpy(xyz_np).to(device)
    colors = torch.from_numpy(rgb_np).to(device)
    bg_color = (1.0, 1.0, 1.0) if args.bg_white else (0.0, 0.0, 0.0)

    cameras = load_cameras(args.camera_json)
    frames = []
    for i, cam in tqdm(enumerate(cameras), total=len(cameras), desc="Rendering views"):
        K = torch.tensor(cam["intrinsics"], dtype=torch.float32, device=device)
        c2w = torch.tensor(cam["extrinsics"], dtype=torch.float32, device=device)
        H = int(cam["height"])
        W = int(cam["width"])
        img = render_single_view(
            points=points,
            colors=colors,
            K=K,
            c2w=c2w,
            H=H,
            W=W,
            radius=args.radius,
            points_per_pixel=args.points_per_pixel,
            bg_color=bg_color,
        )
        frame_u8 = to8b(img)
        frame_path = os.path.join(image_dir, f"{i:05d}.png")
        imageio.imwrite(frame_path, frame_u8)
        frames.append(frame_u8)

    video_path = os.path.join(args.output_dir, "pointcloud_render.mp4")
    imageio.mimwrite(video_path, frames, fps=args.fps)
    print(f"[OK] Saved {len(frames)} frames to {image_dir}")
    print(f"[OK] Saved video to {video_path}")


if __name__ == "__main__":
    main()
