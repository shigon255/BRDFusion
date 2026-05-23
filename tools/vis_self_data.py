import argparse
import os

import imageio
import imageio.v3 as iio
import pyexr
from tqdm import tqdm
import numpy as np

from datasets.dataset_meta import DATASETS_CONFIG


def read_image(path: str, scale: float, srgb: bool = False, normalize_depth: bool = False, normalize_normal: bool = False) -> np.ndarray:
    if path.lower().endswith(".exr"):
        img = pyexr.read(path)
        if normalize_depth:
            img[img < 0] = np.max(img)  # set invalid depth to max depth for better visualization
            img = img / np.max(img) if np.max(img) > 0 else img
        if normalize_normal:
            # normal is exr in [-1, 1], convert to [0, 1]
            # before that, flip z to meet convention of normal map
            img[..., 2] = -img[..., 2]
            img = (img + 1) / 2
        if srgb:
            img = np.power(img, 1/2.2)
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    else:
        img = iio.imread(path)
        if srgb:
            img = img.astype(np.float32) / 255.0
            img = np.power(img, 1/2.2)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if scale != 1.0:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("cv2 is required for resizing; install opencv-python") from exc
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize self-crafted dataset as a multi-view video.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--camera_names", type=str, nargs="+", required=True)
    parser.add_argument("--subdir", type=str, default=None, help="Optional subfolder for priors")
    parser.add_argument("--extension", type=str, default="png")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=20)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--normalize_depth", action="store_true", help="Whether to normalize depth values to [0, 1]")
    parser.add_argument("--normalize_normal", action="store_true", help="Whether to normalize normal vectors to [0, 1]")
    parser.add_argument("--srgb", action="store_true", help="Whether to apply linear to sRGB gamma correction")
    return parser.parse_args()


def main():
    args = parse_args()
    cam_name_to_id = {
        meta["camera_name"]: cam_id for cam_id, meta in DATASETS_CONFIG["self"].items()
    }

    writer = imageio.get_writer(args.output_video, mode="I", fps=args.fps)
    for frame_id in tqdm(range(args.start_frame, args.max_step + 1, args.frame_step), desc="Frames"):
        imgs = []
        for cam in args.camera_names:
            if cam not in cam_name_to_id:
                raise ValueError(f"Unknown camera name '{cam}'. Valid names: {list(cam_name_to_id.keys())}")
            cam_id = cam_name_to_id[cam]
            if args.subdir:
                fname = f"{frame_id:03d}_{cam_id}.{args.extension}"
                path = os.path.join(args.dataset_root, args.subdir, fname)
            else:
                fname = f"{frame_id:03d}_{cam_id}.{args.extension}"
                path = os.path.join(args.dataset_root, "images", fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing image: {path}")
            imgs.append(read_image(path, args.scale, srgb=args.srgb, normalize_depth=args.normalize_depth, normalize_normal=args.normalize_normal))
        writer.append_data(np.concatenate(imgs, axis=1))

    writer.close()
    print(f"Saved video to {args.output_video}")


if __name__ == "__main__":
    main()
