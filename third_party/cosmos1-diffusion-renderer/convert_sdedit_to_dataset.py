#!/usr/bin/env python3
"""
Convert SDEdit concatenated videos to dataset format.
Source: merged videos with 3 cameras side-by-side
Target: individual images as {timestep:03d}_{camid}.jpg
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_video_to_images(
    video_path: str,
    output_dir: str,
    num_cameras: int = 3,
    valid_frames: int = 198,  # Only frames 0-197 are valid
    camera_order: list = None,  # Order of cameras from left to right in video
    target_size: tuple = None,  # Target size (width, height) for each camera image
):
    """
    Convert a concatenated video (cameras side-by-side) to individual images.

    Args:
        video_path: Path to the input video
        output_dir: Directory to save output images
        num_cameras: Number of cameras concatenated horizontally
        valid_frames: Number of valid frames (skip padding at the end)
        camera_order: List mapping position index to camera ID.
                      e.g., [1, 0, 2] means left=cam1, middle=cam0, right=cam2
        target_size: Target size (width, height) for each camera image. If None, no resizing.
    """
    if camera_order is None:
        camera_order = list(range(num_cameras))

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cam_width = frame_width // num_cameras

    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}, Valid frames: {valid_frames}")
    print(f"  Frame size: {frame_width}x{frame_height}")
    print(f"  Camera size: {cam_width}x{frame_height}")
    if target_size:
        print(f"  Target size: {target_size[0]}x{target_size[1]}")
    print(f"  Camera order (left to right): {camera_order}")
    print(f"  Output: {output_dir}")

    frame_idx = 0
    pbar = tqdm(total=valid_frames, desc="Converting")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip invalid frames (padding at the end)
        if frame_idx >= valid_frames:
            break

        # Split frame into individual cameras
        for pos_idx in range(num_cameras):
            x_start = pos_idx * cam_width
            x_end = (pos_idx + 1) * cam_width
            cam_frame = frame[:, x_start:x_end]

            # Resize if target_size is specified
            if target_size is not None:
                cam_frame = cv2.resize(cam_frame, target_size, interpolation=cv2.INTER_LANCZOS4)

            # Map position to actual camera ID
            cam_id = camera_order[pos_idx]

            # Save as {timestep:03d}_{camid}.jpg
            output_path = os.path.join(output_dir, f"{frame_idx:03d}_{cam_id}.jpg")
            cv2.imwrite(output_path, cam_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"  Saved {frame_idx} frames × {num_cameras} cameras = {frame_idx * num_cameras} images")


def main():
    parser = argparse.ArgumentParser(description="Convert SDEdit videos to dataset format")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/project2/yi-ray/BRDFusion/third_party/cosmos1-diffusion-renderer/drivestudio_exp/3/refine_intrinsics_gtrgb_full_refine_overlap50/sdedit_w.2_seed1000",
        help="Source directory containing merged videos",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/project2/yi-ray/BRDFusion/data/waymo/processed/training/003",
        help="Target dataset directory",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="diffusion_renderer_sdeditw02refineoverlap50",
        help="Prefix for output folder names",
    )
    parser.add_argument(
        "--valid_frames",
        type=int,
        default=198,
        help="Number of valid frames (default: 198, i.e., frames 0-197)",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=960,
        help="Target width for each camera image",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=640,
        help="Target height for each camera image",
    )
    args = parser.parse_args()

    # Mapping of video property names to output folder names
    intrinsics = ["basecolor", "depth", "metallic", "normal", "roughness"]
    output_intrinsics = ["albedo", "depth", "metallic", "normal", "roughness"]

    target_size = (args.target_width, args.target_height)

    for intrinsic, output_intrinsic in zip(intrinsics, output_intrinsics):
        video_name = f"fullrefinegt_rgb.{intrinsic}.mp4"
        video_path = os.path.join(args.source_dir, video_name)

        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue

        output_dir = os.path.join(args.target_dir, f"{args.output_prefix}_{output_intrinsic}")

        convert_video_to_images(
            video_path=video_path,
            output_dir=output_dir,
            num_cameras=3,
            valid_frames=args.valid_frames,
            camera_order=[1, 0, 2],  # Left=cam1, Middle=cam0, Right=cam2
            target_size=target_size,
        )
        print()


if __name__ == "__main__":
    main()
