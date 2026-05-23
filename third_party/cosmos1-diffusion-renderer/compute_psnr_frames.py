import argparse
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def compute_psnr(frame1, frame2):
    """
    Compute PSNR between two frames in the range [0, 1].
    """
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(frame1, frame2):
    """
    Compute SSIM between two frames.
    """
    return ssim(frame1, frame2, channel_axis=2, data_range=1.0)


def compute_lpips(frame1, frame2, lpips_fn):
    """
    Compute LPIPS between two frames.
    """
    f1 = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
    f2 = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
    return lpips_fn(f1, f2).item()


def list_frame_files(folder_path):
    frame_dir = Path(folder_path)
    if not frame_dir.is_dir():
        raise ValueError(f"Error: {folder_path} is not a directory")

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    frame_files = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in valid_suffixes and p.is_file()])
    if not frame_files:
        raise ValueError(f"Error: No image frames found in {folder_path}")
    return frame_files


def compute_metrics_folders(folder1_path, folder2_path, max_frames=None, target_size=None):
    """
    Compute PSNR, SSIM, and LPIPS between two frame folders.

    Args:
        folder1_path: Path to first frame folder
        folder2_path: Path to second frame folder
        max_frames: Maximum number of frames to process
        target_size: Optional tuple (width, height) to resize frames before computing metrics
    """
    frame_files1 = list_frame_files(folder1_path)
    frame_files2 = list_frame_files(folder2_path)

    count1 = len(frame_files1)
    count2 = len(frame_files2)

    print(f"Folder 1 frame count: {count1}")
    print(f"Folder 2 frame count: {count2}")

    if count1 != count2 and (max_frames is None or max_frames > min(count1, count2)):
        raise ValueError("Folders have different number of frames. Please set max_frames to the smaller count.")

    lpips_fn = lpips.LPIPS(net="alex")

    psnr_values, ssim_values, lpips_values = [], [], []
    frame_count = 0
    total_frames = min(count1, count2) if max_frames is None else min(count1, count2, max_frames)

    for frame_path1, frame_path2 in zip(frame_files1, frame_files2):
        if max_frames is not None and frame_count >= max_frames:
            break

        frame1 = cv2.imread(str(frame_path1), cv2.IMREAD_COLOR)
        frame2 = cv2.imread(str(frame_path2), cv2.IMREAD_COLOR)

        if frame1 is None or frame2 is None:
            raise ValueError(f"Error reading frame pair: {frame_path1} and {frame_path2}")

        if target_size is not None:
            frame1 = cv2.resize(frame1, target_size)
            frame2 = cv2.resize(frame2, target_size)

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        psnr_values.append(compute_psnr(frame1, frame2))
        ssim_values.append(compute_ssim(frame1, frame2))
        lpips_values.append(compute_lpips(frame1, frame2, lpips_fn))
        frame_count += 1

        print(f"Processed frame {frame_count}/{total_frames}", end="\r")

    if frame_count == 0:
        raise ValueError("No frames processed from folders")

    return {
        "avg_psnr": np.mean(psnr_values),
        "avg_ssim": np.mean(ssim_values),
        "avg_lpips": np.mean(lpips_values),
        "psnr_list": psnr_values,
        "ssim_list": ssim_values,
        "lpips_list": lpips_values,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, and LPIPS between two frame folders")
    parser.add_argument("folder1", help="Path to first frame folder")
    parser.add_argument("folder2", help="Path to second frame folder")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize frames to this size (width height) before computing metrics",
    )

    args = parser.parse_args()

    target_size = tuple(args.target_size) if args.target_size is not None else None

    metrics = compute_metrics_folders(args.folder1, args.folder2, args.max_frames, target_size)
    print(f"Average PSNR: {metrics['avg_psnr']:.2f} dB")
    print(f"Average SSIM: {metrics['avg_ssim']:.4f}")
    print(f"Average LPIPS: {metrics['avg_lpips']:.4f}")
    print(f"Total frames: {len(metrics['psnr_list'])}")
