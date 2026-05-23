import cv2
import numpy as np
import argparse
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import lpips

def compute_psnr(frame1, frame2):
    """
    Compute PSNR between two frames in the range [0, 1].
    """
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return float('inf')
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
    # Convert to tensor format (B, C, H, W) and range [-1, 1]
    f1 = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
    f2 = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
    return lpips_fn(f1, f2).item()

def compute_metrics_videos(video1_path, video2_path, max_frames=None, target_size=None):
    """
    Compute PSNR, SSIM, and LPIPS between two videos.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        max_frames: Maximum number of frames to process
        target_size: Optional tuple (width, height) to resize frames before computing metrics
    """
    import torch

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Error: Could not open videos")
    
    # Get total frame counts for warning check
    count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video 1 frame count: {count1}")
    print(f"Video 2 frame count: {count2}")

    if count1 != count2 and (max_frames is None or max_frames > min(count1, count2)):
        # print(f"Warning: Video frame counts differ! Video 1: {count1}, Video 2: {count2}")
        raise ValueError(f"Videos have different number of frames. Please set max_frames to the smaller count.")

    lpips_fn = lpips.LPIPS(net='alex')
    
    psnr_values, ssim_values, lpips_values = [], [], []
    frame_count = 0
    
    while True:
        # Check if max_frames limit is reached
        if max_frames is not None and frame_count >= max_frames:
            break

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break

        # Resize frames if target_size is specified
        if target_size is not None:
            frame1 = cv2.resize(frame1, target_size)
            frame2 = cv2.resize(frame2, target_size)

        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0
        
        psnr_values.append(compute_psnr(frame1, frame2))
        ssim_values.append(compute_ssim(frame1, frame2))
        lpips_values.append(compute_lpips(frame1, frame2, lpips_fn))
        frame_count += 1
        
        print(f"Processed frame {frame_count}/{max_frames if max_frames is not None else 'end'}", end='\r')
    
    cap1.release()
    cap2.release()
    
    if frame_count == 0:
        raise ValueError("No frames processed from videos")
    
    return {
        'avg_psnr': np.mean(psnr_values),
        'avg_ssim': np.mean(ssim_values),
        'avg_lpips': np.mean(lpips_values),
        'psnr_list': psnr_values,
        'ssim_list': ssim_values,
        'lpips_list': lpips_values
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, and LPIPS between two videos")
    parser.add_argument("video1", help="Path to first video")
    parser.add_argument("video2", help="Path to second video")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--target_size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help="Resize frames to this size (width height) before computing metrics")

    args = parser.parse_args()

    target_size = tuple(args.target_size) if args.target_size is not None else None

    metrics = compute_metrics_videos(args.video1, args.video2, args.max_frames, target_size)
    print(f"Average PSNR: {metrics['avg_psnr']:.2f} dB")
    print(f"Average SSIM: {metrics['avg_ssim']:.4f}")
    print(f"Average LPIPS: {metrics['avg_lpips']:.4f}")
    print(f"Total frames: {len(metrics['psnr_list'])}")