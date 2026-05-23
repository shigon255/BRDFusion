import cv2
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import lpips

def compute_psnr(frame1, frame2):
    """Compute PSNR between two frames in the range [0, 1]."""
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim(frame1, frame2):
    """Compute SSIM between two frames."""
    return ssim(frame1, frame2, channel_axis=2, data_range=1.0)

def compute_lpips(frame1, frame2, lpips_fn, device):
    """Compute LPIPS between two frames."""
    # Convert to tensor format (B, C, H, W) and range [-1, 1]
    f1 = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) * 2 - 1
    f2 = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) * 2 - 1
    return lpips_fn(f1, f2).item()

def compute_metrics_single_pair(gt_path, gen_path, lpips_fn, device, max_frames=None):
    """
    Compute average PSNR, SSIM, and LPIPS between a GT video and a generated video.
    """
    cap_gt = cv2.VideoCapture(gt_path)
    cap_gen = cv2.VideoCapture(gen_path)
    
    if not cap_gt.isOpened() or not cap_gen.isOpened():
        raise ValueError(f"Error: Could not open videos {gt_path} or {gen_path}")
    
    # Check frame counts (informative only)
    count_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
    count_gen = int(cap_gen.get(cv2.CAP_PROP_FRAME_COUNT))
    if count_gt != count_gen:
        # Warning suppressed to keep output clean
        pass 

    psnr_values, ssim_values, lpips_values = [], [], []
    frame_count = 0
    
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break

        ret1, frame_gt = cap_gt.read()
        ret2, frame_gen = cap_gen.read()
        
        if not ret1 or not ret2:
            break
        
        # Normalize to [0, 1]
        frame_gt = frame_gt.astype(np.float32) / 255.0
        frame_gen = frame_gen.astype(np.float32) / 255.0
        
        psnr_values.append(compute_psnr(frame_gt, frame_gen))
        ssim_values.append(compute_ssim(frame_gt, frame_gen))
        lpips_values.append(compute_lpips(frame_gt, frame_gen, lpips_fn, device))
        
        frame_count += 1
    
    cap_gt.release()
    cap_gen.release()
    
    if frame_count == 0:
        raise ValueError("No frames processed")
    
    return {
        'avg_psnr': np.mean(psnr_values),
        'avg_ssim': np.mean(ssim_values),
        'avg_lpips': np.mean(lpips_values)
    }

def parse_inputs(input_list):
    """
    Parses a list of strings in the format 'filepath=weight'.
    Returns a list of dicts: [{'path': ..., 'weight': ...}, ...]
    """
    parsed_data = []
    for item in input_list:
        if '=' not in item:
            print(f"[Error] Invalid format for input: '{item}'. Expected 'path=weight'. Skipping.")
            continue
        
        path, weight_str = item.rsplit('=', 1)
        
        try:
            weight = float(weight_str)
            parsed_data.append({'path': path, 'weight': weight})
        except ValueError:
            print(f"[Error] Invalid weight value in '{item}'. Skipping.")
            continue
            
    return parsed_data

def plot_results(results_list, output_path="metrics_curve.png"):
    """
    Plots PSNR, SSIM, and LPIPS.
    The plot connects points based on the order of weights (Ascending).
    """
    # Sort results by weight to ensure the line plot is a proper curve
    # (If you strictly want file-input-order regardless of weight value, remove this sort)
    results_list.sort(key=lambda x: x['weight'])
    
    weights = [x['weight'] for x in results_list]
    psnrs = [x['psnr'] for x in results_list]
    ssims = [x['ssim'] for x in results_list]
    lpips_vals = [x['lpips'] for x in results_list]

    plt.figure(figsize=(18, 5))

    # PSNR Plot
    plt.subplot(1, 3, 1)
    plt.plot(weights, psnrs, 'b-o', linewidth=2)
    plt.title("PSNR (Higher is Better)")
    plt.xlabel("Weight")
    plt.ylabel("dB")
    plt.grid(True, linestyle='--', alpha=0.7)

    # SSIM Plot
    plt.subplot(1, 3, 2)
    plt.plot(weights, ssims, 'g-o', linewidth=2)
    plt.title("SSIM (Higher is Better)")
    plt.xlabel("Weight")
    plt.ylabel("Score")
    plt.grid(True, linestyle='--', alpha=0.7)

    # LPIPS Plot
    plt.subplot(1, 3, 3)
    plt.plot(weights, lpips_vals, 'r-o', linewidth=2)
    plt.title("LPIPS (Lower is Better)")
    plt.xlabel("Weight")
    plt.ylabel("Distance")
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nCurve plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate video sequence with explicit labels.")
    parser.add_argument("gt_video", help="Path to the Ground Truth video")
    parser.add_argument("inputs", nargs='+', help="List of generated videos with weights. Format: path=weight (e.g., video1.mp4=0.1)")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to check per video")
    parser.add_argument("--output_plot", type=str, default="metrics_curve.png", help="Filename for the output plot")
    
    args = parser.parse_args()

    # 1. Parse Inputs
    video_list = parse_inputs(args.inputs)
    if not video_list:
        print("No valid video inputs found.")
        return

    print(f"Processing {len(video_list)} videos...")

    # 2. Setup LPIPS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LPIPS model on {device}...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    results = []

    # 3. Processing Loop
    print(f"\n{'Weight':<10} | {'File':<40} | {'PSNR':<10} | {'SSIM':<10} | {'LPIPS':<10}")
    print("-" * 95)

    for entry in video_list:
        gen_path = entry['path']
        weight = entry['weight']
        
        if not os.path.exists(gen_path):
            print(f"[Error] File not found: {gen_path}")
            continue

        try:
            metrics = compute_metrics_single_pair(
                args.gt_video, 
                gen_path, 
                lpips_fn, 
                device, 
                max_frames=args.max_frames
            )
            
            result_entry = {
                'weight': weight,
                'file': os.path.basename(gen_path),
                'psnr': metrics['avg_psnr'],
                'ssim': metrics['avg_ssim'],
                'lpips': metrics['avg_lpips']
            }
            results.append(result_entry)
            
            print(f"{weight:<10.2f} | {os.path.basename(gen_path):<40} | {result_entry['psnr']:<10.2f} | {result_entry['ssim']:<10.4f} | {result_entry['lpips']:<10.4f}")

        except Exception as e:
            print(f"Error processing {gen_path}: {e}")

    if not results:
        print("No valid results computed.")
        return

    # 4. Plotting
    plot_results(results, args.output_plot)

if __name__ == "__main__":
    main()