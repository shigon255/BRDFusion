import cv2
import numpy as np
import argparse
import torch
from skimage.metrics import structural_similarity as ssim
import lpips


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images in the range [0, 1].
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(img1, img2):
    """
    Compute SSIM between two images.
    """
    return ssim(img1, img2, channel_axis=2, data_range=1.0)


def compute_lpips(img1, img2, lpips_fn):
    """
    Compute LPIPS between two images.
    """
    # Convert to tensor format (B, C, H, W) and range [-1, 1]
    t1 = torch.from_numpy(img1).unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
    t2 = torch.from_numpy(img2).unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
    return lpips_fn(t1, t2).item()


def compute_metrics_images(image1_path, image2_path, target_size=None):
    """
    Compute PSNR, SSIM, and LPIPS between two images.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        target_size: Optional tuple (width, height) to resize images before computing metrics
    """
    img1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("Error: Could not open one or both images")

    # Resize if requested
    if target_size is not None:
        img1 = cv2.resize(img1, target_size)
        img2 = cv2.resize(img2, target_size)
    else:
        # Require same shape if not resizing
        if img1.shape != img2.shape:
            raise ValueError(
                f"Images have different shapes: {img1.shape} vs {img2.shape}. "
                "Use --target_size WIDTH HEIGHT to resize them first."
            )

    # Convert BGR to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    lpips_fn = lpips.LPIPS(net="alex")

    return {
        "psnr": compute_psnr(img1, img2),
        "ssim": compute_ssim(img1, img2),
        "lpips": compute_lpips(img1, img2, lpips_fn),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, and LPIPS between two images"
    )
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize images to this size (width height) before computing metrics",
    )

    args = parser.parse_args()
    target_size = tuple(args.target_size) if args.target_size is not None else None

    metrics = compute_metrics_images(args.image1, args.image2, target_size)

    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"LPIPS: {metrics['lpips']:.4f}")