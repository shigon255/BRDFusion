import argparse
import numpy as np
import os
import pyexr
import matplotlib.pyplot as plt
import imageio.v3 as iio

def load_hdr_image(path):
    img = iio.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img

def load_exr_image(path):
    data = pyexr.open(path)
    img = data.get()
    return img

def load_np(path):
    data = np.load(path)
    return data

def rgb_to_gray_average(img):
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[..., :3].mean(axis=2)
    else:
        raise ValueError("Input image does not have at least 3 channels (RGB)")
def rgb_to_gray_max(img):
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[..., :3].max(axis=2)
    else:
        raise ValueError("Input image does not have at least 3 channels (RGB)")

def max_pool_gray(gray_img, pool_size=16):
    h, w = gray_img.shape
    # pooling, but keep the size same by padding
    pad_h = (pool_size - h % pool_size) % pool_size
    pad_w = (pool_size - w % pool_size) % pool_size
    gray_img_padded = np.pad(gray_img, ((0, pad_h), (0, pad_w)), mode='edge')
    h_padded, w_padded = gray_img_padded.shape
    gray_img_reshaped = gray_img_padded.reshape(h_padded // pool_size, pool_size, w_padded // pool_size, pool_size)
    pooled = gray_img_reshaped.max(axis=(1, 3))
    return pooled

def plot_heatmap(gray_img, output_path, cmap='inferno'):
    plt.figure(figsize=(8, 6))
    vmin = np.min(gray_img)
    vmax = np.max(gray_img)
    im = plt.imshow(gray_img, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Average Intensity')
    # Set colorbar ticks to real values
    cbar.set_ticks(np.linspace(vmin, vmax, num=7))
    cbar.ax.set_yticklabels([f"{x:.3f}" for x in np.linspace(vmin, vmax, num=7)])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

import numpy as np

def plot_heatmap_log(gray_img, output_path, cmap='inferno', eps=1e-6):
    plt.figure(figsize=(8, 6))

    data = np.asarray(gray_img, float)
    data = np.maximum(data, eps)  # ensure > 0 for log

    log_data = np.log10(data)

    vmin = np.min(log_data)
    vmax = np.max(log_data)


    im = plt.imshow(log_data, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Average Intensity')

    # linear ticks in log space
    ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{10**x:.3g}" for x in ticks])

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a heatmap from an HDR image by averaging RGB channels.")
    parser.add_argument('input', help='Path to input HDR image')
    parser.add_argument('output', help='Path to output heatmap image (e.g., output.png)')
    parser.add_argument('--maxpool', action='store_true', help='Use max pooling')
    parser.add_argument('--cmap', default='inferno', help='Matplotlib colormap to use (default: inferno)')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input file {args.input} does not exist.")
        return

    # check ext of input
    if args.input.lower().endswith('.exr'):
        img = load_exr_image(args.input)
    elif args.input.lower().endswith('.npy'):
        img = load_np(args.input)
    elif args.input.lower().endswith(('.hdr', '.pfm', '.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        img = load_hdr_image(args.input)
    else:
        raise ValueError("Unsupported file format. Supported formats: .exr, .npy, .hdr, .pfm, .png, .jpg, .jpeg, .tiff, .tif")
    # gray = rgb_to_gray_average(img)
    gray = rgb_to_gray_max(img)
    if args.maxpool:
        gray = max_pool_gray(gray, pool_size=4)
    
    # plot_heatmap(gray, args.output, cmap=args.cmap)
    plot_heatmap_log(gray, args.output, cmap=args.cmap)
    print(f"Heatmap saved to {args.output}")

if __name__ == '__main__':
    main()