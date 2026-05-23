import os
import sys
from PIL import Image
import argparse
from tqdm import tqdm

def resize_and_crop(img, size=(1024, 1024)):
    # Calculate aspect ratios
    img_ratio = img.width / img.height
    target_ratio = size[0] / size[1]
    
    if img_ratio > target_ratio:
        # Image is wider than target, crop width
        new_height = img.height
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = new_height
    else:
        # Image is taller than target, crop height
        new_width = img.width
        new_height = int(img.width / target_ratio)
        left = 0
        top = (img.height - new_height) // 2
        right = new_width
        bottom = top + new_height
    
    # Crop the image from center
    img_cropped = img.crop((left, top, right, bottom))
    
    # Resize to target size
    return img_cropped.resize(size, Image.LANCZOS)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in tqdm(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            try:
                img = Image.open(fpath).convert("RGB")
                out_img = resize_and_crop(img)
                out_path = os.path.join(output_dir, fname)
                out_img.save(out_path)
                print(f"Processed {fname}")
            except Exception as e:
                print(f"Failed to process {fname}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Resize and center crop all images in a directory to 1024x1024.")
    parser.add_argument("input_dir", help="Directory containing images to process.")
    parser.add_argument("output_dir", help="Directory to save processed images.")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
