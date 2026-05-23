import os
import sys
from PIL import Image
import argparse
from tqdm import tqdm

def resize_and_pad(img, size=(1024, 1024)):
    # Resize while keeping aspect ratio
    img.thumbnail(size, Image.LANCZOS)
    # Create new image and paste the resized on center
    new_img = Image.new("RGB", size, (0, 0, 0))
    left = (size[0] - img.width) // 2
    top = (size[1] - img.height) // 2
    new_img.paste(img, (left, top))
    return new_img

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in tqdm(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            try:
                img = Image.open(fpath).convert("RGB")
                out_img = resize_and_pad(img)
                out_path = os.path.join(output_dir, fname)
                out_img.save(out_path)
                print(f"Processed {fname}")
            except Exception as e:
                print(f"Failed to process {fname}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Resize and pad all images in a directory to 1024x1024.")
    parser.add_argument("input_dir", help="Directory containing images to process.")
    parser.add_argument("output_dir", help="Directory to save processed images.")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
