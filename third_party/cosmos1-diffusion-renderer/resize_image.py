import os
import argparse
from PIL import Image
from tqdm import tqdm

def resize_images_in_place(input_folder, width, height):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_folder, filename)

            try:
                with Image.open(image_path) as img:
                    resized = img.resize((width, height), Image.Resampling.LANCZOS)
                    resized.save(image_path)
                    # print(f"Resized and overwritten: {image_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Resize all images in a folder to a specified width and height (overwrites originals).")
    parser.add_argument("folder", type=str, help="Path to the folder with images to resize.")
    parser.add_argument("width", type=int, help="Target width for resizing.")
    parser.add_argument("height", type=int, help="Target height for resizing.")
    args = parser.parse_args()

    resize_images_in_place(args.folder, args.width, args.height)

if __name__ == "__main__":
    main()
