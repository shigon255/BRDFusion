import os
import argparse
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

def collect_images(input_folder):
    image_dict = defaultdict(dict)
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for file in os.listdir(input_folder):
        if file.lower().endswith(valid_extensions):
            try:
                name, _ = os.path.splitext(file)
                base, index = name.rsplit('_', 1)
                if index.isdigit():
                    image_dict[base][int(index)] = os.path.join(input_folder, file)
            except ValueError:
                continue  # skip files that don't follow the pattern
    return image_dict

def combine_images(image_paths, order):
    images = []
    for idx in order:
        path = image_paths.get(idx)
        if path:
            images.append(Image.open(path))
        else:
            raise ValueError(f"Missing image for index {idx}")
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    combined = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    return combined

def main():
    parser = argparse.ArgumentParser(description="Combine images in specified order.")
    parser.add_argument('input_folder', type=str, help="Path to input folder containing images.")
    parser.add_argument('output_folder', type=str, help="Path to save combined images.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    image_dict = collect_images(args.input_folder)
    
    combine_order = [1, 0, 2]
    
    for base, paths in tqdm(image_dict.items()):
        try:
            combined_img = combine_images(paths, combine_order)
            output_path = os.path.join(args.output_folder, f"{base}.png")
            combined_img.save(output_path)
            print(f"Saved combined image: {output_path}")
        except Exception as e:
            print(f"Skipping {base} due to error: {e}")

if __name__ == '__main__':
    main()
