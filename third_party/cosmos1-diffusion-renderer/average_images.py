import sys
import numpy as np
from PIL import Image
import os

def average_images(image_paths, output_path):
    if len(image_paths) == 0:
        print("No input images provided.")
        return

    # Load the first image and convert to numpy array
    base_image = Image.open(image_paths[0]).convert("RGB")
    base_array = np.array(base_image, dtype=np.float64)

    # Process the rest of the images
    for path in image_paths[1:]:
        img = Image.open(path).convert("RGB").resize(base_image.size)
        base_array += np.array(img, dtype=np.float64)

    # Average the pixel values
    avg_array = (base_array / len(image_paths)).astype(np.uint8)

    # Save the result
    avg_image = Image.fromarray(avg_array)
    avg_image.save(output_path)
    # print(f"Averaged image saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python average_images.py output_path input1.jpg input2.jpg ...")
    else:
        output = sys.argv[1]
        inputs = sys.argv[2:]
        average_images(inputs, output)
