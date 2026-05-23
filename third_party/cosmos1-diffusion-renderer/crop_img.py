import argparse
import cv2
from pathlib import Path

def crop_image(image_path, bbox, output_path):
    """
    Crop an image based on bounding box coordinates.
    
    Args:
        image_path: Path to input image
        bbox: Bounding box as [x1, y1, x2, y2]
        output_path: Path to save cropped image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    x1, y1, x2, y2 = map(int, bbox)
    cropped = img[y1:y2, x1:x2]
    
    cv2.imwrite(output_path, cropped)
    print(f"Cropped image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Crop an image using bounding box coordinates")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, 
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Bounding box coordinates: x1 y1 x2 y2")
    parser.add_argument("--output", "-o", default="cropped.jpg",
                        help="Path to save cropped image (default: cropped.jpg)")
    
    args = parser.parse_args()
    crop_image(args.image, args.bbox, args.output)


if __name__ == "__main__":
    main()