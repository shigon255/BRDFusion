import argparse
from PIL import Image
import sys

def resize_image(input_path, output_path, width, height, interpolation='lanczos'):
    """
    Resize an image to target dimensions.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        width: Target width in pixels
        height: Target height in pixels
        interpolation: Interpolation method (lanczos, bilinear, bicubic, nearest)
    """
    try:
        # Map interpolation method strings to PIL constants
        methods = {
            'lanczos': Image.Resampling.LANCZOS,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'nearest': Image.Resampling.NEAREST
        }
        
        img = Image.open(input_path)
        resized_img = img.resize((width, height), methods.get(interpolation, Image.Resampling.LANCZOS))
        resized_img.save(output_path)
        print(f"Image resized from {img.size} to ({width}, {height}) and saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Resize an image to target dimensions')
    parser.add_argument('input', help='Path to input image')
    parser.add_argument('output', help='Path to save resized image')
    parser.add_argument('--width', type=int, required=True, help='Target width in pixels')
    parser.add_argument('--height', type=int, required=True, help='Target height in pixels')
    parser.add_argument('--interpolation', choices=['lanczos', 'bilinear', 'bicubic', 'nearest'],
                        default='lanczos', help='Interpolation method (default: lanczos)')
    
    args = parser.parse_args()
    resize_image(args.input, args.output, args.width, args.height, args.interpolation)


if __name__ == '__main__':
    main()