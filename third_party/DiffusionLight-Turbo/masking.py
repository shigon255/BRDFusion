import numpy as np
import imageio

def read_image(path):
    """
    Reads an image from the specified path and returns it as a numpy array.
    
    Args:
        path (str): The file path to the image.
        
    Returns:
        np.ndarray: The image as a numpy array.
    """
    return imageio.imread(path)

img_path = './output/envmap/036_000_0_crop_ev-00_seed140.png'
img = read_image(img_path)
mask = np.zeros_like(img, dtype=np.uint8)
valid_square = [350, 480, 50, 120]
mask[valid_square[2]:valid_square[3], valid_square[0]:valid_square[1]] = 1

# masking
def apply_mask(image, mask):
    """
    Applies a binary mask to an image.
    
    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The binary mask to apply.
        
    Returns:
        np.ndarray: The masked image.
    """
    return image * mask
masked_image = apply_mask(img, mask)
imageio.imwrite('./masked_image.png', masked_image)