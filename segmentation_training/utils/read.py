import cv2
import numpy as np


def read_rgb(image_path: str) -> np.ndarray:
    """
    Read an image file to RGB.
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask_png(mask_path: str) -> np.ndarray:
    """
    Read a single channel mask from a .png file
    """
    return cv2.imread(mask_path)[:, :, 0]
