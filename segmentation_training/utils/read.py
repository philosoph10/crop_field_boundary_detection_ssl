import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image


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


def read_raster(image_path: str, band: int | None = None) -> np.ndarray:
    """
    Read a raster image to a numpy array.
    """
    with rasterio.open(image_path) as dataset:
        if band is not None:
            return dataset.read(band)
        return reshape_as_image(dataset.read())
