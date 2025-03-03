import random

import torch
import cv2
import numpy as np


def visualize_dataset(dataset, logger, num_images=32):
    """
    Logs a sample of images and masks to TensorBoard before training.

    Args:
        dataset (SegmentationDataset): The dataset object.
        logger (TensorBoardLogger): The logger instance.
        num_images (int): Number of images to log.
    """
    if logger is None or logger.experiment is None:
        return  # No logging if logger isn't available

    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

    transform = dataset.final_transform  # Save final transform
    dataset.final_transform = None  # Disable final transform for visualization

    for i, idx in enumerate(indices):
        image_np, mask_np = dataset[idx]  # Get sample
        image_np = image_np[:, :, :3] # Extract RGB channels
        mask_np = (mask_np > 0).astype(np.uint8) * 255  # HW, scaled

        # Convert mask to color (if it's grayscale)
        mask_colored = cv2.applyColorMap(mask_np, cv2.COLORMAP_INFERNO)

        # Blend image & mask
        blended = cv2.addWeighted(image_np.astype(np.uint8), 0.6, mask_colored, 0.4, 0)

        # Convert to Tensor
        img_tensor = torch.tensor(image_np).permute(2, 0, 1) / 255.0    
        mask_tensor = torch.tensor(mask_colored).permute(2, 0, 1) / 255.0
        blended_tensor = torch.tensor(blended).permute(2, 0, 1) / 255.0

        # Log to TensorBoard
        logger.experiment.add_image(f"train_samples/{str(i).zfill(3)}_image", img_tensor, dataformats="CHW")
        logger.experiment.add_image(f"train_samples/{str(i).zfill(3)}_mask", mask_tensor, dataformats="CHW")
        logger.experiment.add_image(f"train_samples/{str(i).zfill(3)}_blended", blended_tensor, dataformats="CHW")
    
    dataset.final_transform = transform  # Re-enable final transform
