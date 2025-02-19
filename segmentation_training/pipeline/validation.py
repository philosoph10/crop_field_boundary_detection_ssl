from pathlib import Path

import numpy as np
import torch
from slicing_utils.tiles import ImageSlicer
from pipeline.inference import sliced_inference
from tqdm import tqdm
from utils.read import read_mask_png, read_rgb
from utils.viz import plot_images


def validate_model(
    model: torch.nn.Module,
    image_paths: list[Path],
    label_paths: list[Path],
    tile_size: tuple[int, int],
    tile_step: tuple[int, int],
    conf: float = 0.5,
    viz_dir: Path | None = None,
) -> dict:
    """
    Validate a PyTorch semantic segmentation model using sliding window inference.

    Parameters:
        model (torch.nn.Module): The PyTorch segmentation model.
        image_paths (list[Path | str]): List of image file paths.
        label_path (Path): Path to the directory containing ground truth masks.
        tile_size (tuple[int, int]): Tile size (h x w) for sliding window inference.
        tile_step (tuple[int, int]): Tile step (h x w) for sliding window inference.
        conf (float): Confidence threshold for generating binary masks.

    Returns:
        dict: Dictionary containing cumulative IoU and F1-score.
    """
    tp = 0
    fp = 0
    fn = 0

    imagewise_ious = []
    imagewise_f1s = []

    for image_path, label_path in tqdm(
        zip(image_paths, label_paths, strict=True), total=len(image_paths), desc="Sliced validation"
    ):
        image = read_rgb(image_path.as_posix())

        slicer = ImageSlicer(
            image_shape=image.shape[:2],
            tile_size=tile_size,
            tile_step=tile_step,
        )

        # Run inference
        pred_mask = sliced_inference(model, image, slicer, conf)
        pred_mask = (pred_mask > 0).astype(np.uint8)  # Ensure binary format

        # Get corresponding label masks
        label_masks = list(label_path.glob("*.png"))

        # Merge label masks into a single binary mask
        gt_mask = np.zeros_like(pred_mask, dtype=bool)
        for mask_path in label_masks:
            mask = (read_mask_png(mask_path.as_posix()) > 0).astype(np.uint8)  # Threshold at 0
            gt_mask = np.logical_or(gt_mask, mask)

        if viz_dir is not None:
            save_path = viz_dir / f"{image_path.stem}_pred_gt.png"
            plot_images([image, pred_mask, gt_mask], title="Image vs prediction vs GT", save_path=save_path)

        cur_tp = int(np.sum((pred_mask == 1) & (gt_mask == 1)))
        cur_fp = int(np.sum((pred_mask == 1) & (gt_mask == 0)))
        cur_fn = int(np.sum((pred_mask == 0) & (gt_mask == 1)))

        # Including these cases would skew the metrics
        if np.sum(gt_mask) > 0 and np.sum(pred_mask) > 0:
            imagewise_ious.append(cur_tp / (cur_tp + cur_fp + cur_fn))
            imagewise_f1s.append(cur_tp / (cur_tp + 0.5 * (cur_fp + cur_fn)))
            # print(f"IOU: {imagewise_ious[-1]}")
            # print(f"F1: {imagewise_f1s[-1]}")

        tp += cur_tp
        fp += cur_fp
        fn += cur_fn

    # Average metrics
    avg_iou = tp / (tp + fp + fn)
    avg_f1 = tp / (tp + 0.5 * (fp + fn))

    return {
        "IoU": {
            "Total": float(avg_iou),
            "Imagewise": float(np.mean(imagewise_ious)),
        },
        "F1": {
            "Total": float(avg_f1),
            "Imagewise": float(np.mean(imagewise_f1s)),
        },
    }
