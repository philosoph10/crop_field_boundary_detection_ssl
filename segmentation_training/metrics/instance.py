import multiprocessing as mp

import numpy as np
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .speed_utils import get_bbox, compute_iou_for_gt


def boundary_to_instance_mask(boundary_mask: np.ndarray) -> np.ndarray:
    """
    Converts a binary boundary mask into an instance segmentation mask.

    Parameters:
    boundary_mask (np.ndarray): Binary mask where object boundaries are 1, background is 0.

    Returns:
    np.ndarray: Instance segmentation mask where each connected component has a unique ID.
    """
    # Invert the mask: boundary pixels become 0, non-boundary pixels become 1
    inverted_mask = (boundary_mask == 0).astype(np.uint8)
    
    # Perform connected component analysis
    num_labels, labels = cv2.connectedComponents(inverted_mask, connectivity=8)
    
    # Ensure background remains zero
    instance_mask = labels.astype(np.uint16)
    
    return instance_mask


def compute_f1_instance_segmentation(gt_mask, pred_mask, iou_threshold=0.5):
    """
    Compute TP, FP, FN for instance segmentation masks using bounding box pre-filtering and parallelized IoU computation.
    
    Parameters:
        gt_mask (numpy.ndarray): Ground truth instance segmentation mask (HxW, unique IDs per instance).
        pred_mask (numpy.ndarray): Prediction instance segmentation mask (HxW, unique IDs per instance).
        iou_threshold (float): IoU threshold for matching.
    
    Returns:
        tp (int): True Positives (matched instances).
        fp (int): False Positives (unmatched predictions).
        fn (int): False Negatives (unmatched ground truths).
    """
    gt_instances = np.unique(gt_mask)
    pred_instances = np.unique(pred_mask)
    
    # Remove background (assuming 0 is background)
    gt_instances = gt_instances[gt_instances > 0]
    pred_instances = pred_instances[pred_instances > 0]
    
    # Precompute bounding boxes for all instances
    gt_bboxes = {gt_id: get_bbox(gt_mask, gt_id) for gt_id in tqdm(gt_instances, desc="Computing GT Bounding Boxes")}
    pred_bboxes = {pred_id: get_bbox(pred_mask, pred_id) for pred_id in tqdm(pred_instances, desc="Computing Prediction Bounding Boxes")}
    
    iou_matrix = np.zeros((len(gt_instances), len(pred_instances)))
    
    # Parallel computation of IoU with tqdm progress bar
    results = process_map(
        compute_iou_for_gt,
        [(gt_id, gt_mask, pred_mask, pred_instances, gt_bboxes, pred_bboxes) for gt_id in gt_instances],
        max_workers=mp.cpu_count(),
        chunksize=1,
        desc="Computing IoU",
    )
    
    # Fill IoU matrix with results
    for i, local_iou in enumerate(results):
        iou_matrix[i] = local_iou
    
    # Matching GT to Predictions using Greedy Matching
    matched_gt = set()
    matched_pred = set()
    
    # Sort matches by IoU (highest first)
    sorted_matches = sorted(
        [(i, j, iou_matrix[i, j]) for i in range(len(gt_instances)) for j in range(len(pred_instances)) if iou_matrix[i, j] >= iou_threshold],
        key=lambda x: x[2], reverse=True
    )
    
    for gt_idx, pred_idx, iou in sorted_matches:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
    
    tp = len(matched_gt)
    fp = len(pred_instances) - len(matched_pred)
    fn = len(gt_instances) - len(matched_gt)
    
    return tp, fp, fn
