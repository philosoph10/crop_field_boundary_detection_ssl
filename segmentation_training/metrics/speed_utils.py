import numpy as np


def get_bbox(mask, instance_id):
    """Compute bounding box [y_min, x_min, y_max, x_max] for a given instance."""
    y_indices, x_indices = np.where(mask == instance_id)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # No pixels found for this instance
    return [np.min(y_indices), np.min(x_indices), np.max(y_indices), np.max(x_indices)]

def bbox_intersects(bbox1, bbox2):
    """Check if two bounding boxes intersect."""
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def compute_iou_for_gt(args):
    """Unpacks arguments and computes IoU for a single ground truth instance."""
    gt_id, gt_mask, pred_mask, pred_instances, gt_bboxes, pred_bboxes = args
    gt_instance = (gt_mask == gt_id).astype(np.uint8)
    gt_bbox = gt_bboxes[gt_id]
    local_iou = np.zeros(len(pred_instances))
    
    for j, pred_id in enumerate(pred_instances):
        pred_bbox = pred_bboxes[pred_id]
        
        # Skip computation if bounding boxes do not intersect
        if not bbox_intersects(gt_bbox, pred_bbox):
            continue
        
        pred_instance = (pred_mask == pred_id).astype(np.uint8)
        
        intersection = np.sum(gt_instance * pred_instance)
        union = np.sum(gt_instance) + np.sum(pred_instance) - intersection
        
        if union > 0:
            local_iou[j] = intersection / union
    
    return local_iou
