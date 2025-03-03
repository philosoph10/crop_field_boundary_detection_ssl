def compute_f1(tp, fp, fn):
    """
    Compute F1 score from True Positives, False Positives, and False Negatives.

    Parameters:
    tp (int): True Positives.
    fp (int): False Positives.
    fn (int): False Negatives.

    Returns:
    float: F1 score.
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f1
