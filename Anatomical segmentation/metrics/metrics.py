import numpy as np
import torch


def multilabel_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection) / (union + 1e-5)
    return float(dice.mean().item())


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Computes Dice, IoU, HD95, ASSD, Precision, Recall, Specificity, Accuracy.
    If both prediction and ground truth are empty: returns Dice=1, IoU=1, HD95=0, ASSD=0.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(target)).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()

    if not np.any(pred) and not np.any(target):
        return {
            "Dice": 1.0,
            "IoU": 1.0,
            "HD95": 0.0,
            "ASSD": 0.0,
            "Precision": 1.0,
            "Recall": 1.0,
            "Specificity": 1.0,
            "Accuracy": 1.0,
        }

    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    if not np.any(pred) or not np.any(target):
        hd95 = float("nan")
        assd = float("nan")
    else:
        from scipy.ndimage import binary_erosion
        from scipy.spatial.distance import cdist

        def get_boundary(m):
            eroded = binary_erosion(m)
            boundary = m ^ eroded
            return np.argwhere(boundary)

        coords_pred = get_boundary(pred)
        coords_target = get_boundary(target)

        if len(coords_pred) == 0 or len(coords_target) == 0:
            hd95 = float("nan")
            assd = float("nan")
        else:
            dists_p_to_t = cdist(coords_pred, coords_target, metric="euclidean").min(axis=1)
            dists_t_to_p = cdist(coords_target, coords_pred, metric="euclidean").min(axis=1)
            combined_dists = np.concatenate([dists_p_to_t, dists_t_to_p])

            hd95 = np.percentile(combined_dists, 95)
            assd = (dists_p_to_t.sum() + dists_t_to_p.sum()) / (len(dists_p_to_t) + len(dists_t_to_p))

    return {
        "Dice": dice,
        "IoU": iou,
        "HD95": hd95,
        "ASSD": assd,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Accuracy": accuracy,
    }
