"""Grounding (IoU/acc@0.5) and change-detection (pixel F1/IoU) metrics.

Pure functions, no dataset dependency — see vqa_metrics.py for the same
design rationale.
"""

from __future__ import annotations

import numpy as np


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes. Returns 0.0 for non-overlapping
    or degenerate (zero-area) boxes rather than raising, since a "no overlap"
    result is itself a meaningful, common metric outcome."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def grounding_acc_at_iou(
    pred_boxes: list[tuple[float, float, float, float]],
    gt_boxes: list[tuple[float, float, float, float]],
    threshold: float = 0.5,
) -> dict:
    """acc@threshold for single-box-per-sample grounding: each predicted box
    is matched against its corresponding ground-truth box (same index — this
    is a per-sample metric, not a detection-matching problem), correct if
    IoU >= threshold. A missing prediction (None) counts as incorrect, not
    excluded, since "the system produced no box" is itself the outcome being
    measured."""
    if not gt_boxes:
        raise ValueError("grounding_acc_at_iou: empty ground_truths")
    if len(pred_boxes) != len(gt_boxes):
        raise ValueError("pred_boxes and gt_boxes must be the same length")
    correct = 0
    ious = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        if pred is None:
            ious.append(0.0)
            continue
        iou = box_iou(pred, gt)
        ious.append(iou)
        if iou >= threshold:
            correct += 1
    return {
        "acc_at_iou": correct / len(gt_boxes),
        "mean_iou": float(np.mean(ious)),
        "threshold": threshold,
        "n": len(gt_boxes),
    }


def pixel_mask_f1_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Pixel-level F1 and IoU for two boolean (or 0/1) change masks of the
    same shape. Raises on shape mismatch rather than silently reshaping."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    tp = int(np.sum(pred_bool & gt_bool))
    fp = int(np.sum(pred_bool & ~gt_bool))
    fn = int(np.sum(~pred_bool & gt_bool))

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    union = tp + fp + fn
    iou = tp / union if union > 0 else None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
