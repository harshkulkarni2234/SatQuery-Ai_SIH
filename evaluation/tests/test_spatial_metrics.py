import numpy as np
import pytest

from evaluation.metrics.spatial_metrics import (
    box_iou,
    grounding_acc_at_iou,
    pixel_mask_f1_iou,
)


def test_box_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_box_iou_no_overlap_is_zero():
    assert box_iou((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0


def test_box_iou_hand_computed_half_overlap():
    # box a: (0,0,10,10) area=100; box b: (5,0,15,10) area=100
    # intersection: (5,0,10,10) area=50; union = 100+100-50=150
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    assert box_iou(a, b) == pytest.approx(50 / 150)


def test_grounding_acc_at_iou_hand_computed():
    preds = [(0, 0, 10, 10), None, (5, 0, 15, 10)]
    gts = [(0, 0, 10, 10), (0, 0, 5, 5), (5, 0, 15, 10)]
    result = grounding_acc_at_iou(preds, gts, threshold=0.5)
    # sample 1: IoU=1.0 (correct), sample 2: no pred (incorrect), sample 3: IoU=1.0 (correct)
    assert result["acc_at_iou"] == pytest.approx(2 / 3)
    assert result["n"] == 3


def test_grounding_acc_at_iou_length_mismatch_raises():
    with pytest.raises(ValueError):
        grounding_acc_at_iou([(0, 0, 1, 1)], [(0, 0, 1, 1), (0, 0, 1, 1)])


def test_pixel_mask_f1_iou_hand_computed():
    # 2x2 masks: pred marks top row, gt marks left column
    pred = np.array([[1, 1], [0, 0]])
    gt = np.array([[1, 0], [1, 0]])
    result = pixel_mask_f1_iou(pred, gt)
    # tp=1 (top-left), fp=1 (top-right), fn=1 (bottom-left)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["iou"] == pytest.approx(1 / 3)


def test_pixel_mask_f1_iou_perfect_match():
    mask = np.array([[1, 0], [0, 1]])
    result = pixel_mask_f1_iou(mask, mask)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["iou"] == 1.0


def test_pixel_mask_f1_iou_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pixel_mask_f1_iou(np.zeros((2, 2)), np.zeros((3, 3)))
