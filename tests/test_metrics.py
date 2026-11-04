import numpy as np
from src.lab.metrics import dice, iou, overlap


def test_dice_iou_known_values():
    a = np.zeros((4, 4), bool); a[:2, :] = True      # 8 px
    b = np.zeros((4, 4), bool); b[1:3, :] = True     # 8 px, overlap 4
    assert abs(dice(a, b) - 0.5) < 1e-9
    assert abs(iou(a, b) - 4 / 12) < 1e-9
    assert overlap(a, b) and not overlap(a, ~a)


def test_empty_masks_score_one_and_probabilities_threshold():
    z = np.zeros((3, 3))
    assert dice(z, z) == 1.0 and iou(z, z) == 1.0
    p = np.array([[0.9, 0.1], [0.6, 0.4]]); t = np.array([[1, 0], [1, 0]])
    assert dice(p, t) == 1.0
