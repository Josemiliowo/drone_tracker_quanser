"""Yellow-centre check that separates the printed fires from red objects (e.g. a red tree)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone'))
from fire_detector import yellow_fraction  # noqa: E402

IMG = cv2.imread(os.path.join(os.path.dirname(__file__), 'data', 'fire_sample.png'))
H, W = IMG.shape[:2]
MIN = 0.08  # default --yellow-min


def test_real_fire_passes():
    assert yellow_fraction(IMG, 0, 0, W, H) > 0.2


def test_red_centre_rejected():
    hsv = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = np.where(hsv[..., 0] < 30, 2, hsv[..., 0])      # yellow centre -> red
    tree = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert yellow_fraction(tree, 0, 0, W, H) < MIN


def test_small_blurry_fire_still_passes():
    far = cv2.resize(cv2.resize(IMG, (W // 4, H // 4)), (W, H))     # ~4x farther / blurrier
    far = cv2.GaussianBlur(far, (5, 5), 0)
    assert yellow_fraction(far, 0, 0, W, H) > MIN


def test_dimmer_light_still_passes():
    dim = (IMG.astype(np.float32) * 0.7).astype(np.uint8)          # 30 % darker
    assert yellow_fraction(dim, 0, 0, W, H) > MIN


def test_background_is_not_yellow():
    assert yellow_fraction(IMG, 0, 0, 20, 20, shrink=1.0) < MIN
