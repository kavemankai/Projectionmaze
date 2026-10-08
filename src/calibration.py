import json
import os

import cv2
import numpy as np

DEFAULT_PATH = "calibration.json"


def target_points(w, h, inset=0.15):
    """Where the projector draws its 4 dots (game px): TL, TR, BR, BL."""
    ix, iy = w * inset, h * inset
    return [(ix, iy), (w - ix, iy), (w - ix, h - iy), (ix, h - iy)]


def compute_homography(cam_pts, game_w, game_h):
    """Camera px (4 clicked dots, same order as target_points) -> game px."""
    src = np.array(cam_pts, np.float32)
    dst = np.array(target_points(game_w, game_h), np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def save(path, H, game_w, game_h):
    with open(path, "w") as f:
        json.dump({"homography": np.asarray(H).tolist(), "game_w": game_w, "game_h": game_h}, f)


def load(path=DEFAULT_PATH):
    """Returns (H, (game_w, game_h)) or None if no file."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return np.array(d["homography"], np.float64), (d["game_w"], d["game_h"])
