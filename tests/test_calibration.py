import numpy as np
import cv2

import calibration


def test_homography_roundtrip(tmp_path):
    targets = calibration.target_points(640, 480)
    # Fake camera: scale 0.5 and offset (30, 20).
    cam = [(x * 0.5 + 30, y * 0.5 + 20) for x, y in targets]
    H = calibration.compute_homography(cam, 640, 480)
    for (cx, cy), (gx, gy) in zip(cam, targets):
        p = H @ np.array([cx, cy, 1.0])
        assert np.allclose(p[:2] / p[2], (gx, gy), atol=1e-3)
    path = tmp_path / "c.json"
    calibration.save(str(path), H, 640, 480)
    H2, size = calibration.load(str(path))
    assert size == (640, 480) and np.allclose(H, H2)


def test_load_missing(tmp_path):
    assert calibration.load(str(tmp_path / "none.json")) is None
