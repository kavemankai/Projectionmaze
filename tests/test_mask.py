import numpy as np
import cv2

from mask import MaskConfig, ShadowMask


def frame_with_rect(x0=200, y0=150, x1=360, y1=300, bg=200, fg=20):
    f = np.full((480, 640, 3), bg, np.uint8)
    f[y0:y1, x0:x1] = fg
    return f


def test_single_rect_gives_one_contour():
    sm = ShadowMask(MaskConfig(smooth_frames=1))
    mask, contours = sm.process(frame_with_rect())
    assert len(contours) == 1
    xs, ys = contours[0][:, 0], contours[0][:, 1]
    # 640x480 -> 160x120 is /4: rect spans x 50..90, y 37..75
    assert abs(xs.min() - 50) <= 2 and abs(xs.max() - 89) <= 2
    assert abs(ys.min() - 37) <= 2 and abs(ys.max() - 74) <= 2


def test_blank_frame_gives_nothing():
    sm = ShadowMask(MaskConfig())
    _, contours = sm.process(np.full((480, 640, 3), 180, np.uint8))
    assert contours == []


def test_small_blob_dropped():
    sm = ShadowMask(MaskConfig(smooth_frames=1, min_area=40))
    f = frame_with_rect(100, 100, 112, 112)  # 3x3 mask px
    _, contours = sm.process(f)
    assert contours == []


def test_smoothing_ors_recent_masks():
    sm = ShadowMask(MaskConfig(smooth_frames=2))
    sm.process(frame_with_rect())
    mask, _ = sm.process(np.full((480, 640, 3), 200, np.uint8))
    assert mask.any()  # previous frame's shadow still present
    mask, _ = sm.process(np.full((480, 640, 3), 200, np.uint8))
    assert not mask.any()


def test_homography_warps_to_game_space():
    # Camera sees the scene shifted right by 100 px; H undoes the shift.
    H = np.array([[1, 0, -100], [0, 1, 0], [0, 0, 1]], np.float64)
    sm = ShadowMask(MaskConfig(smooth_frames=1), H, (640, 480))
    _, contours = sm.process(frame_with_rect(300, 150, 460, 300))
    assert abs(contours[0][:, 0].min() - 50) <= 2  # (300-100)/4


def test_exclude_keeps_previous_mask_under_objects():
    sm = ShadowMask(MaskConfig(smooth_frames=1))
    mask1, _ = sm.process(frame_with_rect())
    # Projected light fills a hole in the camera image inside the rect.
    f = frame_with_rect()
    f[200:240, 260:300] = 200
    exclude = np.zeros((120, 160), np.uint8)
    exclude[45:65, 60:80] = 255
    mask2, _ = sm.process(f, exclude)
    assert mask2[55, 70] == 255
    sm2 = ShadowMask(MaskConfig(smooth_frames=1))
    sm2.process(frame_with_rect())
    mask3, _ = sm2.process(f)
    assert mask3[55, 70] == 0  # without exclude the hole appears


def ring_frame(hole=(260, 200, 300, 240)):
    f = frame_with_rect()
    x0, y0, x1, y1 = hole
    f[y0:y1, x0:x1] = 200
    return f


def test_hole_kept_as_second_contour():
    _, contours = ShadowMask(MaskConfig(smooth_frames=1)).process(ring_frame())
    assert len(contours) == 2


def test_holes_off_gives_solid_blob():
    _, contours = ShadowMask(MaskConfig(smooth_frames=1, holes=False)).process(ring_frame())
    assert len(contours) == 1


def test_island_inside_hole_kept():
    f = ring_frame((220, 160, 340, 290))
    f[190:260, 250:310] = 20
    _, contours = ShadowMask(MaskConfig(smooth_frames=1)).process(f)
    assert len(contours) == 3


def test_small_hole_filled():
    f = ring_frame((260, 200, 268, 208))  # 2x2 mask px
    _, contours = ShadowMask(MaskConfig(smooth_frames=1)).process(f)
    assert len(contours) == 1
