from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class MaskConfig:
    width: int = 160          # mask resolution
    height: int = 120
    thresh_mult: float = 0.60  # pixel is shadow if gray < mean * thresh_mult
    min_area: int = 40         # drop blobs smaller than this (mask pixels)
    epsilon: float = 1.5       # approxPolyDP tolerance (mask pixels)
    smooth_frames: int = 2     # OR together the last N masks to cut flicker
    blur: int = 5              # odd kernel size


class ShadowMask:
    def __init__(self, cfg: MaskConfig, homography=None, game_size=(640, 480)):
        """homography: 3x3 camera->game matrix. If set, frames are warped to
        game_size before thresholding so mask pixels line up with game pixels."""
        self.cfg = cfg
        self.homography = homography
        self.game_size = game_size
        self._history = []
        self._prev = None

    def process(self, frame, exclude=None):
        """Returns (mask, contours). mask is uint8 0/255, contours are Nx2 int arrays.

        exclude: optional uint8 mask (same size as output) of pixels where the
        projector is drawing objects. Light from those objects brightens real
        shadow and would punch holes in it, so inside the footprint the previous
        mask wins.
        """
        c = self.cfg
        if self.homography is not None:
            frame = cv2.warpPerspective(frame, self.homography, self.game_size)
        small = cv2.resize(frame, (c.width, c.height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        k = c.blur | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

        cutoff = float(gray.mean()) * c.thresh_mult
        mask = (gray < cutoff).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self._history.append(mask)
        self._history = self._history[-max(1, c.smooth_frames):]
        mask = np.bitwise_or.reduce(self._history)
        if exclude is not None and self._prev is not None:
            mask = mask | (exclude & self._prev)
        self._prev = mask

        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        for cnt in found:
            if cv2.contourArea(cnt) < c.min_area:
                continue
            approx = cv2.approxPolyDP(cnt, c.epsilon, True)
            if len(approx) >= 3:
                contours.append(approx.reshape(-1, 2))
        return mask, contours
