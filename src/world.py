import cv2
import numpy as np
import pymunk

from shapes import contours_to_segments


class World:
    """Physics space. Shadows are static segments that get swapped in bulk."""

    def __init__(self, cfg, mask_size):
        self.cfg = cfg
        self.mask_w, self.mask_h = mask_size
        self.sx = cfg.game_w / self.mask_w
        self.sy = cfg.game_h / self.mask_h
        self.space = pymunk.Space()
        self.space.gravity = (0, cfg.gravity)
        self.balls = []
        self._shadow = []
        self._acc = 0.0
        self._depth = None
        self._grad = None
        self._add_bounds()

    def _add_bounds(self):
        w, h = self.cfg.game_w, self.cfg.game_h
        body = self.space.static_body
        for a, b in [((0, 0), (0, h)), ((w, 0), (w, h)), ((0, h), (w, h))]:
            s = pymunk.Segment(body, a, b, 2)
            s.friction = 0.8
            self.space.add(s)

    def spawn(self, x, y):
        r = self.cfg.ball_radius
        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, r))
        body.position = (x, y)
        shape = pymunk.Circle(body, r)
        shape.friction = 0.6
        shape.elasticity = 0.4
        self.space.add(body, shape)
        self.balls.append((body, shape))

    def clear_balls(self):
        for body, shape in self.balls:
            self.space.remove(body, shape)
        self.balls = []

    def set_geometry(self, mask, contours):
        """Swap in new shadow geometry, then push out any ball it swallowed."""
        for s in self._shadow:
            self.space.remove(s)
        self._shadow = contours_to_segments(
            contours, self.space.static_body, (self.sx, self.sy), self.cfg.seg_radius)
        for s in self._shadow:
            self.space.add(s)
        self._depth = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        self._grad = (cv2.Sobel(self._depth, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(self._depth, cv2.CV_32F, 0, 1, ksize=3))
        self._push_out()

    def _push_out(self):
        """A ball whose centre is inside the shadow moves out along the
        shortest escape direction (down the distance-transform gradient)
        instead of being left for the solver to eject violently."""
        c = self.cfg
        for body, _ in self.balls:
            x = int(np.clip(body.position.x / self.sx, 0, self.mask_w - 1))
            y = int(np.clip(body.position.y / self.sy, 0, self.mask_h - 1))
            depth = float(self._depth[y, x])
            if depth <= 0:
                continue
            gx, gy = float(self._grad[0][y, x]), float(self._grad[1][y, x])
            n = (gx * gx + gy * gy) ** 0.5
            dx, dy = (-gx / n, -gy / n) if n > 1e-6 else (0.0, -1.0)
            dist = depth * (self.sx + self.sy) / 2 + c.ball_radius + c.seg_radius
            nx = float(np.clip(body.position.x + dx * dist, 0, c.game_w))
            ny = float(np.clip(body.position.y + dy * dist, 0, c.game_h))
            body.position = (nx, ny)
            body.velocity = (0, 0)

    def step(self, dt):
        """Advance by dt seconds in fixed substeps."""
        h = 1.0 / self.cfg.physics_hz
        self._acc = min(self._acc + dt, 0.1)
        while self._acc >= h:
            self.space.step(h)
            self._acc -= h

    def footprint(self, pad=2.0):
        """Mask-resolution uint8 image (255 where objects are drawn)."""
        fp = np.zeros((self.mask_h, self.mask_w), np.uint8)
        for body, shape in self.balls:
            cx = int(body.position.x / self.sx)
            cy = int(body.position.y / self.sy)
            r = int((shape.radius + pad) / self.sx) + 1
            cv2.circle(fp, (cx, cy), r, 255, -1)
        return fp
