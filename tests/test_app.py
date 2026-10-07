import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

import render
from app import App
from config import SimConfig


class FakeCam:
    def __init__(self):
        f = np.full((480, 640, 3), 200, np.uint8)
        f[320:400, 240:400] = 20
        self.f = f

    def read(self):
        return self.f.copy()


def test_end_to_end_ball_lands_on_shadow():
    scfg = SimConfig()
    screen = render.make_window((scfg.game_w, scfg.game_h))
    app = App(FakeCam(), sim_cfg=scfg)
    app.world.spawn(320, 100)
    for _ in range(300):
        app.tick(1 / 60, screen)
    assert app.contours
    assert app.world.balls[0][0].position.y < 320
    # projector view: no outline, objects dim
    app.debug = False
    app.tick(1 / 60, screen)
    img = render.surface_to_bgr(screen)
    assert 0 < img.max() <= int(255 * scfg.brightness)
    pygame.quit()
