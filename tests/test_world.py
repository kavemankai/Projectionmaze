import numpy as np
import cv2

from config import SimConfig
from world import World


def rect_mask(x0, y0, x1, y1):
    m = np.zeros((120, 160), np.uint8)
    m[y0:y1, x0:x1] = 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return m, [c.reshape(-1, 2) for c in cnts]


def test_ball_rests_on_shadow():
    w = World(SimConfig(), (160, 120))
    m, c = rect_mask(60, 80, 100, 100)  # game px x 240..400, y 320..400
    w.set_geometry(m, c)
    w.spawn(320, 100)
    for _ in range(600):
        w.step(1 / 60)
    body = w.balls[0][0]
    assert body.position.y < 320  # sat on top, did not fall through to floor (470)
    assert abs(body.position.x - 320) < 40


def test_ball_falls_to_floor_without_shadow():
    w = World(SimConfig(), (160, 120))
    w.spawn(320, 100)
    for _ in range(600):
        w.step(1 / 60)
    assert w.balls[0][0].position.y > 450


def test_push_out_when_shadow_swallows_ball():
    cfg = SimConfig()
    w = World(cfg, (160, 120))
    w.spawn(320, 360)  # game px; mask px (80, 90)
    m, c = rect_mask(60, 80, 100, 100)  # ball centre is inside
    w.set_geometry(m, c)
    body = w.balls[0][0]
    # Moved fully outside the rect in game px.
    inside_x = 240 <= body.position.x <= 400
    inside_y = 320 <= body.position.y <= 400
    assert not (inside_x and inside_y)
    assert body.velocity.length == 0


def test_footprint_marks_ball():
    w = World(SimConfig(), (160, 120))
    w.spawn(320, 240)
    fp = w.footprint()
    assert fp[60, 80] == 255 and fp[0, 0] == 0


def test_clear_balls():
    w = World(SimConfig(), (160, 120))
    w.spawn(100, 100)
    w.clear_balls()
    assert w.balls == []


def ring_mask():
    m = np.zeros((120, 160), np.uint8)
    m[60:100, 40:120] = 255
    m[70:90, 60:100] = 0  # hole, game px x 240..400, y 280..360
    cnts, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return m, [c.reshape(-1, 2) for c in cnts]


def test_ball_rests_inside_hole():
    w = World(SimConfig(), (160, 120))
    w.set_geometry(*ring_mask())
    w.spawn(320, 200)  # above the ring, lands on the top wall
    w.spawn(320, 320)  # inside the hole
    for _ in range(600):
        w.step(1 / 60)
    inside = w.balls[1][0]
    assert 280 < inside.position.y < 360 and 240 < inside.position.x < 400
