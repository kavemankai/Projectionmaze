import cv2
import numpy as np
import pygame


def make_window(size, display=None):
    """Windowed for monitor testing, fullscreen on `display` for the projector."""
    pygame.init()
    if display is None:
        return pygame.display.set_mode(size)
    return pygame.display.set_mode(size, pygame.FULLSCREEN, display=display)


def draw(screen, world, cfg, contours=None, scale=None):
    """Black background, dim objects. Black pixels emit no light, so nothing is
    drawn for the shadow itself unless `contours` is given (monitor debug view)."""
    screen.fill((0, 0, 0))
    v = int(255 * cfg.brightness)
    for body, shape in world.balls:
        pygame.draw.circle(screen, (v, v, v), (int(body.position.x), int(body.position.y)),
                           int(shape.radius))
    if contours is not None:
        sx, sy = scale
        for cnt in contours:
            pts = [((x + 0.5) * sx, (y + 0.5) * sy) for x, y in cnt]
            pygame.draw.polygon(screen, (0, 160, 0), pts, 1)


def surface_to_bgr(screen):
    arr = pygame.surfarray.array3d(screen)  # (w, h, 3) RGB
    return cv2.cvtColor(np.transpose(arr, (1, 0, 2)), cv2.COLOR_RGB2BGR)
