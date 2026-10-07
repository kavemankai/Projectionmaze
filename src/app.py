"""Milestones 2-4: shadows as physics geometry.

Monitor test:   python src/app.py [camera_index]
Projector:      python src/app.py [camera_index] --projector DISPLAY
                (needs calibration.json from src/calibrate.py)

Keys: click spawn ball | c clear | d toggle shadow outline | q/Esc quit
      [ ] threshold | - = min blob area | , . epsilon | s smoothing | e lock exposure
"""
import argparse

import calibration
from config import SimConfig
from mask import MaskConfig, ShadowMask
from world import World
import render


class App:
    def __init__(self, source, mask_cfg=None, sim_cfg=None, homography=None):
        self.source = source
        self.mcfg = mask_cfg or MaskConfig()
        self.scfg = sim_cfg or SimConfig()
        self.shadow = ShadowMask(self.mcfg, homography, (self.scfg.game_w, self.scfg.game_h))
        self.world = World(self.scfg, (self.mcfg.width, self.mcfg.height))
        self.contours = []
        self.debug = True
        self._since_rebuild = 1e9

    def tick(self, dt, screen=None):
        """One loop iteration: maybe rebuild shadow geometry, step physics, draw."""
        self._since_rebuild += dt
        if self._since_rebuild >= 1.0 / self.scfg.rebuild_hz:
            frame = self.source.read()
            if frame is not None:
                exclude = self.world.footprint()
                mask, self.contours = self.shadow.process(frame, exclude)
                self.world.set_geometry(mask, self.contours)
                self._since_rebuild = 0.0
        self.world.step(dt)
        if screen is not None:
            render.draw(screen, self.world, self.scfg,
                        self.contours if self.debug else None,
                        (self.world.sx, self.world.sy))


def main():
    import pygame
    from capture import Camera

    ap = argparse.ArgumentParser()
    ap.add_argument("camera", nargs="?", type=int, default=0)
    ap.add_argument("--projector", type=int, default=None, help="display index for fullscreen output")
    ap.add_argument("--calibration", default=calibration.DEFAULT_PATH)
    args = ap.parse_args()

    scfg = SimConfig()
    H = None
    if args.projector is not None:
        cal = calibration.load(args.calibration)
        if cal is None:
            raise SystemExit("No calibration file. Run src/calibrate.py first.")
        H, (scfg.game_w, scfg.game_h) = cal

    screen = render.make_window((scfg.game_w, scfg.game_h), args.projector)
    app = App(Camera(args.camera), sim_cfg=scfg, homography=H)
    app.debug = args.projector is None
    cfg = app.mcfg
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                app.world.spawn(*e.pos)
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_c:
                    app.world.clear_balls()
                elif k == pygame.K_d:
                    app.debug = not app.debug
                elif k == pygame.K_LEFTBRACKET:
                    cfg.thresh_mult = max(0.1, cfg.thresh_mult - 0.02)
                elif k == pygame.K_RIGHTBRACKET:
                    cfg.thresh_mult = min(0.95, cfg.thresh_mult + 0.02)
                elif k == pygame.K_MINUS:
                    cfg.min_area = max(0, cfg.min_area - 10)
                elif k == pygame.K_EQUALS:
                    cfg.min_area += 10
                elif k == pygame.K_COMMA:
                    cfg.epsilon = max(0.0, cfg.epsilon - 0.25)
                elif k == pygame.K_PERIOD:
                    cfg.epsilon += 0.25
                elif k == pygame.K_s:
                    cfg.smooth_frames = cfg.smooth_frames % 5 + 1
                elif k == pygame.K_e:
                    app.source.lock_exposure()
        app.tick(dt, screen)
        pygame.display.flip()
    app.source.close()
    pygame.quit()


if __name__ == "__main__":
    main()
