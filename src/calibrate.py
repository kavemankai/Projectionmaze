"""4-point calibration. Run: python src/calibrate.py [camera_index] [projector_display]

The projector shows 4 dots. Click them in the camera preview in order
(top-left, top-right, bottom-right, bottom-left). Enter saves, r resets, q quits.
"""
import sys

import cv2
import pygame

import calibration
from capture import Camera
from config import SimConfig


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    display = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = SimConfig()
    pygame.init()
    if display is None:
        screen = pygame.display.set_mode((cfg.game_w, cfg.game_h))
    else:
        screen = pygame.display.set_mode((cfg.game_w, cfg.game_h), pygame.FULLSCREEN, display=display)
    screen.fill((0, 0, 0))
    for p in calibration.target_points(cfg.game_w, cfg.game_h):
        pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), 14)
    pygame.display.flip()

    cam = Camera(index)
    pts = []
    win = "calibrate: click dots TL TR BR BL, Enter saves"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        pygame.event.pump()
        frame = cam.read()
        if frame is not None:
            for i, p in enumerate(pts):
                cv2.circle(frame, p, 6, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, frame)
        k = cv2.waitKey(10) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            pts.clear()
        elif k in (13, 10) and len(pts) == 4:
            H = calibration.compute_homography(pts, cfg.game_w, cfg.game_h)
            calibration.save(calibration.DEFAULT_PATH, H, cfg.game_w, cfg.game_h)
            print("saved", calibration.DEFAULT_PATH)
            break
    cam.close()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
