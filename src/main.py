"""Milestone 1: webcam -> shadow mask -> contours, shown in a preview window.

Keys:  q quit | [ ] threshold -/+ | - = min blob area -/+ | , . epsilon -/+
       s smoothing frames +1 (wraps) | e try to lock exposure
"""
import sys
import cv2
import numpy as np

from capture import Camera
from mask import MaskConfig, ShadowMask


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cfg = MaskConfig()
    sm = ShadowMask(cfg)
    cam = Camera(index)

    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(10) == ord("q"):
                break
            continue

        mask, contours = sm.process(frame)

        # Draw everything at preview size
        h, w = frame.shape[:2]
        sx, sy = w / cfg.width, h / cfg.height
        view = frame.copy()
        for cnt in contours:
            pts = (cnt * np.array([sx, sy])).astype(np.int32)
            cv2.polylines(view, [pts], True, (0, 255, 0), 2)
        mask_big = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(mask_big, cv2.COLOR_GRAY2BGR)
        out = np.hstack([view, mask_bgr])

        txt = (f"thresh {cfg.thresh_mult:.2f}  min_area {cfg.min_area}  "
               f"eps {cfg.epsilon:.1f}  smooth {cfg.smooth_frames}  blobs {len(contours)}")
        cv2.putText(out, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow("shadow preview (left: camera, right: mask)", out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("["):
            cfg.thresh_mult = max(0.1, cfg.thresh_mult - 0.02)
        elif k == ord("]"):
            cfg.thresh_mult = min(0.95, cfg.thresh_mult + 0.02)
        elif k == ord("-"):
            cfg.min_area = max(0, cfg.min_area - 10)
        elif k == ord("="):
            cfg.min_area += 10
        elif k == ord(","):
            cfg.epsilon = max(0.0, cfg.epsilon - 0.25)
        elif k == ord("."):
            cfg.epsilon += 0.25
        elif k == ord("s"):
            cfg.smooth_frames = cfg.smooth_frames % 5 + 1
        elif k == ord("e"):
            cam.lock_exposure()

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
