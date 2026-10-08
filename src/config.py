from dataclasses import dataclass


@dataclass
class SimConfig:
    game_w: int = 640            # game space = projector space, in px
    game_h: int = 480
    rebuild_hz: float = 20.0     # how often shadow geometry is swapped
    physics_hz: float = 120.0    # substep rate
    seg_radius: float = 3.0      # shadow segment thickness (game px)
    ball_radius: float = 10.0
    gravity: float = 900.0
    brightness: float = 0.35     # 0..1 scale on projected objects (dim = less feedback)
