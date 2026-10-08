import pymunk


def contours_to_segments(contours, body, scale, radius, friction=0.8, elasticity=0.2):
    """Closed polylines (mask px) -> pymunk Segments on `body` (game px).

    scale is (sx, sy), mask px -> game px. Pixel centres sit at +0.5.
    """
    sx, sy = scale
    segs = []
    for cnt in contours:
        pts = [((x + 0.5) * sx, (y + 0.5) * sy) for x, y in cnt]
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            if a == b:
                continue
            s = pymunk.Segment(body, a, b, radius)
            s.friction = friction
            s.elasticity = elasticity
            segs.append(s)
    return segs
