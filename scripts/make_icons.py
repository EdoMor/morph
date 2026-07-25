#!/usr/bin/env python3
"""Rasterise webapp/icon.svg into the PNG sizes the manifest declares.

Pure Python — no Pillow, no cairosvg. The mark is simple enough (a rounded
square plus one polyline) that rasterising it directly is cheaper than adding a
dependency to a project meant to run on a phone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morph.tools.image import encode_png  # noqa: E402

# The "M" polyline from icon.svg, in a 512x512 viewBox.
POINTS = [(120, 360), (120, 152), (188, 248), (256, 152), (324, 248), (392, 152), (392, 360)]
STROKE = 40.0
CORNER = 112.0
BG = (11, 13, 16)
START = (110, 231, 183)  # #6ee7b7
END = (59, 130, 246)  # #3b82f6


def _distance_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    length_sq = vx * vx + vy * vy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    dx, dy = px - (ax + t * vx), py - (ay + t * vy)
    return (dx * dx + dy * dy) ** 0.5


def _inside_rounded_square(x: float, y: float, size: float, radius: float) -> bool:
    cx = min(max(x, radius), size - radius)
    cy = min(max(y, radius), size - radius)
    dx, dy = x - cx, y - cy
    return dx * dx + dy * dy <= radius * radius


def render(size: int) -> bytes:
    scale = size / 512.0
    stroke = STROKE * scale / 2.0
    radius = CORNER * scale
    points = [(x * scale, y * scale) for x, y in POINTS]
    segments = list(zip(points, points[1:]))

    rows: list[bytes] = []
    for y in range(size):
        row = bytearray()
        py = y + 0.5
        for x in range(size):
            px = x + 0.5
            if not _inside_rounded_square(px, py, float(size), radius):
                row.extend((0, 0, 0))  # transparent-ish edge; PNG here is opaque RGB
                continue

            distance = min(
                _distance_to_segment(px, py, a[0], a[1], b[0], b[1]) for a, b in segments
            )
            if distance <= stroke:
                t = px / size * 0.5 + py / size * 0.5  # diagonal gradient
                colour = tuple(
                    int(START[i] + (END[i] - START[i]) * t) for i in range(3)
                )
            elif distance <= stroke + 1.2:  # 1px feather so edges are not jagged
                blend = (distance - stroke) / 1.2
                t = px / size * 0.5 + py / size * 0.5
                stroke_colour = [int(START[i] + (END[i] - START[i]) * t) for i in range(3)]
                colour = tuple(
                    int(stroke_colour[i] * (1 - blend) + BG[i] * blend) for i in range(3)
                )
            else:
                colour = BG
            row.extend(colour)
        rows.append(bytes(row))
    return encode_png(rows, size, size)


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "webapp"
    for size in (192, 512):
        target = out / f"icon-{size}.png"
        target.write_bytes(render(size))
        print(f"wrote {target.relative_to(out.parent)} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
