#!/usr/bin/env python3
"""Render the Glyph app icon: the swipe trail of the word "glyph" on QWERTY.

    ../research/.venv/bin/python tools/make_icon.py [--out App/Assets.xcassets/AppIcon.appiconset]

One 1024×1024 PNG (iOS 17+ asset catalogs take a single size) — the accent
blue behind a white trail with a soft glow, drawn as a Catmull-Rom spline
through the key centres g → l → y → p → h, tapering from the touch-down
dot to the lift. Deterministic; re-run after changing the accent.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
INSET = [0.0, 0.05, 0.15]
ACCENT = (0x2B, 0x70, 0xF0)  # Color.glyph in OnboardingView.swift: (0.17, 0.44, 0.94)


def key_center(ch):
    for r, row in enumerate(ROWS):
        c = row.find(ch)
        if c >= 0:
            return (c + 0.5) / 10 + INSET[r], (r + 0.5) / 3
    raise KeyError(ch)


def catmull_rom(pts, n=60):
    out = []
    p = [pts[0]] + list(pts) + [pts[-1]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for k in range(n):
            t = k / n
            out.append(tuple(0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t * t + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t ** 3) for d in range(2)))
    out.append(tuple(pts[-1]))
    return out


def render(size=1024, word="glyph", scale=4):
    S = size * scale
    img = Image.new("RGB", (S, S), ACCENT)
    # gentle vertical gradient: a touch lighter at the top
    top = tuple(min(255, int(c * 1.12)) for c in ACCENT)
    grad = Image.linear_gradient("L").resize((S, S))
    img = Image.composite(Image.new("RGB", (S, S), top), img, grad.point(lambda v: 255 - v))
    # key centres of the word, mapped into a padded box; the keyboard is 10 wide × 3 tall
    pts = [key_center(ch) for ch in word]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, h = max(xs) - min(xs), (max(ys) - min(ys)) * (3 / 10 * (10 / 3))  # keep the layout's aspect: 1 key = size/10 horizontally, size/3 * (key h/w) vertically
    key_w = 0.62 * S / max(w / 0.1, 1e-6)  # trail spans 62% of the icon width
    key_w = min(key_w, 0.62 * S / 4)
    key_h = 0.40 * S / max((max(ys) - min(ys)) * 3, 1e-6)  # trail spans 40% of the icon height whatever rows the word touches
    cx, cy = S / 2, S / 2 + 0.02 * S
    def to_px(p):
        return (cx + (p[0] - (min(xs) + max(xs)) / 2) * 10 * key_w, cy + (p[1] - (min(ys) + max(ys)) / 2) * 3 * key_h)
    px = [to_px(p) for p in pts]
    curve = catmull_rom(px, 80)

    trail = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(trail)
    n = len(curve)
    for i in range(n - 1):
        t = i / (n - 1)
        r = (0.068 - 0.032 * t) * S / 2  # taper: fat at touch-down, thinner at lift
        (x0, y0), (x1, y1) = curve[i], curve[i + 1]
        d.line([(x0, y0), (x1, y1)], fill=255, width=int(2 * r))
        d.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)
    x0, y0 = curve[0]; r0 = 0.095 * S / 2
    d.ellipse([x0 - r0, y0 - r0, x0 + r0, y0 + r0], fill=255)
    glow = trail.filter(ImageFilter.GaussianBlur(S * 0.03)).point(lambda v: int(v * 0.55))
    img.paste(Image.new("RGB", (S, S), (255, 255, 255)), (0, 0), glow)
    img.paste(Image.new("RGB", (S, S), (255, 255, 255)), (0, 0), trail)
    return img.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "App/Assets.xcassets/AppIcon.appiconset"))
    ap.add_argument("--preview", default=None, help="also write a 180 px preview here")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img = render()
    img.save(out / "icon-1024.png", optimize=True)
    (out / "Contents.json").write_text(json.dumps({"images": [{"filename": "icon-1024.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}], "info": {"author": "xcode", "version": 1}}, indent=2) + "\n")
    (out.parent / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2) + "\n")
    if a.preview:
        img.resize((180, 180), Image.LANCZOS).save(a.preview)
    print(f"wrote {out / 'icon-1024.png'}")


if __name__ == "__main__":
    main()
