#!/usr/bin/env python3
"""Measure key rectangles in a keyboard screenshot and diff two keyboards.

    python tools/measure_layout.py native.png [ours.png] [--width 402]

Finds key faces as uniform blobs (white in light mode, #3D3D3D in dark) in the
bottom third of the screen, prints (x, y, w, h) in points, and — given two
screenshots — pairs keys by nearest center and prints the per-key deviation.
Also reports the glyph box (dark or light pixels) inside each key, so font
size and baseline can be matched, not just geometry.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def load(path: str, width_pt: float):
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    scale = im.shape[1] / width_pt
    return im, scale


def key_rects(im, scale, region_pt=340):
    H = im.shape[0]
    y0 = int(H - region_pt * scale)
    sub = im[y0:]
    mn, mx = sub.min(axis=2), sub.max(axis=2)
    bg = im[int(H - 150 * scale), int(2 * scale)]
    dark_mode = bg.mean() < 100
    if not dark_mode:
        face = (mn > 240)
    else:
        # Key faces are the most common colour in the region that is not the
        # background (Apple #3D3D3D, Gboard similar, SwiftKey #777777).
        q = (sub // 4).reshape(-1, 3)
        notbg = np.abs(sub - bg).max(axis=2).reshape(-1) > 12
        vals, counts = np.unique(q[notbg], axis=0, return_counts=True)
        face_col = vals[counts.argmax()] * 4 + 2
        face = np.abs(sub - face_col).max(axis=2) < 8
    lab, _ = ndimage.label(face)
    rects = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h < 25 * scale or w < 20 * scale or h > 80 * scale or w > 300 * scale:
            continue
        if (lab[sl] == i + 1).sum() < 0.8 * h * w:
            continue
        rects.append((sl[1].start / scale, (sl[0].start + y0) / scale, w / scale, h / scale))
    rects.sort(key=lambda r: (round(r[1] / 10), r[0]))
    return rects, dark_mode


def glyph_box(im, scale, rect, dark_mode):
    x, y, w, h = rect
    reg = im[int(y * scale):int((y + h) * scale), int(x * scale):int((x + w) * scale)]
    ink = (reg.max(axis=2) < 120) if not dark_mode else (reg.min(axis=2) > 200)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return (cols.min() / scale, rows.min() / scale, (cols.max() - cols.min() + 1) / scale,
            (rows.max() - rows.min() + 1) / scale)  # relative to key origin


def describe(path, width_pt):
    im, scale = load(path, width_pt)
    rects, dark = key_rects(im, scale)
    print(f"{path}: {len(rects)} keys ({'dark' if dark else 'light'}), "
          f"bg {tuple(im[int(im.shape[0] - 150 * scale), int(2 * scale)])}")
    out = []
    for r in rects:
        g = glyph_box(im, scale, r, dark)
        out.append((r, g))
        gs = "  glyph x%.1f y%.1f w%.1f h%.1f" % g if g else "  (blank)"
        print("  key x%6.1f y%6.1f w%6.1f h%5.1f%s" % (r + (gs,)))
    return out


def letter_grid(rects):
    """Replay Grid from key rects: the three letter rows are the rows with 10, 9 and 7
    keys; cells are key + gap, so the grid's left edge is the q key's left minus half
    the horizontal gap and its top is the first row's top minus half the vertical gap."""
    rows = {}
    for r in rects:
        rows.setdefault(round(r[1] / 8), []).append(r)
    rows = [sorted(v, key=lambda r: r[0]) for _, v in sorted(rows.items())]
    letter = [r for r in rows if len(r) in (10, 9, 7)]
    top10 = next(r for r in letter if len(r) == 10)
    gap = np.mean([b[0] - (a[0] + a[2]) for a, b in zip(top10, top10[1:])])
    pitch = np.mean([b[0] - a[0] for a, b in zip(top10, top10[1:])])
    ys = [np.mean([k[1] for k in r]) for r in letter]
    row_pitch = np.mean(np.diff(ys))
    h = np.mean([k[3] for k in top10])
    return dict(left=top10[0][0] - gap / 2, width=10 * pitch, top=ys[0] - (row_pitch - h) / 2, rowPitch=row_pitch)


def grid_string(path, width_pt):
    im, scale = load(path, width_pt)
    rects, _ = key_rects(im, scale)
    g = letter_grid(rects)
    return "%.2f,%.2f,%.2f,%.2f" % (g["left"], g["width"], g["top"], g["rowPitch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("native")
    ap.add_argument("ours", nargs="?")
    ap.add_argument("--width", type=float, default=402.0, help="screen width in points")
    ap.add_argument("--tol", type=float, default=0.5, help="pt tolerance for the pass/fail line")
    ap.add_argument("--grid", action="store_true", help="print the replay Grid (left,width,top,rowPitch) of the letter rows")
    a = ap.parse_args()
    if a.grid:
        print(grid_string(a.native, a.width)); return
    nat = describe(a.native, a.width)
    if not a.ours:
        return
    our = describe(a.ours, a.width)
    print("\n== pairing by nearest center (dx dy dw dh in pt; glyph ddx ddy ddw ddh) ==")
    worst = 0.0
    unmatched = 0
    for (r, g) in nat:
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        best = min(our, key=lambda o: (o[0][0] + o[0][2] / 2 - cx) ** 2 + (o[0][1] + o[0][3] / 2 - cy) ** 2)
        o, og = best
        d = tuple(o[i] - r[i] for i in range(4))
        dist = max(abs(v) for v in d)
        if abs(o[0] + o[2] / 2 - cx) > 15 or abs(o[1] + o[3] / 2 - cy) > 15:
            unmatched += 1
            print("  native key at (%.1f, %.1f) has no counterpart" % (cx, cy))
            continue
        worst = max(worst, dist)
        gd = ""
        if g and og:
            gd = "   glyph %+5.1f %+5.1f %+5.1f %+5.1f" % tuple(og[i] - g[i] for i in range(4))
        elif g != og:
            gd = "   glyph mismatch (%s vs %s)" % ("ink" if g else "blank", "ink" if og else "blank")
        flag = "" if dist <= a.tol else "  <-- off"
        print("  key (%6.1f,%6.1f)  %+5.1f %+5.1f %+5.1f %+5.1f%s%s" % ((cx, cy) + d + (gd, flag)))
    print(f"\nworst key-rect deviation {worst:.2f} pt, unmatched {unmatched}, "
          f"{'PASS' if worst <= a.tol and unmatched == 0 else 'FAIL'} at tol {a.tol}")
    sys.exit(0 if worst <= a.tol and unmatched == 0 else 1)


if __name__ == "__main__":
    main()
