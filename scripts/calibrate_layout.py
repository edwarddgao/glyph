#!/usr/bin/env python3
"""Recover each corpus's keyboard geometry from its own touch data.

We cannot read key rectangles out of either dataset -- neither ships a layout
description. But touch-down points cluster tightly on the intended first key, so
the grid can be recovered by taking, for each letter, the median touch-down
position of every swipe whose target word starts with that letter.

Two things come out of this:

1. The horizontal grid. Both corpora match ``layout.key_center`` (10 columns,
   row insets 0 / 0.05 / 0.15) to within a few thousandths of keyboard width.

2. The vertical mapping. FUTO's canvas is exactly the 3 letter rows. How We
   Swipe's ``keyb_height`` also covers a 4th (space) row, so it needs an affine
   correction before the two are comparable.

A subtlety worth preserving: users touch systematically *below* key centers.
That bias is real behavioral signal, so we must not fit it away. We take only
the row *pitch* from the data (a difference of medians, so the bias cancels) and
set the origin by assuming How We Swipe carries the same bias FUTO does.

Usage:
    python scripts/calibrate_layout.py --hws data/how_we_swipe/swipelogs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import urllib.request
from collections import defaultdict

from swipe_typing import layout

HF_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=futo-org%2Fswipe.futo.org&config=swipe-1&split=train"
    "&offset={offset}&length=100"
)


def _row_centers(touchdowns: dict[str, list[tuple[float, float]]], min_n: int):
    """Median touch-down y per row, and per-letter x error vs the canonical grid."""
    per_row = defaultdict(list)
    x_err = []
    for r, row in enumerate(layout.ROWS):
        for c, ch in enumerate(row):
            pts = touchdowns.get(ch, [])
            if len(pts) < min_n:
                continue
            mx = statistics.median(p[0] for p in pts)
            my = statistics.median(p[1] for p in pts)
            per_row[r].append(my)
            x_err.append(abs(mx - ((c + 0.5) / 10 + layout.ROW_INSET[r])))
    centers = {r: statistics.median(v) for r, v in per_row.items() if v}
    return centers, x_err


def collect_futo(n_batches: int = 12) -> dict[str, list[tuple[float, float]]]:
    td = defaultdict(list)
    for i in range(n_batches):
        url = HF_ROWS.format(offset=i * 7919)
        with urllib.request.urlopen(url, timeout=60) as fh:
            payload = json.load(fh)
        for item in payload.get("rows", []):
            row = item["row"]
            word, pts = row["word"].lower(), row["data"]
            if not word.isalpha() or not pts:
                continue
            td[word[0]].append((pts[0]["x"], pts[0]["y"]))
    return td


def collect_hws(logdir: str, limit: int | None = None):
    td = defaultdict(list)
    aspects = []
    files = sorted(glob.glob(os.path.join(logdir, "*.log")))[:limit]
    for path in files:
        with open(path, errors="replace") as fh:
            fh.readline()  # header
            for line in fh:
                f = line.split(" ")
                if len(f) < 12 or f[4] != "touchstart":
                    continue
                word = f[10].lower()
                if not word.isalpha():
                    continue
                try:
                    kw, kh = float(f[2]), float(f[3])
                    x, y = float(f[5]) / kw, float(f[6]) / kh
                except (ValueError, ZeroDivisionError):
                    continue
                td[word[0]].append((x, y))
                aspects.append(kw / kh)
    return td, aspects, len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hws", default="data/how_we_swipe/swipelogs")
    ap.add_argument("--min-samples", type=int, default=30)
    args = ap.parse_args()

    print("== FUTO (swipe-1, sampled via HF datasets-server) ==")
    fu_centers, fu_xerr = _row_centers(collect_futo(), args.min_samples)
    for r in sorted(fu_centers):
        print(f"  row{r} observed y={fu_centers[r]:.4f}  ideal={(r + 0.5) / 3:.4f}")
    if fu_xerr:
        print(f"  max |x error| vs canonical grid: {max(fu_xerr):.4f}")

    # Touch bias, in units of row pitch. FUTO's canvas *is* the letter grid, so
    # its deviation from the ideal center is pure behavioral offset.
    bias = statistics.mean(fu_centers[r] - (r + 0.5) / 3 for r in fu_centers)
    bias_rows = bias * 3.0
    print(f"  downward touch bias: {bias:+.4f} canonical = {bias_rows:+.4f} row pitch")

    if not os.path.isdir(args.hws):
        print(f"\n[skip] How We Swipe logs not found at {args.hws}")
        return

    print("\n== How We Swipe ==")
    hw_td, aspects, n_files = collect_hws(args.hws)
    hw_centers, hw_xerr = _row_centers(hw_td, args.min_samples)
    for r in sorted(hw_centers):
        print(f"  row{r} observed y={hw_centers[r]:.4f} (fraction of keyb_height)")
    if hw_xerr:
        print(f"  max |x error| vs canonical grid: {max(hw_xerr):.4f}")

    rows = sorted(hw_centers)
    pitch = (hw_centers[rows[-1]] - hw_centers[rows[0]]) / (rows[-1] - rows[0])
    # De-bias row 0 using FUTO's measured offset, then walk out half a row.
    true_row0 = hw_centers[rows[0]] - rows[0] * pitch - bias_rows * pitch
    y0 = true_row0 - pitch / 2
    span = 3 * pitch

    print(f"\n  row pitch          = {pitch:.5f} of keyb_height")
    print(f"  letter-grid top    Y0   = {y0:.5f}")
    print(f"  letter-grid height SPAN = {span:.5f}")
    print(f"  implied extra rows below = {(1 - y0 - span) / pitch:.2f}")
    print(f"\n  paste into sources/how_we_swipe.py:")
    print(f"    _GRID_Y0, _GRID_SPAN = {y0:.5f}, {span:.5f}")
    if aspects:
        print(f"\n  median keyb aspect {statistics.median(aspects):.3f}"
              f" -> letter-grid aspect {statistics.median(aspects) / span:.3f}"
              f"   ({n_files} logs)")


if __name__ == "__main__":
    main()
