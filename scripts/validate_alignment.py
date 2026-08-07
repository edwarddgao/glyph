#!/usr/bin/env python3
"""Check that every cached corpus lands in the same coordinate space.

The whole point of the canonical schema is that a model trained on one corpus
can be evaluated on another. That only holds if their geometry agrees. This
compares, per corpus, the median touch-down position for each first letter
against the canonical key center -- and against each other.

If a loader's coordinate mapping regresses, this is where it shows up.

Usage:
    python scripts/validate_alignment.py --cache data/canonical
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from swipe_typing import cache
from swipe_typing.layout import KeyboardLayout


def touchdown_medians(path: Path, min_n: int = 30) -> dict[str, tuple[float, float]]:
    pts = defaultdict(list)
    for sw in cache.read(path, columns=["word", "x", "y", "t", "source",
                                        "split", "session", "aspect",
                                        "sentence", "word_idx", "flagged"]):
        if sw.word:
            pts[sw.word[0]].append((float(sw.x[0]), float(sw.y[0])))
    return {
        ch: (statistics.median(p[0] for p in v), statistics.median(p[1] for p in v))
        for ch, v in pts.items() if len(v) >= min_n
    }


def aspect_median(path: Path) -> float:
    vals = [sw.aspect for sw in cache.read(path) if sw.aspect > 0]
    return statistics.median(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="max offset from key center, in key half-widths")
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    corpora = [d for d in sorted(root.glob("*/*")) if d.is_dir()]
    if not corpora:
        raise SystemExit(f"no cached corpora under {root}")

    meds = {}
    print(f"{'corpus':<28} {'letters':>8} {'med |dx|':>9} {'med |dy|':>9} "
          f"{'max':>7} {'aspect':>7}")
    for d in corpora:
        name = str(d.relative_to(root))
        m = touchdown_medians(d)
        meds[name] = m
        dx, dy, worst = [], [], 0.0
        for ch, (x, y) in m.items():
            cx, cy = kb.center(ch)
            rx, ry = kb.radii[kb.index(ch)]
            dx.append(abs(x - cx) / rx)
            dy.append(abs(y - cy) / ry)
            worst = max(worst, dx[-1], dy[-1])
        print(f"{name:<28} {len(m):>8} {statistics.median(dx):>9.3f} "
              f"{statistics.median(dy):>9.3f} {worst:>7.3f} "
              f"{aspect_median(d):>7.3f}")

    names = list(meds)
    if len(names) > 1:
        print("\npairwise agreement (median offset between corpora, "
              "in key half-widths):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = set(meds[a]) & set(meds[b])
                if not shared:
                    continue
                d = []
                for ch in shared:
                    rx, ry = kb.radii[kb.index(ch)]
                    d.append(np.hypot((meds[a][ch][0] - meds[b][ch][0]) / rx,
                                      (meds[a][ch][1] - meds[b][ch][1]) / ry))
                print(f"  {a} vs {b}: {statistics.median(d):.3f} "
                      f"over {len(shared)} letters")

    print("\nNote: a consistent small positive dy across corpora is expected -- "
          "users\ntouch below key centers. That is behavioral signal, not "
          "misalignment.")


if __name__ == "__main__":
    main()
