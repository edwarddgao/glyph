#!/usr/bin/env python3
"""Write label-filtered training caches (#81's decoder-independent filter).

    python scripts/build_clean_caches.py

Outputs:
  data/canonical/futo_clean/train   futo/train minus flagged labels
  data/canonical/hws_clean/train    how_we_swipe/test, users with
                                    user_bucket < 0.7 (#8's train half), minus
                                    flagged labels -- so hws held-out users
                                    (>= 0.7) stay a clean evaluation set
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swipe_typing import cache
from swipe_typing.layout import KeyboardLayout
from swipe_typing.geomllm import GestureDP, GeomConfig
from eval_decoder import build_lexicon
from finetune_transfer import user_bucket

COST_THR = 6.0
_kb = _eng = _cfg = None


def _init(eng, thr):
    global _kb, _eng, _cfg, COST_THR
    _kb, _eng, _cfg, COST_THR = KeyboardLayout.qwerty(), eng, GeomConfig(time_weight=1.25), thr


def _judge(sw):
    from wordfreq import zipf_frequency
    w = sw.word
    if w not in _eng:
        return "non_english"
    if max(zipf_frequency(w, l) for l in ("es", "de", "fr", "pt", "it", "nl")) >= 3.0 and zipf_frequency(w, "en") < 2.5:
        return "foreign"
    path = float(np.linalg.norm(np.diff(sw.points, axis=0), axis=1).sum()) / (2 * float(_kb.radii[0][0]))
    if len(w) >= 4 and path < 0.5 * (len(w) - 1):
        return "aborted"
    if GestureDP(sw.points, sw.t, _kb, _cfg).word_cost(w) / len(w) > COST_THR:
        return "untraced"
    return "ok"


def build(src, dst, keep_fn, eng, workers=8, thr=COST_THR):
    sws = [sw for sw in cache.read(src) if keep_fn(sw)]
    with Pool(workers, initializer=_init, initargs=(eng, thr)) as pool:
        verdicts = pool.map(_judge, sws, chunksize=512)
    from collections import Counter
    c = Counter(verdicts)
    kept = [sw for sw, v in zip(sws, verdicts) if v == "ok"]
    cache.write(kept, dst)
    print(f"{src} -> {dst}: {len(sws):,} in, {len(kept):,} kept ({100 * (1 - len(kept) / len(sws)):.2f}% dropped) {dict(c)}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-thr", type=float, default=COST_THR,
                    help="drop if label geometry cost per letter exceeds this (6 = #81/#82)")
    ap.add_argument("--suffix", default="clean", help="output dirs futo_<suffix>/train, hws_<suffix>/train")
    ap.add_argument("--futo-only", action="store_true")
    args = ap.parse_args()
    root = Path("data/canonical")
    eng = set(build_lexicon("train+wf320k", root, KeyboardLayout.qwerty().letters, 1.0)._counts)
    build(root / "futo/train", root / f"futo_{args.suffix}/train", lambda sw: True, eng, thr=args.cost_thr)
    if not args.futo_only:
        build(root / "how_we_swipe/test", root / f"hws_{args.suffix}/train",
              lambda sw: user_bucket(sw.session) < 0.7, eng, thr=args.cost_thr)


if __name__ == "__main__":
    main()
