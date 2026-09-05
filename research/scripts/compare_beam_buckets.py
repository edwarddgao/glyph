#!/usr/bin/env python3
"""Where does a first-pass delta live? Paired beam predictions, bucketed.

    python scripts/compare_beam_buckets.py runs/ar_clean_s1/preds_how_we_swipe_test.npz \
        runs/ar_enc_n128/preds_how_we_swipe_test.npz --split how_we_swipe/test

Reads two `eval_ar_decoder.py --save-preds` files for the same split (same
corpus order, asserted on refs), reloads the corpus for per-swipe raw point
count, duration and word length, and prints the paired delta plus fixed /
broken counts per bucket. Written for #83 to ask whether the 128-frame
encoder's cross-corpus gain was a long-gesture story (it was not).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from swipe_typing.layout import ALPHABET
from swipe_typing.model import SwipeCorpus

BUCKETS = {
    "raw points": [0, 32, 64, 128, 10**9],
    "duration ms": [0, 500, 1000, 2000, 10**9],
    "word len": [0, 3, 5, 8, 100],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", required=True)
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--lag", default="beam")
    args = ap.parse_args()

    corpus = SwipeCorpus.load(Path(args.cache) / args.split, ALPHABET,
                              limit=args.limit)
    A = np.load(args.a, allow_pickle=True)
    B = np.load(args.b, allow_pickle=True)
    assert (A["refs"] == B["refs"]).all(), "different splits"
    assert len(A["refs"]) == len(corpus), "limit does not match the preds"
    assert (A["refs"] == np.array(corpus.words, dtype=object)).all()
    ok_a, ok_b = A[args.lag] == A["refs"], B[args.lag] == B["refs"]

    npts = np.diff(corpus.offsets)
    dur = np.array([corpus.times(i)[-1] - corpus.times(i)[0]
                    for i in range(len(corpus))])
    wl = np.array([len(w) for w in corpus.words])
    print(f"== {args.split}  n={len(corpus):,}  {args.lag}: "
          f"{ok_a.mean():.4f} -> {ok_b.mean():.4f} "
          f"({100 * (ok_b.mean() - ok_a.mean()):+.2f})")
    print(f"  raw points median {np.median(npts):.0f} "
          f"(p90 {np.percentile(npts, 90):.0f}), duration median "
          f"{np.median(dur):.0f} ms (p90 {np.percentile(dur, 90):.0f})")
    for name, v in [("raw points", npts), ("duration ms", dur),
                    ("word len", wl)]:
        edges = BUCKETS[name]
        print(f"  by {name}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            if not m.any():
                continue
            win = int((~ok_a & ok_b & m).sum())
            loss = int((ok_a & ~ok_b & m).sum())
            print(f"    [{lo},{hi if hi < 10**9 else 'inf'}): n={m.sum():5d}  "
                  f"a {ok_a[m].mean():.4f}  b {ok_b[m].mean():.4f}  "
                  f"delta {100 * (ok_b[m].mean() - ok_a[m].mean()):+.2f}  "
                  f"({win} fixed / {loss} broken)")


if __name__ == "__main__":
    main()
