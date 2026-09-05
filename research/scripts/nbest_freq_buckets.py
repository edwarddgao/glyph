#!/usr/bin/env python3
"""Bucket n-best misses by the target word's training frequency.

The encoder carries an implicit LM: emissions favour letter sequences that
were common in training. This script makes that visible on any dump from
``dump_nbest.py`` — truth-in-beam rate climbs monotonically with training
count, and words the encoder never trained on land in the beam far less
often than common ones (64% vs 99.8% on futo/validation).

The split that matters for deciding whether to act on it: a missed word the
lexicon does not contain can never be emitted by the trie, so no encoder
change helps it. The actionable slice is *in-lexicon* words with few or no
training examples. On futo/validation that slice is ~0.5% of swipes (its
unseen tail is out-of-lexicon proper nouns); on How We Swipe it is ~3.9%
(everyday conversational vocabulary FUTO's transcription prompts
underrepresent) — which is why implicit-LM dilution pays cross-corpus and
does nothing in-domain.

With two dumps, prints the per-bucket comparison (baseline vs candidate).

Usage:
    python scripts/nbest_freq_buckets.py runs/mmi/nbest/futo_validation.npz
    python scripts/nbest_freq_buckets.py data/nbest/how_we_swipe_test.npz \
        runs/perm25e13/nbest/how_we_swipe_test.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon  # noqa: E402
from swipe_typing.layout import ALPHABET  # noqa: E402

BUCKETS = [(0, 0, "0 (unseen)"), (1, 5, "1-5"), (6, 20, "6-20"),
           (21, 100, "21-100"), (101, 1000, "101-1k"), (1001, 10**12, ">1k")]


def train_counts(cache: Path) -> Counter:
    ds = pads.dataset(cache / "futo/train", format="parquet")
    counts: Counter = Counter()
    for batch in ds.to_batches(columns=["word"]):
        counts.update(batch.column("word").to_pylist())
    return counts


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["words"], d["target"]


def profile(words, target, counts, lex):
    n = len(words)
    in_beam, top1 = target >= 0, target == 0
    wc = np.array([counts.get(w, 0) for w in words])
    bk = np.array([next(i for i, (lo, hi, _) in enumerate(BUCKETS)
                        if lo <= c <= hi) for c in wc])
    in_lex = np.array([w in lex for w in words])
    miss = ~in_beam

    print(f"n = {n:,}   truth-in-beam {in_beam.mean():.4f}   "
          f"top-1 {top1.mean():.4f}   misses {miss.sum():,}")
    hdr = (f"{'train count':>12} | {'swipes':>7} {'%swipes':>7} | "
           f"{'in-beam':>7} {'top-1':>6} | {'%miss':>6} {'overrep':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, (_, _, label) in enumerate(BUCKETS):
        m = bk == i
        if not m.sum():
            continue
        share_miss = (m & miss).sum() / max(miss.sum(), 1)
        print(f"{label:>12} | {m.sum():>7,} {100*m.mean():>6.1f}% | "
              f"{in_beam[m].mean():>7.4f} {top1[m].mean():>6.3f} | "
              f"{100*share_miss:>5.1f}% {share_miss/m.mean():>6.1f}x")
    act = miss & (wc <= 5) & in_lex
    print(f"actionable (in-lex, count<=5): {act.sum():,} misses "
          f"= {100*act.sum()/max(miss.sum(),1):.1f}% of misses "
          f"= {100*act.sum()/n:.2f}% of swipes")
    unfix = miss & ~in_lex
    print(f"unfixable (not in lexicon):    {unfix.sum():,} misses "
          f"= {100*unfix.sum()/max(miss.sum(),1):.1f}% of misses")


def compare(wb, tb, wp, tp, counts, lex):
    assert len(wb) == len(wp) and (wb == wp).all(), "dumps not aligned"
    n = len(wb)
    wc = np.array([counts.get(w, 0) for w in wb])
    bk = np.array([next(i for i, (lo, hi, _) in enumerate(BUCKETS)
                        if lo <= c <= hi) for c in wc])
    in_lex = np.array([w in lex for w in wb])
    ib_b, t1_b = tb >= 0, tb == 0
    ib_p, t1_p = tp >= 0, tp == 0

    print(f"n = {n:,}")
    for name, b, p in [("truth-in-beam", ib_b, ib_p), ("beam top-1", t1_b, t1_p)]:
        print(f"{name:>16}: {b.mean():.4f} -> {p.mean():.4f} "
              f"({p.mean()-b.mean():+.4f})")
    hdr = (f"{'train count':>12} | {'swipes':>7} | {'in-beam':>17} "
           f"{'delta':>7} | {'top-1':>17} {'delta':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, (_, _, label) in enumerate(BUCKETS):
        m = bk == i
        if not m.sum():
            continue
        print(f"{label:>12} | {m.sum():>7,} | "
              f"{ib_b[m].mean():.4f} -> {ib_p[m].mean():.4f} "
              f"{ib_p[m].mean()-ib_b[m].mean():>+7.4f} | "
              f"{t1_b[m].mean():.4f} -> {t1_p[m].mean():.4f} "
              f"{t1_p[m].mean()-t1_b[m].mean():>+7.4f}")
    m = (wc <= 5) & in_lex
    print(f"target slice (in-lex, count<=5): n={m.sum():,}  "
          f"in-beam {ib_b[m].mean():.4f} -> {ib_p[m].mean():.4f}  "
          f"top-1 {t1_b[m].mean():.4f} -> {t1_p[m].mean():.4f}")
    se = np.sqrt(t1_b.mean() * (1 - t1_b.mean()) / n)
    print(f"1 SE on top-1 = {se:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+",
                    help="one dump to profile, or baseline + candidate to compare")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--lexicon", default="train+wf320k")
    args = ap.parse_args()
    if len(args.dumps) > 2:
        ap.error("pass one or two dumps")

    cache = Path(args.cache)
    counts = train_counts(cache)
    lex = build_lexicon(args.lexicon, cache, ALPHABET, 1.0)

    if len(args.dumps) == 1:
        profile(*load(Path(args.dumps[0])), counts, lex)
    else:
        wb, tb = load(Path(args.dumps[0]))
        wp, tp = load(Path(args.dumps[1]))
        compare(wb, tb, wp, tp, counts, lex)


if __name__ == "__main__":
    main()
