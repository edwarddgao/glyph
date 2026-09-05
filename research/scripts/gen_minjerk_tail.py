#!/usr/bin/env python3
"""Synthesize gestures for the lexicon's training-poor tail.

    python scripts/gen_minjerk_tail.py --per-word 2

#36 located the actionable slice: in-lexicon words with few or no training
examples (0.5% of futo/val, 3.9% of How We Swipe). Real data cannot cover
them by definition; a generator can. This samples every wf320k word whose
futo/train count is at or below a threshold, uniformly (coverage, not
naturalness — the trie supplies word identity, the gestures only need to
teach the geometry), through the domain-randomized min-jerk generator.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from swipe_typing import cache, minjerk
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model.lexicon import english_counts
from swipe_typing.model import SwipeCorpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--model", default="runs/minjerk_rand_model.json")
    ap.add_argument("--out", default="data/canonical/minjerk_tail/train")
    ap.add_argument("--wf", type=int, default=320_000)
    ap.add_argument("--max-train-count", type=int, default=2,
                    help="a word is 'tail' if futo/train has at most this "
                         "many gestures for it")
    ap.add_argument("--per-word", type=int, default=2)
    ap.add_argument("--max-word-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    model = minjerk.MinJerkModel.load(args.model)
    train_counts = Counter(
        SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters).words)
    general = english_counts(args.wf, alphabet=kb.letters)
    tail = sorted(
        w for w in general
        if train_counts[w] <= args.max_train_count
        and 2 <= len(w) <= args.max_word_len
    )
    print(f"wf{args.wf // 1000}k: {len(general):,} words, "
          f"tail (train count <= {args.max_train_count}): {len(tail):,}")

    rng = np.random.default_rng(args.seed)
    words = tail * args.per_word
    rng.shuffle(words)
    print(f"generating {len(words):,} swipes ...")
    t0 = time.time()
    swipes = minjerk.generate_many(model, words, kb, seed=args.seed)
    for sw in swipes:
        sw.split = "train"
    cache.write(swipes, args.out)
    print(f"wrote {len(swipes):,} to {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
