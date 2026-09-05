#!/usr/bin/env python3
"""How much lexical ambiguity does gesture-only invariance create in principle?

    python scripts/probe_shape_collisions.py

For every word: ideal template = polyline through its key centers
(aspect-corrected), arclength-resampled. Four representations are compared:

    anchored            templates as-is (the canonical encoder's regime)
    translation-only    bbox center removed, true scale kept
    scale-only          unit-scaled about the bbox center, position kept
    shape               both removed (features.shape_normalize)

Collision criterion, expressed physically: w' collides with w if aligning w'
onto w (bbox proxy) leaves an RMS residual under eps key widths. Normalized
residuals are mapped back to key widths via w's own long side, so a tap --
whose long side is the scale floor -- collides with everything, which is the
correct semantics of scale invariance: a dot is any word scaled to zero.

Two masses are reported, weighted by the validation slice's word distribution:
words with *any* collider, and words that lose to a *more frequent* collider,
the top-1 upper bound for a perfect shape matcher with a unigram tie-break.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from swipe_typing import features  # noqa: E402
from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402

#: FUTO's letter grid, README "Canonical coordinate space".
ASPECT = 2.38
KEY_W = ASPECT / 10.0
N_PTS = 24


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--eps", nargs="+", type=float, default=[0.125, 0.25, 0.5],
                    help="alignment tolerance in key widths")
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lex = build_lexicon(args.lexicon, root, kb.letters, 1.0)
    counts = lex.counts()
    words = sorted(counts)
    print(f"lexicon: {len(words):,} words")

    def template(word: str) -> np.ndarray:
        pts = np.array([kb.center(c) for c in word], dtype=np.float32)
        pts[:, 0] *= ASPECT
        return features.resample(pts, None, n=N_PTS, mode="arclength")

    T = np.stack([template(w) for w in words])
    lo, hi = T.min(1), T.max(1)
    mid = (lo + hi) / 2.0
    ls = np.maximum((hi - lo).max(1), features.SHAPE_SCALE_FLOOR)

    # (name, flattened templates, per-word factor mapping residuals to
    # physical units)
    variants = [
        ("anchored", T, np.ones(len(words))),
        ("translation-only", T - mid[:, None, :], np.ones(len(words))),
        ("scale-only",
         mid[:, None, :] + (T - mid[:, None, :]) / ls[:, None, None], ls),
        ("shape", (T - mid[:, None, :]) / ls[:, None, None], ls),
    ]

    corpus = SwipeCorpus.load(root / args.split, kb.letters, limit=args.limit)
    val_counts = Counter(corpus.words)
    val_words = sorted(w for w in val_counts if w in counts)
    covered = sum(val_counts[w] for w in val_words)
    print(f"{args.split}: {len(corpus):,} swipes, "
          f"{covered / len(corpus):.1%} in lexicon")

    widx = {w: i for i, w in enumerate(words)}
    rows = np.array([widx[w] for w in val_words])
    w_of = np.array([val_counts[w] for w in val_words], dtype=np.float64)
    cnt = np.array([counts[w] for w in words], dtype=np.float64)

    for name, M3, phys in variants:
        M = M3.reshape(len(words), -1).astype(np.float32)
        sq = (M * M).sum(1)
        print(f"\n==== {name} ====")
        for eps_frac in args.eps:
            eps = eps_frac * KEY_W
            amb = lost = 0.0
            examples: list[tuple[str, str, float, int]] = []
            for s in range(0, len(rows), 512):
                idx = rows[s:s + 512]
                d2 = sq[idx][:, None] + sq[None, :] - 2.0 * (M[idx] @ M.T)
                d = np.sqrt(np.maximum(d2, 0) / N_PTS) * phys[idx][:, None]
                d[np.arange(len(idx)), idx] = np.inf
                coll = d < eps
                stronger = (coll & (cnt[None, :] > cnt[idx][:, None])).any(1)
                for j in np.flatnonzero(coll.any(1)):
                    g = s + j
                    amb += w_of[g]
                    if stronger[j]:
                        lost += w_of[g]
                        if len(examples) < 6 and w_of[g] >= 20:
                            k = int(np.argmin(d[j]))
                            examples.append((val_words[g], words[k],
                                             d[j][k] / KEY_W, int(w_of[g])))
            print(f"  eps={eps_frac} keyw: collider mass "
                  f"{amb / len(corpus):6.2%}, lost to a more frequent "
                  f"collider {lost / len(corpus):6.2%} "
                  f"-> ceiling <= {1 - lost / len(corpus):.2%}")
            for w, w2, dk, n in examples:
                print(f"      {w:<12} ~ {w2:<12} d={dk:.2f} keyw (val n={n})")


if __name__ == "__main__":
    main()
