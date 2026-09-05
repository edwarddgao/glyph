#!/usr/bin/env python3
"""Which word's swipe gesture is most unique?

    .venv/bin/python scripts/gesture_uniqueness.py [--top 20000] [--names trace,skate,...]

For each of the N most frequent lexicon words, the ideal gesture is the polyline
through its key centres (repeated letters collapsed), resampled to 32 points by
arclength in key units. Uniqueness is the distance to the nearest *other* word's
ideal gesture (mean point-to-point distance, SHARK2's location channel) — the
larger, the less any other word looks like it on the keyboard. Prints the most
unique words overall and the nearest neighbour of each candidate name.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from swipe_typing.layout import ALPHABET, KeyboardLayout
from eval_decoder import build_lexicon

N_PTS = 32

def ideal(word, kb):
    pts = [kb.centers[kb.index(c)] / kb.radii[0] for c in word]
    col = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if not np.allclose(a, p)]
    P = np.asarray(col, np.float64)
    if len(P) == 1: return np.repeat(P, N_PTS, axis=0)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); u = np.concatenate([[0], np.cumsum(seg)])
    g = np.linspace(0, u[-1], N_PTS)
    return np.stack([np.interp(g, u, P[:, 0]), np.interp(g, u, P[:, 1])], 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20000)
    ap.add_argument("--names", default="trace,skate,glyph,swipe,glide,flick,slide,drift,streak,scribe,wisp,loop,dash,flow,stroke,squiggle,sweep,swish,quill,ink,graze,glissade,zigzag,scrawl,doodle,swoop,skim,quirk,jinx,lynx")
    ap.add_argument("--min-len", type=int, default=4)
    a = ap.parse_args()
    kb = KeyboardLayout.qwerty()
    lex = build_lexicon("train+wf320k", ROOT / "data/canonical", ALPHABET, 1.0)
    words = [w for w, _ in lex._counts.most_common() if w.isalpha()][: a.top]
    names = [n for n in a.names.split(",") if n and all(c in ALPHABET for c in n)]
    for n in names:
        if n not in words: words.append(n)
    idx = {w: i for i, w in enumerate(words)}
    G = np.stack([ideal(w, kb) for w in words]).astype(np.float32).reshape(len(words), -1)   # (W, 64)
    sq = (G * G).sum(1)
    nn_d = np.full(len(words), np.inf, np.float32); nn_i = np.zeros(len(words), np.int64)
    for s in range(0, len(words), 512):
        D = sq[s:s + 512, None] + sq[None, :] - 2 * G[s:s + 512] @ G.T
        for r in range(D.shape[0]): D[r, s + r] = np.inf
        j = D.argmin(1); nn_d[s:s + 512] = np.sqrt(np.maximum(D[np.arange(D.shape[0]), j], 0) / N_PTS); nn_i[s:s + 512] = j
    print(f"{len(words)} words; distance = RMS point gap to the nearest other word's ideal gesture, in key widths (x half-extent units)\n")
    print("most unique gestures among common words (length >= %d, top 8k by frequency):" % a.min_len)
    order = [i for i in np.argsort(-nn_d) if len(words[i]) >= a.min_len and i < 8000]
    for i in order[:25]:
        print(f"  {words[i]:<12} {nn_d[i]:5.2f}   nearest: {words[nn_i[i]]}")
    print("\ncandidate names:")
    for n in sorted(names, key=lambda n: -nn_d[idx[n]]):
        i = idx[n]; print(f"  {n:<10} {nn_d[i]:5.2f}   nearest: {words[nn_i[i]]}   (rank {int((nn_d > nn_d[i]).sum()) + 1} of {len(words)})")

if __name__ == "__main__":
    main()
