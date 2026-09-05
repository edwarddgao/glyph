#!/usr/bin/env python3
"""Out-of-list candidates from the training-free geometric trie beam (#73).

The fused search can only rank what the AR beam surfaces; on hws 7.2% of
swipes have no true word anywhere in the 24-deep list. This script re-searches
the same lexicon with the *other* acoustic channel — the GestureDP alignment
cost — and keeps the best new words, so the fused search gets candidates the
trained beam pruned. Each proposal is scored under both channels:

    (word, ar_logp, unigram_logp, length, geom_cost)

ar_logp is the AR decoder's teacher-forced score, the same quantity the
bundle lists carry. #73's first cell showed why that score must not be the
*veto*: proposals are exactly the words the AR model prunes, so ranking them
by ar_logp buries them (the mirror of #70's bias relocation). It is kept so
eval_geom_fusion.py can score every candidate with one formula.

    python scripts/gen_geom_proposals.py --bundle fused_base_hws.pkl \
        --data data/canonical/how_we_swipe/test --out geom_props_hws.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_ar_decoder import load_ar  # noqa: E402
from eval_decoder import build_lexicon, pick_device  # noqa: E402
from eval_geom_trie import build_trie, decode  # noqa: E402
from swipe_typing.geomllm import GeomConfig, GestureDP  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (SwipeCorpus, SwipeDataset,  # noqa: E402
                                make_loader)
from swipe_typing.model.ar import score_words  # noqa: E402


def calibrate_offset(corpus: SwipeCorpus, kb: KeyboardLayout,
                     n: int) -> tuple[float, float]:
    """Label-free touch-offset estimate from swipe endpoints (#70's recipe)."""
    res = []
    for i in range(min(n, len(corpus))):
        pts = corpus.points(i)
        for p in (pts[0], pts[-1]):
            d = (p - kb.centers) / kb.radii
            res.append(p - kb.centers[np.argmin((d * d).sum(-1))])
    return tuple(np.median(np.asarray(res), axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="fused_base_hws.pkl")
    ap.add_argument("--data", default="data/canonical/how_we_swipe/test")
    ap.add_argument("--checkpoint", default="runs/ar_full/ar_decoder.pt")
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--out", default="geom_props_hws.pkl")
    ap.add_argument("--m", type=int, default=24,
                    help="list depth the fused search uses; only words "
                         "outside it are proposed")
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--beam", type=int, default=1500)
    ap.add_argument("--uni-weight", type=float, default=0.5,
                    help="unigram weight in the proposer's own ranking "
                         "(eval_geom_trie's prior; the fused search re-ranks)")
    ap.add_argument("--calibrate", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    lists, refs = bundle["lists"], bundle["refs"]
    n = len(refs)

    corpus = SwipeCorpus.load(args.data, ALPHABET, limit=n)
    assert list(corpus.words) == list(refs), "bundle/corpus order mismatch"
    kb = KeyboardLayout.qwerty()
    offset = calibrate_offset(corpus, kb, args.calibrate)
    print(f"touch offset: ({offset[0]:+.4f}, {offset[1]:+.4f})")
    gcfg = GeomConfig(offset=offset)

    lex = build_lexicon(args.lexicon, Path("data/canonical"), ALPHABET, 1.0)
    root = build_trie(w for w in lex._counts)
    print(f"lexicon: {len(lex)} words")

    proposals: list[list[tuple[str, float, float]]] = []
    t0 = time.time()
    for i in range(n):
        dp = GestureDP(corpus.points(i), corpus.times(i), kb, gcfg)
        cands = decode(dp, root, kb, args.beam)[:400]
        have = {w for w, *_ in lists[i][:args.m]}
        scored = [(w, g, lex.logp(w)) for w, g in cands if w not in have]
        scored.sort(key=lambda t: t[1] - args.uni_weight * t[2])
        proposals.append(scored[:args.keep])
        if i % 2000 == 0 and i:
            rate = i / (time.time() - t0)
            print(f"  {i}/{n}  {rate:.1f}/s  eta {(n - i) / rate / 60:.0f}m",
                  flush=True)

    device = pick_device(args.device)
    model, alphabet, mode = load_ar(args.checkpoint, device)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)
    feats = torch.cat([x for x, _, _ in loader])

    pairs = [(i, w, g, uni) for i in range(n)
             for w, g, uni in proposals[i] if np.isfinite(uni)]
    print(f"AR-scoring {len(pairs)} (swipe, word) pairs")
    out: list[list[tuple[str, float, float, int, float]]] = [[] for _ in
                                                             range(n)]
    with torch.no_grad():
        for s in range(0, len(pairs), 1024):
            chunk = pairs[s:s + 1024]
            x = feats[[i for i, *_ in chunk]].to(device)
            lp = score_words(model, x, [w for _, w, *_ in chunk],
                             alphabet).cpu().numpy()
            for (i, w, g, uni), ar in zip(chunk, lp):
                out[i].append((w, float(ar), float(uni), len(w), float(g)))

    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    surfaced = sum(refs[i] in {w for w, *_ in out[i]} for i in range(n)
                   if refs[i] not in {w for w, *_ in lists[i][:args.m]})
    misses = sum(refs[i] not in {w for w, *_ in lists[i][:args.m]}
                 for i in range(n))
    print(f"wrote {args.out}; truth surfaced on {surfaced}/{misses} "
          f"coverage misses")


if __name__ == "__main__":
    main()
