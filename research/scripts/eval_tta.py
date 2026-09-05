#!/usr/bin/env python3
"""Test-time augmentation: does averaging over several views help?

The encoder trains under heavy co-augmentation but is evaluated on a single
clean view, so it has never been asked to use the invariance it was taught.

This is sound here for a specific reason. Co-augmentation transforms the
trajectory *and* the layout together, so every augmented view is a legitimate
input rather than a corrupted one, and the output space -- per-frame posteriors
over a-z plus blank -- carries no spatial coordinates, so views are directly
comparable. The affine is purely spatial and never touches timing, so frame *t*
denotes the same instant in every view and frame-wise averaging is well defined.

Two combination rules, which do different things:

  posterior averaging   one decode over the mean of the views' probabilities.
                        Reduces variance; cannot introduce a candidate no single
                        view would have produced.
  beam union            decode every view, pool the candidates. Costs N decodes
                        but can *raise the n-best ceiling*, which posterior
                        averaging cannot.

The ceiling is what matters now: every second-pass gain is capped by it.

Usage:
    python scripts/eval_tta.py --limit 20000 --views 4
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.augment import DEFAULT as DEFAULT_AUG  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402


@torch.no_grad()
def posteriors(model, corpus, kb, device, mode, key_units, cfg, seed):
    """(N, T, C) log-probabilities for one view of the corpus."""
    ds = SwipeDataset(corpus, kb, augment_cfg=cfg, resample_mode=mode,
                      key_units=key_units, seed=seed)
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=4)
    out = []
    for x, _, _ in loader:
        out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def score(nbest, refs, k):
    top1 = sum(bool(b) and b[0][0] == refs[i] for i, b in enumerate(nbest))
    hit = sum(refs[i] in [w for w, _ in b[:k]] for i, b in enumerate(nbest))
    return top1 / len(refs), hit / len(refs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--views", type=int, default=4,
                    help="total views including the clean one")
    ap.add_argument("--strength", type=float, default=0.4,
                    help="augmentation strength for TTA views (1.0 = training)")
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    refs = corpus.words
    n = len(refs)
    cfg = BeamConfig(beam_width=args.beam_width, top_k=args.top_k)

    # View 0 is clean; the rest are mild co-augmented views.
    aug = DEFAULT_AUG.scaled(args.strength)
    views = [posteriors(model, corpus, kb, device, mode, key_units, None, 0)]
    for v in range(1, args.views):
        views.append(posteriors(model, corpus, kb, device, mode, key_units,
                                aug, seed=100 + v))
    print(f"{args.split}: n={n:,}  {len(views)} views  "
          f"strength {args.strength}\n")

    def average(k: int) -> np.ndarray:
        """Mean in probability space, back to log."""
        stack = np.stack(views[:k])                      # (k, N, T, C)
        m = stack.max(axis=0, keepdims=True)
        return (m + np.log(np.exp(stack - m).mean(axis=0, keepdims=True)))[0]

    print(f"  {'config':<28}{'top-1':>9}{'top-' + str(args.top_k):>9}{'sec':>8}")

    per_view = []
    for v, lp in enumerate(views):
        t0 = time.time()
        nb = [beam_search(item, lexicon, alphabet, cfg) for item in lp]
        per_view.append(nb)
        t1, tk = score(nb, refs, args.top_k)
        label = "single view (clean)" if v == 0 else f"single view {v} (aug)"
        print(f"  {label:<28}{t1:>9.4f}{tk:>9.4f}{time.time() - t0:>8.1f}")

    for k in (2, len(views)):
        if k > len(views):
            continue
        t0 = time.time()
        nb = [beam_search(item, lexicon, alphabet, cfg) for item in average(k)]
        t1, tk = score(nb, refs, args.top_k)
        print(f"  {'avg posteriors x' + str(k):<28}{t1:>9.4f}{tk:>9.4f}"
              f"{time.time() - t0:>8.1f}")

    # Beam union: pool candidates across views, keeping each word's best score.
    t0 = time.time()
    union = []
    for i in range(n):
        pooled: dict[str, float] = {}
        for nb in per_view:
            for w, s in nb[i]:
                if w not in pooled or s > pooled[w]:
                    pooled[w] = s
        union.append(sorted(pooled.items(), key=lambda kv: kv[1], reverse=True))
    t1, tk = score(union, refs, args.top_k)
    print(f"  {'beam union x' + str(len(views)):<28}{t1:>9.4f}{tk:>9.4f}"
          f"{time.time() - t0:>8.1f}")

    base1, basek = score(per_view[0], refs, args.top_k)
    print(f"\n  baseline (clean): top-1 {base1:.4f}  top-{args.top_k} {basek:.4f}")
    se = float(np.sqrt(base1 * (1 - base1) / n))
    print(f"  1 SE at n={n:,} is {se:.4f}; treat anything under {2 * se:.4f} "
          f"as noise")


if __name__ == "__main__":
    main()
