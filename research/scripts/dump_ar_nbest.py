#!/usr/bin/env python3
"""Dump the AR decoder's n-best lists in the stack's npz format.

    python scripts/dump_ar_nbest.py --split futo/train --limit 150000
    python scripts/dump_ar_nbest.py --split futo/validation --limit 20000

Same schema as dump_nbest.py (features, candidates, scores, valid, target,
words, sessions), so train_rescorer.py and eval_deferred_commit.py consume it
unchanged. Scores are the ranked first-pass combination ar + a*logp + b*len,
mirroring how the CTC dumps store beam_search's combined score.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.ar import FlatTrie, ar_beam  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, pick_device  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_full/ar_decoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/train")
    ap.add_argument("--out", default="data/nbest_ar")
    ap.add_argument("--limit", type=int, default=150000)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, mode = load_ar(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    trie = FlatTrie(lexicon, alphabet)

    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      n_points=model.cfg.n_frames,
                      shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=4)
    print(f"{args.split}: {len(corpus):,} swipes")

    feats = []
    for x, _, _ in loader:
        feats.append(x.numpy().astype(np.float32))
    features = np.concatenate(feats)

    K = args.top_k
    cand = np.zeros((len(corpus), K), dtype=object)
    scores = np.full((len(corpus), K), -1e9, dtype=np.float32)
    valid = np.zeros((len(corpus), K), dtype=bool)
    target = np.full(len(corpus), -1, dtype=np.int16)

    t0 = time.time()
    for s in range(0, len(features), args.chunk):
        x = torch.from_numpy(features[s:s + args.chunk]).to(device)
        lists = ar_beam(model, x, trie, alphabet, beam_width=args.beam_width)
        for j, cl in enumerate(lists):
            i = s + j
            ranked = sorted(
                cl, key=lambda c: c[1] + args.alpha * c[2] + args.beta * c[3],
                reverse=True,
            )[:K]
            for k, (w, ar, lp, n) in enumerate(ranked):
                cand[i, k] = w
                scores[i, k] = ar + args.alpha * lp + args.beta * n
                valid[i, k] = True
                if w == corpus.words[i]:
                    target[i] = k
        done = s + len(lists)
        if (s // args.chunk) % 50 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{len(corpus):,}  {rate:.0f}/s  "
                  f"eta {(len(corpus) - done) / max(rate, 1) / 60:.1f} min",
                  flush=True)

    has = target >= 0
    print(f"\n  true word in n-best: {has.mean():.4f}")
    print(f"  first-pass top-1:    {(target == 0).mean():.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.split.replace("/", "_")
    np.savez_compressed(
        out / f"{name}.npz",
        features=features,
        candidates=cand.astype(str),
        scores=scores,
        valid=valid,
        target=target,
        words=np.array(corpus.words, dtype=str),
        sessions=np.array(corpus.sessions, dtype=str),
    )
    print(f"  wrote {out / (name + '.npz')}")


if __name__ == "__main__":
    main()
