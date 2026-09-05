#!/usr/bin/env python3
"""Dump first-pass n-best lists as training data for a second-pass reranker.

The LM ceiling test showed ~73% of the remaining n-best headroom is not
language-modelable (a pretrained GPT-2 recovers +1.3 of 4.96 points, and scaling
82M -> 124M buys only 0.14). What is left is acoustic: ``philipp``/``philip``,
``wayne``/``warner`` -- candidates the first pass ranks wrongly on gesture
evidence alone.

A second pass can attack that because it is not bound by the constraints the
first pass runs under. CTC assumes frames are conditionally independent given
the label, and beam search must stay streamable; a rescorer sees the whole
gesture against only ~8 candidates, so it can model the dependencies both give up.

Writes one .npz per split holding the resampled gesture features, the candidate
words with their first-pass scores, and which candidate is correct.

Usage:
    python scripts/dump_nbest.py --split futo/train --limit 150000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (  # noqa: E402
    SwipeCorpus,
    SwipeDataset,
    make_loader,
)
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/train")
    ap.add_argument("--out", default="data/nbest")
    ap.add_argument("--limit", type=int, default=150000)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)

    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4)
    print(f"{args.split}: {len(corpus):,} swipes")

    feats, chunks = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            feats.append(x.numpy().astype(np.float32))
            chunks.append(model(x.to(device)).float().cpu().numpy())
    features = np.concatenate(feats)
    log_probs = np.concatenate(chunks)
    print(f"features {features.shape}, posteriors {log_probs.shape}")

    cfg = BeamConfig(beam_width=args.beam_width, top_k=args.top_k)
    K = args.top_k
    cand = np.zeros((len(corpus), K), dtype=object)
    scores = np.full((len(corpus), K), -1e9, dtype=np.float32)
    valid = np.zeros((len(corpus), K), dtype=bool)
    target = np.full(len(corpus), -1, dtype=np.int16)

    t0 = time.time()
    for i, item in enumerate(log_probs):
        hyps = beam_search(item, lexicon, alphabet, cfg)
        for j, (w, s) in enumerate(hyps[:K]):
            cand[i, j] = w
            scores[i, j] = s
            valid[i, j] = True
            if w == corpus.words[i]:
                target[i] = j
        if (i + 1) % 20000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1:,}/{len(corpus):,}  {rate:.0f}/s  "
                  f"eta {(len(corpus) - i - 1) / rate / 60:.1f} min")

    has = target >= 0
    top1 = target == 0
    print(f"\n  true word in n-best: {has.mean():.4f}")
    print(f"  first-pass top-1:    {top1.mean():.4f}")
    print(f"  rerankable (in list, not first): {(has & ~top1).mean():.4f}")

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
