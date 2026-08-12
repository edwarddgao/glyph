#!/usr/bin/env python3
"""Write a synthetic corpus from the diffusion generator.

    python scripts/gen_diffusion_corpus.py --checkpoint runs/gesturediff/gesturediff.pt

Same protocol as the other generators: one gesture per real training swipe,
identical word multiset, so the quality oracle compares like with like.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from swipe_typing import cache, minjerk
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.gesturediff import (DiffConfig, GestureDiffusion,
                                            sample_swipes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--checkpoint", default="runs/gesturediff/gesturediff.pt")
    ap.add_argument("--checkpoint-file", default=None)
    ap.add_argument("--out", default="data/canonical/gesturediff/train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--duration-model", default="runs/minjerk_rand_model.json")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = args.checkpoint
    if args.checkpoint_file:
        ckpt = Path(args.checkpoint_file).read_text().strip()
    print(f"checkpoint {ckpt}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = GestureDiffusion(DiffConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    kb = KeyboardLayout.qwerty()
    dur = minjerk.MinJerkModel.load(args.duration_model)
    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.limit)
    print(f"generating {len(train):,} swipes ({args.steps} DDIM steps) ...")
    t0 = time.time()
    swipes = sample_swipes(model, list(train.words), kb, device,
                           aspect=float(ck["aspect"]), duration_model=dur,
                           steps=args.steps, eta=args.eta,
                           batch_size=args.batch_size, seed=args.seed)
    for sw in swipes:
        sw.split = "train"
    cache.write(swipes, args.out)
    print(f"wrote {len(swipes):,} to {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
