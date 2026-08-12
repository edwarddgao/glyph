#!/usr/bin/env python3
"""Upper bound for a generator family: reconstruct real gestures, not sample.

    python scripts/gen_reconstruct_corpus.py --checkpoint runs/gesturegen_warp/epoch9.pt

"The learned generator loses to min-jerk" has two very different causes and
this separates them. Encode each *real* training gesture, take the
posterior mean, and decode it: the output is the best this parameterization
can do when it is handed the answer. Train a decoder on that corpus and:

  * scores near real data  -> the parameterization is fine and the loss is
    in the prior, i.e. in what the model invents when it is not shown a
    gesture. That is a sampling problem (better prior, adversarial or
    diffusion sampling, more latent capacity), not an architecture problem.
  * scores near the sampled corpus -> the parameterization itself destroys
    what the decoder feeds on, and no amount of better sampling will help.

Without this control, "learned generators are worse here" is not
distinguishable from "I built them badly".
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from swipe_typing import cache, features
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.gesturegen import GenConfig, build, prototype
from swipe_typing.schema import Swipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--checkpoint", default="runs/gesturegen_warp/epoch9.pt")
    ap.add_argument("--out", default="data/canonical/gesturegen_recon/train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build(GenConfig(**ck["cfg"]),
                  ck["args"].get("arch", "ar")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    kb = KeyboardLayout.qwerty()
    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.limit)
    print(f"reconstructing {len(train):,} real gestures ...")
    t0 = time.time()
    protos: dict[str, np.ndarray] = {}
    out: list[Swipe] = []
    n = features.N_POINTS
    for s in range(0, len(train), args.batch_size):
        idx = range(s, min(s + args.batch_size, len(train)))
        words = [train.words[i] for i in idx]
        g = np.stack([features.resample(train.points(i), train.times(i),
                                        n=n, mode="time") for i in idx])
        p = np.stack([protos.setdefault(w, prototype(w, kb, n)) for w in words])
        dur = np.array([max(float(train.times(i)[-1] - train.times(i)[0]), 50.0)
                        for i in idx]) / 1000.0
        gt = torch.from_numpy(g).to(device)
        pt = torch.from_numpy(p).to(device)
        ld = torch.from_numpy(np.log(dur).astype(np.float32)).to(device)
        with torch.no_grad():
            mu, _ = model.encode(gt, pt, ld)
            pred, _ = (model.decode(pt, mu, ld) if hasattr(model, "_sample_curve")
                       else model.decode_tf(gt, pt, ld, mu, prev_dropout=0.0))
        pts = pred.cpu().numpy()
        for j, i in enumerate(idx):
            t = np.linspace(0.0, dur[j] * 1000.0, n)
            out.append(Swipe(word=words[j], x=pts[j, :, 0], y=pts[j, :, 1],
                             t=np.round(t).astype(np.int32),
                             aspect=float(ck["aspect"]),
                             session="recon", source="recon", split="train"))
        if s and s % (args.batch_size * 200) == 0:
            print(f"  {s:,}/{len(train):,}  ({time.time() - t0:.0f}s)",
                  flush=True)

    cache.write(out, args.out)
    print(f"wrote {len(out):,} to {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
