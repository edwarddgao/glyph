#!/usr/bin/env python3
"""Pick the MDN sampling temperature by texture match, before generating.

    python scripts/probe_gen_temperature.py --checkpoint runs/gesturegen_mdn/gesturegen.pt

Graves-style rollouts accumulate their own sampling noise: too hot and 64
sampled displacements random-walk away from the word, too cold and the
mixture collapses back to its mean and reproduces the v1 glide. The choice
is made on *texture statistics against real gestures* — dwell fraction,
path/prototype ratio, endpoint error — never on decoder accuracy, which is
the quantity the generator is supposed to be judged by later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from swipe_typing import features
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.gesturegen import (GenConfig, build, prototype,
                                           sample_swipes)


def stats(pts, words, kb, proto_len):
    step = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    dwell = float((step < 0.25 * step.mean(axis=1, keepdims=True)).mean())
    ratio = float(step.sum(axis=1).mean() / proto_len)
    end = float(np.mean([np.linalg.norm(p[-1] - kb.center(w[-1]))
                         for p, w in zip(pts, words)]))
    visit = float(np.mean([
        np.mean([np.linalg.norm(p - kb.center(c), axis=1).min() for c in w])
        for p, w in zip(pts, words)]))
    return dwell, ratio, end, visit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/gesturegen_mdn/gesturegen.pt",
                    help="single checkpoint, or use --run to sweep epochs")
    ap.add_argument("--run", default=None,
                    help="run directory; sweeps every epoch*.pt in it, since "
                         "reconstruction loss selects against texture")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--temps", default="0.2,0.35,0.5,0.7,1.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None,
                    help="write the winning temperature to this file")
    ap.add_argument("--out-checkpoint", default=None,
                    help="write the winning checkpoint path to this file")
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    device = torch.device(args.device)
    if args.run:
        ckpts = sorted(Path(args.run).glob("epoch*.pt"),
                       key=lambda p: int(p.stem[5:]))
    else:
        ckpts = [Path(args.checkpoint)]

    val = SwipeCorpus.load(Path(args.cache) / "futo/validation", kb.letters,
                           limit=20000)
    idx = [i for i in range(len(val)) if 3 <= len(val.words[i]) <= 8][:args.n]
    words = [val.words[i] for i in idx]
    proto_len = float(np.mean([
        np.linalg.norm(np.diff(prototype(w, kb), axis=0), axis=1).sum()
        for w in words]))

    real = np.stack([features.resample(val.points(i), val.times(i),
                                       n=features.N_POINTS, mode="time")
                     for i in idx])
    r = stats(real, words, kb, proto_len)
    print(f"real      dwell {r[0]:.3f}  path/proto {r[1]:.2f}  "
          f"end-err {r[2]:.4f}  visit {r[3]:.4f}")

    best, best_score = None, float("inf")
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = build(GenConfig(**ck["cfg"]),
                      ck["args"].get("arch", "ar")).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        temps = ([float(x) for x in args.temps.split(",")]
                 if model.K else [1.0])
        for t in temps:
            sw = sample_swipes(model, words, kb, device, float(ck["aspect"]),
                               step_temperature=t, seed=11)
            pts = np.stack([np.stack([s.x, s.y], axis=1) for s in sw])
            d, ratio, end, visit = stats(pts, words, kb, proto_len)
            # Scale-free geometry match: the two statistics that decide
            # whether a gesture is readable at all (does it reach the keys,
            # does it reach the end), with dwell as the texture tie-breaker.
            score = (abs(ratio - r[1]) / r[1] + abs(end - r[2]) / r[2]
                     + 0.5 * abs(d - r[0]) / r[0])
            print(f"{path.name:<12} T={t:<5} dwell {d:.3f}  "
                  f"path/proto {ratio:.2f}  end-err {end:.4f}  "
                  f"visit {visit:.4f}  score {score:.3f}", flush=True)
            if score < best_score:
                best, best_score = (path, t), score

    print(f"\nbest: {best[0]}  T={best[1]}  (score {best_score:.3f})")
    if args.out:
        Path(args.out).write_text(str(best[1]))
    if args.out_checkpoint:
        Path(args.out_checkpoint).write_text(str(best[0]))


if __name__ == "__main__":
    main()
