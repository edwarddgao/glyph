#!/usr/bin/env python3
"""Write a synthetic corpus from the learned generator, mirroring futo/train.

    python scripts/gen_learned_corpus.py --checkpoint runs/gesturegen/gesturegen.pt

Same protocol as gen_minjerk_corpus: one synthetic gesture per real training
swipe, identical word multiset, so decoder-training differences are
attributable to the gestures alone. The bar is the domain-randomized
min-jerk corpus: 85.65 val beam synthetic-only (#57).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from swipe_typing import cache
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.gesturegen import GenConfig, build, sample_swipes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--checkpoint", default="runs/gesturegen/gesturegen.pt")
    ap.add_argument("--checkpoint-file", default=None,
                    help="read --checkpoint from this file (the texture "
                         "probe's winner)")
    ap.add_argument("--out", default="data/canonical/gesturegen/train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="latent scale")
    ap.add_argument("--step-temperature", type=float, default=1.0,
                    help="MDN draw scale; pick with probe_gen_temperature.py")
    ap.add_argument("--step-temperature-file", default=None,
                    help="read --step-temperature from this file")
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
    model = build(GenConfig(**ck["cfg"]),
                  ck["args"].get("arch", "ar")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    kb = KeyboardLayout.qwerty()
    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.limit)
    step_t = args.step_temperature
    if args.step_temperature_file:
        step_t = float(Path(args.step_temperature_file).read_text().strip())
    print(f"generating {len(train):,} swipes "
          f"(z T={args.temperature}, step T={step_t}) ...")
    t0 = time.time()
    swipes = sample_swipes(model, list(train.words), kb, device,
                           aspect=float(ck["aspect"]),
                           temperature=args.temperature,
                           step_temperature=step_t, seed=args.seed)
    for sw in swipes:
        sw.split = "train"
    cache.write(swipes, args.out)
    print(f"wrote {len(swipes):,} to {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
