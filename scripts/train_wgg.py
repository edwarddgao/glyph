#!/usr/bin/env python3
"""Train WordGesture-GAN to the paper's recipe, then write its corpus.

    python scripts/train_wgg.py --epochs 6 --out runs/wgg
    python scripts/train_wgg.py --generate --checkpoint runs/wgg/wgg.pt \
        --out-corpus data/canonical/wgg/train

The control arm: the published method, unmodified where the paper is
explicit, so the repo's own generator designs can be judged against it
rather than against my reading of it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from swipe_typing import cache
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.wgg import (WGGConfig, WordGestureGAN, encode_gesture,
                                    sample_swipes, word_prototype)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_gesture_gen import pick_device  # noqa: E402


class WGGDataset(Dataset):
    def __init__(self, corpus: SwipeCorpus, layout: KeyboardLayout, n: int):
        self.corpus, self.layout, self.n = corpus, layout, n
        self.protos: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, i):
        w = self.corpus.words[i]
        g = encode_gesture(self.corpus.points(i), self.corpus.times(i), self.n)
        p = self.protos.get(w)
        if p is None:
            p = word_prototype(w, self.layout, self.n)
            self.protos[w] = p
        return torch.from_numpy(g), torch.from_numpy(p)


def generate(args) -> None:
    device = pick_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WordGestureGAN(WGGConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    kb = KeyboardLayout.qwerty()
    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.limit)
    print(f"generating {len(train):,} swipes ...")
    t0 = time.time()
    swipes = sample_swipes(model, list(train.words), kb, device,
                           aspect=float(ck["aspect"]), seed=args.seed)
    cache.write(swipes, args.out_corpus)
    print(f"wrote {len(swipes):,} to {args.out_corpus} "
          f"({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="runs/wgg")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--checkpoint", default="runs/wgg/wgg.pt")
    ap.add_argument("--out-corpus", default="data/canonical/wgg/train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.generate:
        return generate(args)

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kb = KeyboardLayout.qwerty()
    cfg = WGGConfig()

    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.train_limit)
    loader = DataLoader(WGGDataset(train, kb, cfg.n_points),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    print(f"device {device}  train {len(train):,}")

    model = WordGestureGAN(cfg).to(device)
    print(f"params: {model.num_parameters():,}")
    # The paper trains D and G with the same optimizer settings; betas follow
    # the WGAN-GP convention used with spectral norm.
    opt_d = torch.optim.Adam(model.disc.parameters(), lr=args.lr,
                             betas=(0.5, 0.9))
    opt_g = torch.optim.Adam(
        list(model.gen.parameters()) + list(model.enc.parameters()),
        lr=args.lr, betas=(0.5, 0.9))

    aspect = float(np.median(train.aspects))
    history, step = [], 0
    for epoch in range(args.epochs):
        run = {"d": 0.0, "adv": 0.0, "rec": 0.0, "lat": 0.0, "kld": 0.0}
        seen, t0, pending = 0, time.time(), 0
        for real, proto in loader:
            real, proto = real.to(device), proto.to(device)
            d_loss = model.critic_loss(real, proto)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            run["d"] += float(d_loss.detach())
            pending += 1
            if pending < cfg.critic_steps:
                continue
            pending = 0
            g_loss, parts = model.generator_loss(real, proto)
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            step += 1
            seen += 1
            for k in ("adv", "rec", "lat", "kld"):
                run[k] += parts[k]
            if step % args.log_every == 0:
                print(f"  e{epoch} g-step {step} "
                      f"d {run['d'] / (seen * cfg.critic_steps):.3f} "
                      + " ".join(f"{k} {run[k] / seen:.4f}"
                                 for k in ("adv", "rec", "lat", "kld")),
                      flush=True)

        row = {"epoch": epoch, "secs": round(time.time() - t0, 1),
               **{k: round(v / max(seen, 1), 5) for k, v in run.items()}}
        history.append(row)
        print(f"epoch {epoch}: " + "  ".join(f"{k} {v}" for k, v in row.items()),
              flush=True)
        torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                    "aspect": aspect, "args": vars(args), "epoch": epoch},
                   out / "wgg.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nsaved {out / 'wgg.pt'}")


if __name__ == "__main__":
    main()
