#!/usr/bin/env python3
"""Train the diffusion gesture generator, then write its corpus.

    python scripts/train_gesture_diffusion.py --epochs 8 --out runs/gesturediff

Per-epoch diagnostics are the same texture/geometry panel the VAE arms
report, so the arms are directly comparable: path/prototype ratio,
endpoint error, dwell fraction. Sampling a panel every epoch costs a few
seconds at 50 DDIM steps and is the only honest early signal — the
denoising loss says nothing about whether a sample reaches its letters.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from swipe_typing import features, minjerk
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.gesturediff import (DiffConfig, GestureDiffusion,
                                            sample_swipes)
from swipe_typing.model.gesturegen import prototype

import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_gesture_gen import (GestureGenDataset, endpoint_error,  # noqa: E402
                               pick_device, texture_stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="runs/gesturediff")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--duration-model", default="runs/minjerk_rand_model.json")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kb = KeyboardLayout.qwerty()
    dur_model = minjerk.MinJerkModel.load(args.duration_model)
    print(f"device: {device}")

    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.train_limit)
    val = SwipeCorpus.load(Path(args.cache) / "futo/validation", kb.letters,
                           limit=10_000)
    loader = DataLoader(GestureGenDataset(train, kb),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    print(f"train {len(train):,}")

    panel_words = list(val.words)[:256]
    real_pts = np.stack([
        features.resample(val.points(i), val.times(i), n=features.N_POINTS,
                          mode="time") for i in range(256)])
    real_dwell, real_jerk, real_len = texture_stats(real_pts)
    real_end = endpoint_error(real_pts, panel_words, kb)
    proto_len = float(np.mean([
        np.linalg.norm(np.diff(prototype(w, kb), axis=0), axis=1).sum()
        for w in panel_words]))
    aspect = float(np.median(train.aspects))
    print(f"real: dwell {real_dwell:.3f}  path/proto {real_len / proto_len:.2f}"
          f"  end-err {real_end:.4f}")

    cfg = DiffConfig(d_model=args.d_model, n_blocks=args.blocks,
                     timesteps=args.timesteps)
    model = GestureDiffusion(cfg).to(device)
    print(f"params: {model.num_parameters():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500)
        * 0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0))))

    history, step = [], 0
    for epoch in range(args.epochs):
        model.train()
        run, seen, t0 = 0.0, 0, time.time()
        for gesture, proto, _ in loader:
            loss = model.loss(gesture.to(device), proto.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += 1
            run += float(loss.detach())
            if step % args.log_every == 0:
                print(f"  e{epoch} {step}/{total} loss {run / seen:.4f} "
                      f"{seen * args.batch_size / (time.time() - t0):.0f}/s",
                      flush=True)

        sw = sample_swipes(model, panel_words, kb, device, aspect, dur_model,
                           steps=args.sample_steps, eta=args.eta, seed=epoch)
        pts = np.stack([np.stack([s.x, s.y], axis=1) for s in sw])
        dwell, jerk, plen = texture_stats(pts)
        end_err = endpoint_error(pts, panel_words, kb)
        row = {"epoch": epoch, "loss": round(run / seen, 5),
               "dwell": round(dwell, 4), "jerk": round(jerk, 6),
               "path_ratio": round(plen / proto_len, 4),
               "end_err": round(end_err, 5),
               "secs": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {epoch}: loss {run / seen:.4f}  dwell {dwell:.3f} "
              f"(real {real_dwell:.3f})  path/proto {plen / proto_len:.2f} "
              f"(real {real_len / proto_len:.2f})  end-err {end_err:.4f} "
              f"(real {real_end:.4f})", flush=True)
        payload = {"model": model.state_dict(), "cfg": vars(cfg),
                   "aspect": aspect, "args": vars(args), "epoch": epoch}
        torch.save(payload, out / f"epoch{epoch}.pt")
        torch.save(payload, out / "gesturediff.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nsaved {out / 'gesturediff.pt'}")


if __name__ == "__main__":
    main()
