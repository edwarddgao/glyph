#!/usr/bin/env python3
"""Train the learned gesture generator (prototype-conditioned CVAE, AR decoder).

    python scripts/train_gesture_gen.py --epochs 10 --out runs/gesturegen

Per-epoch diagnostics beyond the losses: prior samples for a fixed word set,
reporting dwell fraction and a jerk proxy against real gestures for the same
words. Dwell collapse toward zero is the spline failure mode (#56) arriving
through mode-averaging — the early-warning signal this script exists to show.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from swipe_typing import features
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.gesturegen import (GenConfig, build, gen_loss,
                                           prototype, sample_swipes)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GestureGenDataset(Dataset):
    def __init__(self, corpus: SwipeCorpus, layout: KeyboardLayout,
                 n: int = features.N_POINTS):
        self.corpus = corpus
        self.layout = layout
        self.n = n
        self.protos: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, i):
        word = self.corpus.words[i]
        pts = features.resample(self.corpus.points(i), self.corpus.times(i),
                                n=self.n, mode="time")
        proto = self.protos.get(word)
        if proto is None:
            proto = prototype(word, self.layout, self.n)
            self.protos[word] = proto
        t = self.corpus.times(i)
        dur = max(float(t[-1] - t[0]), 50.0) / 1000.0
        return (torch.from_numpy(pts), torch.from_numpy(proto),
                torch.tensor(math.log(dur), dtype=torch.float32))


def texture_stats(pts: np.ndarray) -> tuple[float, float, float]:
    """(dwell fraction, jerk proxy, path length) over (B, n, 2) trajectories.

    Path length is the geometry check: real gestures run ~1.15x the
    straight-line prototype (overshoot, rounded corners). A mean-seeking
    decoder averages over step-length *and* step-direction modes, so its
    rollouts come out shorter than the polyline through the keys — which
    makes visiting every key impossible and strands the gesture before the
    final one. Watch this ratio, not just the losses.
    """
    step = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    dwell = float((step < 0.25 * step.mean(axis=1, keepdims=True)).mean())
    jerk = float(np.abs(np.diff(step, n=2, axis=1)).mean())
    return dwell, jerk, float(step.sum(axis=1).mean())


def endpoint_error(pts: np.ndarray, words: list[str],
                   layout: KeyboardLayout) -> float:
    return float(np.mean([np.linalg.norm(p[-1] - layout.center(w[-1]))
                          for p, w in zip(pts, words)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="runs/gesturegen")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--beta-warmup-epochs", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--prev-dropout", type=float, default=0.15)
    ap.add_argument("--arch", default="ar", choices=["ar", "warp", "smooth"],
                    help="ar = free-running step decoder (v1/v2); "
                         "warp = prototype-anchored offsets + monotone "
                         "time warp (v3, no accumulation); smooth = v4, "
                         "offsets restricted to a low-frequency basis")
    ap.add_argument("--mdn", type=int, default=0,
                    help="mixture components on the next-step displacement; "
                         "0 = L1 regression (mean-seeking, rolls out smooth)")
    ap.add_argument("--step-temperature", type=float, default=1.0,
                    help="MDN sampling temperature for the epoch diagnostics")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kb = KeyboardLayout.qwerty()
    print(f"device: {device}")

    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters,
                             limit=args.train_limit)
    val = SwipeCorpus.load(Path(args.cache) / "futo/validation", kb.letters,
                           limit=10_000)
    print(f"train {len(train):,}  val {len(val):,}")
    train_loader = DataLoader(GestureGenDataset(train, kb),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(GestureGenDataset(val, kb),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=2)

    # Fixed diagnostic panel: real texture stats for 512 val words, and the
    # same words for prior sampling each epoch.
    panel_words = list(val.words)[:512]
    real_pts = np.stack([
        features.resample(val.points(i), val.times(i), n=features.N_POINTS,
                          mode="time") for i in range(512)])
    real_dwell, real_jerk, real_len = texture_stats(real_pts)
    real_end = endpoint_error(real_pts, panel_words, kb)
    proto_len = float(np.mean([
        np.linalg.norm(np.diff(prototype(w, kb), axis=0), axis=1).sum()
        for w in panel_words]))
    aspect = float(np.median(train.aspects))
    print(f"real texture: dwell {real_dwell:.3f}  jerk {real_jerk:.5f}  "
          f"path/proto {real_len / proto_len:.2f}  end-err {real_end:.4f}")

    cfg = GenConfig(d_model=args.d_model, z_dim=args.z_dim,
                    prev_dropout=args.prev_dropout,
                    mdn_components=args.mdn)
    model = build(cfg, args.arch).to(device)
    print(f"params: {model.num_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(total, 1))))

    history = []
    step = 0
    best = float("inf")
    warm = args.beta_warmup_epochs * len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        run = {"rec": 0.0, "dur": 0.0, "kld": 0.0}
        seen, t0 = 0, time.time()
        for gesture, proto, logdur in train_loader:
            beta = args.beta * min(step / max(warm, 1), 1.0)
            loss, parts = gen_loss(model, gesture.to(device),
                                   proto.to(device), logdur.to(device),
                                   beta, args.free_bits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += 1
            for k in run:
                run[k] += parts[k]
            if step % args.log_every == 0:
                print(f"  e{epoch} {step}/{total} "
                      + " ".join(f"{k} {v / seen:.4f}" for k, v in run.items())
                      + f" beta {beta:.1e} "
                      f"{seen * args.batch_size / (time.time() - t0):.0f}/s",
                      flush=True)

        model.eval()
        vrec, vn = 0.0, 0
        with torch.no_grad():
            for gesture, proto, logdur in val_loader:
                gesture = gesture.to(device)
                pred, prev, _, _, _ = model(gesture, proto.to(device),
                                            logdur.to(device))
                # MDN: held-out NLL (nats); regression: mean |error|. Both are
                # "how well does it model a real gesture", on their own scale.
                v = (model.mdn_nll(pred, gesture - prev) if model.K
                     else (pred - gesture).abs().mean())
                vrec += float(v) * len(gesture)
                vn += len(gesture)
        vrec /= vn

        samples = sample_swipes(model, panel_words, kb, device, aspect,
                                step_temperature=args.step_temperature,
                                seed=epoch)
        pts = np.stack([np.stack([s.x, s.y], axis=1) for s in samples])
        dwell, jerk, plen = texture_stats(pts)
        end_err = endpoint_error(pts, panel_words, kb)
        row = {"epoch": epoch, **{k: round(v / seen, 5) for k, v in run.items()},
               "val_rec": round(vrec, 5), "dwell": round(dwell, 4),
               "jerk": round(jerk, 6),
               "path_ratio": round(plen / proto_len, 4),
               "end_err": round(end_err, 5),
               "secs": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {epoch}: val_rec {vrec:.4f}  "
              f"sample dwell {dwell:.3f} (real {real_dwell:.3f})  "
              f"jerk {jerk:.5f} (real {real_jerk:.5f})  "
              f"path/proto {plen / proto_len:.2f} (real "
              f"{real_len / proto_len:.2f})  "
              f"end-err {end_err:.4f} (real {real_end:.4f})", flush=True)

        # Every epoch is kept (the model is ~200k params). Reconstruction
        # loss is the wrong selection criterion for this artifact: v1's
        # texture kept improving for six epochs after val_rec bottomed out,
        # so best-val would ship the least usable generator. The choice is
        # made downstream on texture match (probe_gen_temperature.py).
        payload = {"model": model.state_dict(), "cfg": vars(cfg),
                   "aspect": aspect, "args": vars(args), "epoch": epoch}
        torch.save(payload, out / f"epoch{epoch}.pt")
        torch.save(payload, out / "gesturegen_last.pt")
        if vrec < best:
            best = vrec
            torch.save(payload, out / "gesturegen.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nsaved {out / 'gesturegen.pt'}")


if __name__ == "__main__":
    main()
