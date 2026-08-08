#!/usr/bin/env python3
"""Train the second-pass acoustic rescorer on dumped n-best lists.

    python scripts/dump_nbest.py --split futo/train --limit 150000
    python scripts/dump_nbest.py --split futo/validation --limit 20000
    python scripts/train_rescorer.py --epochs 8

Reported against the two numbers that bound it: the first-pass top-1 it has to
beat, and the n-best ceiling it cannot exceed (the true word has to be in the
list for a rescorer to find it).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model.rescore import (  # noqa: E402
    MAX_CAND_LEN,
    RescoreConfig,
    Rescorer,
    encode_candidates,
    listwise_loss,
)

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import pick_device  # noqa: E402


class NBestSet(Dataset):
    """One dumped n-best file. Candidates are pre-encoded once, not per epoch."""

    def __init__(self, path: Path, layout: KeyboardLayout):
        data = np.load(path, allow_pickle=False)
        self.features = data["features"]
        self.scores = data["scores"].astype(np.float32)
        self.valid = data["valid"]
        self.target = data["target"].astype(np.int64)
        cands = data["candidates"]
        n, k = cands.shape
        flat = [str(w) for w in cands.reshape(-1)]
        ids, pos, mask = encode_candidates(flat, layout)
        self.cand_ids = ids.reshape(n, k, MAX_CAND_LEN)
        self.cand_pos = pos.reshape(n, k, MAX_CAND_LEN, 2)
        self.cand_mask = mask.reshape(n, k, MAX_CAND_LEN)
        # A slot with no candidate must not be selectable. Also drop any
        # non-finite score defensively: beam_search no longer emits them, but a
        # single -inf here silently NaNs the whole run rather than failing loud.
        bad = ~np.isfinite(self.scores)
        if bad.any():
            print(f"  [warn] {int(bad.sum())} non-finite first-pass scores "
                  f"-> marked invalid")
            self.valid = self.valid & ~bad
            self.target[(self.target >= 0)
                        & bad[np.arange(len(self.target)),
                              np.clip(self.target, 0, None)]] = -1
        self.scores[~self.valid] = -1e4
        assert np.isfinite(self.scores).all()
        assert np.isfinite(self.features).all()

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i):
        return (torch.from_numpy(self.features[i]),
                torch.from_numpy(self.cand_ids[i]),
                torch.from_numpy(self.cand_pos[i]),
                torch.from_numpy(self.cand_mask[i]),
                torch.from_numpy(self.scores[i]),
                torch.tensor(self.target[i]),
                torch.from_numpy(self.valid[i]))

    @property
    def baseline(self) -> float:
        return float((self.target == 0).mean())

    @property
    def ceiling(self) -> float:
        return float((self.target >= 0).mean())


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    hit = total = 0
    for feats, ids, pos, mask, scores, target, valid in loader:
        logits = model(feats.to(device), ids.to(device), pos.to(device),
                       mask.to(device), scores.to(device))
        logits = logits.masked_fill(~valid.to(device), -1e4)
        pick = logits.argmax(dim=1).cpu()
        hit += int((pick == target).sum())
        total += len(target)
    return hit / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", default="data/nbest")
    ap.add_argument("--train", default="futo_train.npz")
    ap.add_argument("--val", default="futo_validation.npz")
    ap.add_argument("--out", default="runs/rescorer")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.nbest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_set = NBestSet(root / args.train, kb)
    val_set = NBestSet(root / args.val, kb)
    print(f"train {len(train_set):,}  val {len(val_set):,}  "
          f"({time.time() - t0:.0f}s)")
    print(f"val first-pass top-1 {val_set.baseline:.4f}   "
          f"n-best ceiling {val_set.ceiling:.4f}   "
          f"headroom {val_set.ceiling - val_set.baseline:.4f}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=2, persistent_workers=True)

    cfg = RescoreConfig(n_input=train_set.features.shape[-1],
                        d_model=args.d_model, dropout=args.dropout)
    model = Rescorer(cfg).to(device)
    flat = train_set.features.reshape(-1, cfg.n_input)
    sample = flat[np.random.default_rng(0).choice(len(flat), 200_000, replace=False)]
    model.set_normalization(sample.mean(0), sample.std(0))
    model.to(device)
    print(f"rescorer params: {model.num_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(s / 200, 1.0) * 0.5 *
        (1 + math.cos(math.pi * min(s / max(total_steps, 1), 1.0)))
    )

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        run, seen, t_ep = 0.0, 0, time.time()
        for feats, ids, pos, mask, scores, target, valid in train_loader:
            logits = model(feats.to(device), ids.to(device), pos.to(device),
                           mask.to(device), scores.to(device))
            loss = listwise_loss(logits, target.to(device), valid.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss)
            seen += 1
        acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: loss {run / max(seen, 1):.4f}  "
              f"val top-1 {acc:.4f} ({acc - val_set.baseline:+.4f})  "
              f"scale {float(model.first_pass_scale.detach()):.3f}  "
              f"{time.time() - t_ep:.0f}s")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)},
                       out / "rescorer.pt")

    print(f"\nbest val top-1 {best:.4f}  "
          f"(first pass {val_set.baseline:.4f}, ceiling {val_set.ceiling:.4f})")
    print(f"recovered {100 * (best - val_set.baseline) / max(val_set.ceiling - val_set.baseline, 1e-9):.1f}% "
          f"of the available headroom")


if __name__ == "__main__":
    main()
