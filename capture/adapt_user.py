#!/usr/bin/env python3
"""Per-user adaptation (#58's margin, this user's data): fine-tune the
canonical encoder on the user's own captured swipes from the TRAIN sets,
evaluate later with composed_stack.py on the held-out sets.

Two arms:
  user      — fine-tune on user swipes alone (low LR, few epochs)
  user+rep  — user swipes + a FUTO replay slice, guarding general accuracy

Checkpoints land in capture/runs/<arm>/encoder.pt in the standard format.

Usage:
  .venv/bin/python capture/adapt_user.py --train-sets 2,3,5,6,8
  .venv/bin/python capture/composed_stack.py --lm gpt2 --sets 1,4,7 \
      --encoders canonical,userft=<abs>/capture/runs/user/encoder.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import ConcatDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT.parent / "scripts"))
sys.path.insert(0, str(ROOT))

from swipe_typing.layout import ALPHABET, KeyboardLayout            # noqa: E402
from swipe_typing.model import (SwipeCorpus, SwipeDataset,          # noqa: E402
                                ctc_loss, make_loader)
from eval_decoder import load_model, pick_device                    # noqa: E402
from decode_capture import build_corpus, latest_per_sentence        # noqa: E402


def finetune(base_ckpt: dict, model, datasets, device, epochs, lr, out: Path):
    model = copy.deepcopy(model).to(device).train()
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader = make_loader(ds, batch_size=32, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        tot = n = 0
        for x, targets, lengths in loader:
            opt.zero_grad()
            lp = model(x.to(device))
            loss = ctc_loss(lp, targets.to(device), lengths.to(device),
                            model.cfg.blank)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(x)
            n += len(x)
        print(f"    epoch {ep + 1}/{epochs}  loss {tot / n:.4f}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": base_ckpt["cfg"],
                "alphabet": base_ckpt["alphabet"],
                "args": base_ckpt["args"]}, out / "encoder.pt")
    print(f"    saved {out / 'encoder.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sets", default="2,3,5,6,8")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay", type=int, default=1500,
                    help="FUTO replay swipes for the user+rep arm")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the chronologically FIRST N train swipes "
                         "(dose-response: simulates early usage)")
    ap.add_argument("--out", default="",
                    help="output dir name under capture/runs "
                         "(replay arm only; default: user / user_rep)")
    args = ap.parse_args()

    device = pick_device("auto")
    kb = KeyboardLayout.qwerty()
    base_path = ROOT.parent / "runs/full/encoder.pt"
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    model, alphabet, key_units, mode = load_model(str(base_path), device)

    keep = {int(s) for s in args.train_sets.split(",")}
    captures = {k: v for k, v in latest_per_sentence("capture").items()
                if v.get("set", 1) in keep}
    if args.limit:
        # chronological prefix — the first N swipes a user would produce
        ordered = sorted(captures.items(), key=lambda kv: kv[1]["ts"])
        taken, n = {}, 0
        for k, v in ordered:
            if n >= args.limit:
                break
            taken[k] = v
            n += len(v["gestures"])
        captures = taken
    corpus, meta = build_corpus(captures, alphabet)
    print(f"user train swipes: {len(corpus)} from sets {sorted(keep)}"
          + (f" (dose limit {args.limit})" if args.limit else ""))

    # augmentation ON (dataset default) — it is the training-time default
    user_ds = SwipeDataset(corpus, kb, resample_mode=mode,
                           key_units=key_units)
    replay = SwipeCorpus.load(ROOT.parent / "data/canonical/futo/train",
                              alphabet, limit=args.replay)
    rep_ds = SwipeDataset(replay, kb, resample_mode=mode,
                          key_units=key_units)

    if args.out:
        print(f"arm 'user+rep' ({len(corpus)} swipes + {len(replay)} replay)")
        finetune(base_ckpt, model, [user_ds, rep_ds], device, args.epochs,
                 args.lr, ROOT / "runs" / args.out)
        return
    print(f"arm 'user' ({len(corpus)} swipes, lr={args.lr}, "
          f"epochs={args.epochs})")
    finetune(base_ckpt, model, [user_ds], device, args.epochs, args.lr,
             ROOT / "runs/user")
    print(f"arm 'user+rep' (+{len(replay)} FUTO replay)")
    finetune(base_ckpt, model, [user_ds, rep_ds], device, args.epochs,
             args.lr, ROOT / "runs/user_rep")


if __name__ == "__main__":
    main()
