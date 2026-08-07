#!/usr/bin/env python3
"""Is the cross-corpus gap unseen distribution, or intrinsic difficulty?

Diagnostics ruled out the cheap explanations: participant subgroups move
accuracy by ~0.03, keyboard geometry by ~0.02 within the bulk, sampling rate is
comparable (61 vs 57 Hz), and the model's actual input distributions match
across corpora to within 1-15% (key affinity to within 1%). So the inputs look
alike and the labels are still harder to recover.

That leaves two possibilities, and they need opposite responses:

  unseen distribution   the encoder has simply never trained on this kind of
                        gesture. A little in-domain data should close most of
                        the gap -> collect/mix more diverse data.

  intrinsic difficulty  these gestures are sloppier, or their labels noisier,
                        than FUTO's. In-domain data will not help much -> the
                        gap is a property of the corpus, not a bug to fix.

Fine-tuning on a *user-disjoint* slice of How We Swipe separates them. The test
users are never trained on, so the measurement stays honest.

Usage:
    python scripts/finetune_transfer.py --epochs 3 --hws-train-frac 0.7
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from swipe_typing import cache  # noqa: E402
from swipe_typing.augment import DEFAULT as DEFAULT_AUG  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (  # noqa: E402
    SwipeCorpus,
    SwipeDataset,
    collate,
    ctc_loss,
    decode,
    make_loader,
)

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import load_model, pick_device  # noqa: E402


def user_bucket(session: str) -> float:
    """Stable hash of a donor id into [0, 1), so the split never drifts."""
    digest = hashlib.sha1(session.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def split_by_user(path: Path, alphabet: str, frac: float, limit: int | None):
    """Partition a corpus into two user-disjoint halves."""
    train, test = [], []
    for i, sw in enumerate(cache.read(path)):
        if limit and i >= limit:
            break
        (train if user_bucket(sw.session) < frac else test).append(sw)
    return (SwipeCorpus.from_swipes(train, alphabet, origin="hws-train"),
            SwipeCorpus.from_swipes(test, alphabet, origin="hws-test"))


@torch.no_grad()
def greedy_eval(model, loader, device, alphabet, max_batches=None):
    model.eval()
    preds, refs = [], []
    for i, (x, targets, lengths) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        lp = model(x.to(device))
        preds.extend(decode.greedy_decode(lp, model.cfg.blank, alphabet))
        refs.extend(decode.target_strings(targets, lengths, alphabet))
    return decode.score(preds, refs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hws-train-frac", type=float, default=0.7)
    ap.add_argument("--futo-limit", type=int, default=200000,
                    help="FUTO swipes mixed in, to prevent forgetting")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hws_train, hws_test = split_by_user(root / "how_we_swipe/test", alphabet,
                                        args.hws_train_frac, None)
    overlap = set(hws_train.sessions) & set(hws_test.sessions)
    assert not overlap, f"user leakage: {len(overlap)} sessions in both halves"
    print(f"how_we_swipe: {len(hws_train):,} train swipes "
          f"({len(set(hws_train.sessions))} users) / "
          f"{len(hws_test):,} test ({len(set(hws_test.sessions))} users)")

    # --futo-limit 0 trains on How We Swipe alone. That maximally adapts to the
    # target domain (and regresses futo), which is the strongest form of the
    # test: if even undiluted in-domain training barely moves the number, the
    # gap is not about what the encoder has seen.
    futo_train = (SwipeCorpus.load(root / "futo/train", alphabet,
                                   limit=args.futo_limit)
                  if args.futo_limit else None)
    futo_val = SwipeCorpus.load(root / "futo/validation", alphabet, limit=20000)
    print(f"futo: {len(futo_train) if futo_train else 0:,} train (mixed in), "
          f"{len(futo_val):,} val")

    def make(corpus, augment):
        return SwipeDataset(corpus, kb, augment_cfg=DEFAULT_AUG if augment else None,
                            resample_mode=mode, key_units=key_units)

    parts = [make(hws_train, True)]
    if futo_train is not None:
        parts.insert(0, make(futo_train, True))
    train_ds = ConcatDataset(parts)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              drop_last=True, persistent_workers=True)
    hws_test_loader = make_loader(make(hws_test, False), batch_size=args.batch_size,
                                  shuffle=False, num_workers=2)
    futo_val_loader = make_loader(make(futo_val, False), batch_size=args.batch_size,
                                  shuffle=False, num_workers=2)

    base_hws = greedy_eval(model, hws_test_loader, device, alphabet,
                           args.eval_batches)
    base_futo = greedy_eval(model, futo_val_loader, device, alphabet,
                            args.eval_batches)
    print(f"\nbefore  hws-test wacc={base_hws['wacc']:.4f} cer={base_hws['cer']:.4f}"
          f"   futo-val wacc={base_futo['wacc']:.4f} cer={base_futo['cer']:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0)))
    )

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for x, targets, lengths in train_loader:
            lp = model(x.to(device))
            loss = ctc_loss(lp, targets.to(device), lengths.to(device),
                            model.cfg.blank)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss)
            seen += 1
            step += 1
            if step % 300 == 0:
                print(f"  e{epoch} step {step}/{total} loss {running / seen:.4f}"
                      f" {seen * args.batch_size / (time.time() - t0):.0f} swipes/s")

        hws = greedy_eval(model, hws_test_loader, device, alphabet,
                          args.eval_batches)
        futo = greedy_eval(model, futo_val_loader, device, alphabet,
                           args.eval_batches)
        print(f"epoch {epoch}: loss {running / max(seen, 1):.4f}"
              f"   hws-test wacc={hws['wacc']:.4f} ({hws['wacc'] - base_hws['wacc']:+.4f})"
              f"   futo-val wacc={futo['wacc']:.4f} "
              f"({futo['wacc'] - base_futo['wacc']:+.4f})")

    torch.save({"model": model.state_dict(), "cfg": vars(model.cfg),
                "alphabet": alphabet, "args": vars(args)}, out / "encoder.pt")
    print(f"\nsaved {out / 'encoder.pt'}")
    print("\nReading: a large hws gain means the gap was unseen distribution and "
          "more diverse\ndata fixes it. A small one means it is intrinsic to the "
          "corpus -- sloppier\ngestures or noisier labels -- and data will not "
          "close it.")


if __name__ == "__main__":
    main()
