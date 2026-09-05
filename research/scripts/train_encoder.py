#!/usr/bin/env python3
"""Train the layout-agnostic swipe encoder with CTC.

    python scripts/train_encoder.py --epochs 6

Metrics are lexicon-free on purpose (greedy CTC decode -> CER and exact-match
word accuracy). A trie-constrained beam search would score far higher, but it
would also let a strong lexicon paper over a weak encoder. These numbers are the
encoder standing on its own.

Evaluation runs on up to three sets:
  futo/validation      held-out donor sessions, same corpus, same layout
  how_we_swipe/test    different corpus, different devices, different users
  --alt-layout-eval    real gestures on layouts never seen in training
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

# torch has no MPS kernel for aten::_ctc_loss, so that one op falls back to CPU.
# Must be set before torch is imported. The tensor it ferries is small next to
# the conv stack (B x 64 x 27), so the round trip is not the bottleneck --
# measured at roughly a 15% step-time cost against a pure-CPU run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import (
    EncoderConfig,
    SwipeCorpus,
    SwipeDataset,
    SwipeEncoder,
    ctc_loss,
    decode,
    make_loader,
)
from swipe_typing.model.encoder import fit_normalization


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_eval(path: Path, kb: KeyboardLayout, limit: int | None, batch_size: int,
               workers: int, mode: str, key_units: bool,
               shape_only: bool = False):
    if not path.exists():
        return None
    corpus = SwipeCorpus.load(path, kb.letters, limit=limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units, shape_only=shape_only)
    return make_loader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=workers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="runs/encoder")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--dilations", default="1,2,4,8,1,2",
                    help="comma-separated dilation per residual block")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resample-mode", default="time",
                    choices=["time", "arclength"])
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=20000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--permute-prob", type=float, default=0.0,
                    help="fraction of training samples relabelled under a "
                         "random letter permutation of the layout, to dilute "
                         "the encoder's implicit LM (eval is never permuted)")
    ap.add_argument("--shape-only", action="store_true",
                    help="translation/scale-invariant ablation: per-gesture "
                         "normalized 8-channel shape features replace the key "
                         "affinity block, so the model never sees where on "
                         "the keyboard a gesture happened or how big it was")
    ap.add_argument("--no-key-units", action="store_true",
                    help="measure motion in grid-heights instead of keys "
                         "(ablation: puts layouts with different row counts on "
                         "different velocity scales; the transfer cost of this "
                         "has not been measured)")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache)
    kb = KeyboardLayout.qwerty()

    print(f"device: {device}")
    t0 = time.time()
    train_corpus = SwipeCorpus.load(cache_root / "futo/train", kb.letters,
                                    limit=args.train_limit)
    print(f"train: {len(train_corpus):,} swipes  ({time.time() - t0:.0f}s)")

    from swipe_typing.augment import DEFAULT as DEFAULT_AUG

    train_ds = SwipeDataset(
        train_corpus, kb,
        augment_cfg=None if args.no_augment else DEFAULT_AUG,
        resample_mode=args.resample_mode,
        key_units=not args.no_key_units,
        permute_prob=args.permute_prob,
        shape_only=args.shape_only,
    )
    train_loader = make_loader(train_ds, batch_size=args.batch_size,
                               num_workers=args.workers)

    evals = {}
    for name, rel in [("futo/val", "futo/validation"),
                      ("how_we_swipe", "how_we_swipe/test")]:
        loader = build_eval(cache_root / rel, kb, args.eval_limit,
                            args.batch_size, max(args.workers // 2, 1),
                            args.resample_mode, not args.no_key_units,
                            shape_only=args.shape_only)
        if loader is not None:
            evals[name] = loader
            print(f"eval {name}: {len(loader.dataset):,} swipes")

    cfg = EncoderConfig(
        n_keys=len(kb.letters),
        d_model=args.d_model,
        dilations=tuple(int(d) for d in args.dilations.split(",")),
        dropout=args.dropout,
        shape_only=args.shape_only,
    )
    model = SwipeEncoder(cfg).to(device)
    print(f"params: {model.num_parameters():,}")

    print("fitting input normalization...")
    fit_normalization(model, train_loader)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    history = []
    step = 0
    best = float("inf")

    for epoch in range(args.epochs):
        # Reseed so each epoch draws fresh augmentations.
        train_ds.seed = epoch + 1
        model.train()
        running, seen, t_epoch = 0.0, 0, time.time()

        for x, targets, lengths in train_loader:
            x = x.to(device, non_blocking=True)
            log_probs = model(x)
            loss = ctc_loss(log_probs, targets.to(device), lengths.to(device),
                            cfg.blank)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += float(loss)
            seen += 1

            if step % args.log_every == 0:
                rate = seen * args.batch_size / (time.time() - t_epoch)
                print(f"  e{epoch} step {step}/{total_steps} "
                      f"loss {running / seen:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"{rate:.0f} swipes/s")

        train_loss = running / max(seen, 1)
        row = {"epoch": epoch, "train_loss": train_loss,
               "secs": round(time.time() - t_epoch, 1)}
        for name, loader in evals.items():
            m = decode.evaluate(model, loader, device, kb.letters,
                                max_batches=args.eval_batches)
            row[name] = {"cer": round(m["cer"], 4), "wacc": round(m["wacc"], 4),
                         "n": m["n"]}
        history.append(row)
        print(f"epoch {epoch}: loss {train_loss:.4f}  " + "  ".join(
            f"{k} cer={v['cer']:.3f} wacc={v['wacc']:.3f}"
            for k, v in row.items() if isinstance(v, dict)
        ))

        # Select on held-out CER, not train loss -- the two diverge once the
        # model starts fitting donor-specific motor habits.
        selector = row.get("futo/val", {}).get("cer", train_loss)
        if selector < best:
            best = selector
            torch.save(
                {"model": model.state_dict(), "cfg": vars(cfg),
                 "alphabet": kb.letters, "args": vars(args)},
                out / "encoder.pt",
            )
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nsaved {out / 'encoder.pt'}")
    for name, loader in evals.items():
        m = decode.evaluate(model, loader, device, kb.letters, collect=8)
        print(f"\n== {name}: cer={m['cer']:.4f} wacc={m['wacc']:.4f} "
              f"n={m['n']} ==")
        for ref, pred in m.get("samples", []):
            flag = " " if ref == pred else "x"
            print(f"  {flag} {ref:<18} -> {pred}")


if __name__ == "__main__":
    main()
