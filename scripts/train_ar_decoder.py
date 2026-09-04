#!/usr/bin/env python3
"""Train the autoregressive letter decoder (cross-entropy, teacher forcing).

    python scripts/train_ar_decoder.py --shape-only --d-model 128 \
        --dilations 1,2,4,8,1,2,4,8 --epochs 10 --out runs/ar_shape10

The CTC coupling probe (#44) showed that on translation/scale-invariant input
the letter posterior is coupled-multimodal and CTC's factorized emissions
cannot express it. This trains the same TCN trunk with an AR decoder head,
which can. Per-epoch metrics are unconstrained greedy AR decode -- the
analogue of CTC greedy, lexicon-free on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus, SwipeDataset, decode, make_loader
from swipe_typing.model.ar import ARConfig, ARSwipeDecoder, ar_loss, greedy_decode
from swipe_typing.model.encoder import fit_normalization


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, alphabet, max_batches=40):
    model.eval()
    preds, refs = [], []
    for i, (x, targets, lengths) in enumerate(loader):
        if i >= max_batches:
            break
        preds.extend(greedy_decode(model, x.to(device), alphabet))
        refs.extend(decode.target_strings(targets, lengths, alphabet))
    model.train()
    m = decode.score(preds, refs)
    m["n"] = len(refs)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--train-path", default="futo/train",
                    help="training corpus, relative to --cache; lets a "
                         "synthetic corpus (e.g. minjerk/train) stand in "
                         "for the real one. Comma-separated specs mix "
                         "corpora, an optional :N suffix truncates one — "
                         "'futo/train:9000,minjerk_rand/train' is the "
                         "1%%-real mixing arm")
    ap.add_argument("--init", default=None,
                    help="checkpoint to initialize weights from (synthetic "
                         "pretrain -> real fine-tune); normalization "
                         "buffers come with it and are not refit")
    ap.add_argument("--out", default="runs/ar")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--dilations", default="1,2,4,8,1,2,4,8")
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--dec-ffn", type=int, default=None,
                    help="decoder FFN width; default 2*d_model (256 at the "
                         "baseline width, i.e. unchanged)")
    ap.add_argument("--dec-heads", type=int, default=None,
                    help="decoder attention heads; default keeps head dim "
                         "32 (4 at the baseline width, i.e. unchanged)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resample-mode", default="time",
                    choices=["time", "arclength"])
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=20000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--shape-only", action="store_true")
    ap.add_argument("--permute-prob", type=float, default=0.0,
                    help="fraction of training samples relabelled under a "
                         "random letter permutation of the layout (#37's "
                         "recipe; dilutes the implicit LM in both the "
                         "affinity reading and the AR head)")
    ap.add_argument("--anchor-jitter", type=float, default=0.0,
                    help="std (keys) of gesture-only translation during "
                         "training; the layout stays put. Partial-invariance "
                         "knob: 0 = exact anchor, large = shape-only-ish")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed init, dropout and data order; also offsets the "
                         "per-epoch augmentation seed by 1000*seed. Runs before "
                         "this flag existed were unseeded (augmentation seed "
                         "offset 0).")
    args = ap.parse_args()

    device = pick_device(args.device)
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    aug_offset = 1000 * (args.seed or 0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache)
    kb = KeyboardLayout.qwerty()

    print(f"device: {device}")
    t0 = time.time()

    def resolve_layout(name: str) -> KeyboardLayout:
        if not name or name == "qwerty":
            return kb
        from swipe_typing.sources import futo
        return KeyboardLayout.from_futo_json(
            futo.download_layout(name)).reindex(kb.letters)

    from swipe_typing.augment import DEFAULT as DEFAULT_AUG

    # Spec grammar: path[@layout][:N]. The layout is what the sample is
    # featurized against — mixing specs with different layouts trains one
    # decoder across keyboards (each dataset yields identical feature/target
    # shapes, so plain concatenation works).
    train_sets, n_total = [], 0
    for spec in args.train_path.split(","):
        path, _, n = spec.partition(":")
        path, _, layout_name = path.partition("@")
        corpus = SwipeCorpus.load(cache_root / path, kb.letters,
                                  limit=int(n) if n else args.train_limit)
        train_sets.append(SwipeDataset(
            corpus, resolve_layout(layout_name),
            augment_cfg=None if args.no_augment else DEFAULT_AUG,
            resample_mode=args.resample_mode,
            shape_only=args.shape_only,
            anchor_jitter=args.anchor_jitter,
            permute_prob=args.permute_prob,
        ))
        n_total += len(corpus)
        print(f"  {spec}: {len(corpus):,} swipes")
    print(f"train: {n_total:,} swipes  ({time.time() - t0:.0f}s)")
    train_ds = (train_sets[0] if len(train_sets) == 1
                else torch.utils.data.ConcatDataset(train_sets))
    train_loader = make_loader(train_ds, batch_size=args.batch_size,
                               num_workers=args.workers)

    evals = {}
    for name, rel in [("futo/val", "futo/validation"),
                      ("how_we_swipe", "how_we_swipe/test")]:
        path = cache_root / rel
        if not path.exists():
            continue
        corpus = SwipeCorpus.load(path, kb.letters, limit=args.eval_limit)
        ds = SwipeDataset(corpus, kb, augment_cfg=None,
                          resample_mode=args.resample_mode,
                          shape_only=args.shape_only)
        evals[name] = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=max(args.workers // 2, 1))
        print(f"eval {name}: {len(ds):,} swipes")

    cfg = ARConfig(
        n_keys=len(kb.letters),
        d_model=args.d_model,
        dilations=tuple(int(d) for d in args.dilations.split(",")),
        dropout=args.dropout,
        shape_only=args.shape_only,
        dec_layers=args.dec_layers,
        dec_ffn=args.dec_ffn if args.dec_ffn else 2 * args.d_model,
        dec_heads=args.dec_heads if args.dec_heads else max(4, args.d_model // 32),
    )
    model = ARSwipeDecoder(cfg).to(device)
    print(f"params: {model.num_parameters():,}")

    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.to(device)
        print(f"initialized from {args.init}")
    else:
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
        for ds in train_sets:
            ds.seed = epoch + 1 + aug_offset
        model.train()
        running, seen, t_epoch = 0.0, 0, time.time()

        for x, targets, lengths in train_loader:
            loss = ar_loss(model, x.to(device, non_blocking=True),
                           targets, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += float(loss.detach())
            seen += 1

            if step % args.log_every == 0:
                rate = seen * args.batch_size / (time.time() - t_epoch)
                print(f"  e{epoch} step {step}/{total_steps} "
                      f"loss {running / seen:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"{rate:.0f} swipes/s", flush=True)

        train_loss = running / max(seen, 1)
        row = {"epoch": epoch, "train_loss": train_loss,
               "secs": round(time.time() - t_epoch, 1)}
        for name, loader in evals.items():
            m = evaluate(model, loader, device, kb.letters, args.eval_batches)
            row[name] = {"cer": round(m["cer"], 4), "wacc": round(m["wacc"], 4),
                         "n": m["n"]}
        history.append(row)
        print(f"epoch {epoch}: loss {train_loss:.4f}  " + "  ".join(
            f"{k} cer={v['cer']:.3f} wacc={v['wacc']:.3f}"
            for k, v in row.items() if isinstance(v, dict)
        ), flush=True)

        selector = row.get("futo/val", {}).get("cer", train_loss)
        if selector < best:
            best = selector
            torch.save(
                {"model": model.state_dict(), "cfg": vars(cfg),
                 "alphabet": kb.letters, "args": vars(args)},
                out / "ar_decoder.pt",
            )
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nsaved {out / 'ar_decoder.pt'}")


if __name__ == "__main__":
    main()
