#!/usr/bin/env python3
"""Train the sentence-level AR decoder: one decoder, two conditioning streams.

    python scripts/train_sentence_ar.py --epochs 10 --out runs/sent_ar

Matched-budget counterpart of scripts/train_ar_decoder.py (which produced
runs/ar_full): same trunk, same head width and depth, same augmentation, same
number of passes over the same 916k swipes — the only change is that the
token stream crosses word boundaries, so self-attention over the sentence
prefix becomes the language model. Per-epoch metrics are teacher-forced val
loss plus unconstrained greedy sentence decode (decoded context fed back),
the analogue of the word decoder's greedy wacc.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from swipe_typing.model.sentence_ar import (
    SentenceARConfig, SentenceARDecoder, SentenceDataset, collate_sentences,
    greedy_sentences, sentence_loss,
)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(ds, batch_size, shuffle, num_workers):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_sentences,
                      drop_last=shuffle, persistent_workers=num_workers > 0)


def fit_normalization(model, loader, max_batches: int = 40) -> None:
    total = torch.zeros(model.cfg.n_input, dtype=torch.float64)
    total_sq = torch.zeros(model.cfg.n_input, dtype=torch.float64)
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        flat = batch[0].reshape(-1, batch[0].shape[-1]).double()
        total += flat.sum(0)
        total_sq += flat.square().sum(0)
        count += flat.shape[0]
    mean = total / max(count, 1)
    var = (total_sq / max(count, 1) - mean.square()).clamp_min(0)
    model.set_normalization(mean.float(), var.sqrt().float())


@torch.no_grad()
def evaluate(model, loader, device, alphabet, corpus, max_batches=40):
    model.eval()
    loss_sum, tok_hit, tok_n = 0.0, 0, 0
    word_hit, word_n = 0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, sent_id, word_id, tgt_in, tgt_out, tok_word, n_words, idxs = \
            (b.to(device) for b in batch)
        loss = sentence_loss(model, x, sent_id, word_id, tgt_in, tgt_out,
                             tok_word, n_words)
        loss_sum += float(loss)
        from swipe_typing.model.sentence_ar import scatter_memory
        memory = scatter_memory(model, x, sent_id, word_id, len(n_words))
        logits = model.decode_step(memory, tgt_in, tok_word)
        mask = tgt_out >= 0
        tok_hit += int((logits.argmax(-1)[mask] == tgt_out[mask]).sum())
        tok_n += int(mask.sum())
        if i < max_batches // 4:  # greedy decode is the slow part
            sents = greedy_sentences(model, x, sent_id, word_id, n_words,
                                     alphabet)
            flat_idx = idxs.cpu().tolist()
            k = 0
            for words in sents:
                for w in words:
                    word_hit += int(w == corpus.words[flat_idx[k]])
                    k += 1
            word_n += len(flat_idx)
    model.train()
    n_b = min(max_batches, i + 1)
    return {"loss": loss_sum / max(n_b, 1),
            "tok_acc": tok_hit / max(tok_n, 1),
            "wacc": word_hit / max(word_n, 1), "n_words": word_n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="runs/sent_ar")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="sentences per batch (~9.3 swipes each)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--dilations", default="1,2,4,8,1,2,4,8")
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resample-mode", default="time",
                    choices=["time", "arclength"])
    ap.add_argument("--eval-limit", type=int, default=20000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--shape-only", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache)
    kb = KeyboardLayout.qwerty()

    print(f"device: {device}")
    t0 = time.time()
    train_corpus = SwipeCorpus.load(cache_root / "futo/train", kb.letters)
    print(f"train: {len(train_corpus):,} swipes  ({time.time() - t0:.0f}s)")

    from swipe_typing.augment import DEFAULT as DEFAULT_AUG

    cfg = SentenceARConfig(
        n_keys=len(kb.letters),
        d_model=args.d_model,
        dilations=tuple(int(d) for d in args.dilations.split(",")),
        dropout=args.dropout,
        shape_only=args.shape_only,
        dec_layers=args.dec_layers,
    )
    train_ds = SentenceDataset(
        train_corpus, kb, cfg,
        augment_cfg=None if args.no_augment else DEFAULT_AUG,
        resample_mode=args.resample_mode,
        shape_only=args.shape_only,
    )
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    print(f"train: {len(train_ds):,} sentences, "
          f"{len(train_loader):,} steps/epoch")

    evals = {}
    eval_corpora = {}
    for name, rel in [("futo/val", "futo/validation"),
                      ("how_we_swipe", "how_we_swipe/test")]:
        path = cache_root / rel
        if not path.exists():
            continue
        corpus = SwipeCorpus.load(path, kb.letters, limit=args.eval_limit)
        ds = SentenceDataset(corpus, kb, cfg, augment_cfg=None,
                             resample_mode=args.resample_mode,
                             shape_only=args.shape_only)
        evals[name] = make_loader(ds, args.batch_size, False,
                                  max(args.workers // 2, 1))
        eval_corpora[name] = corpus
        print(f"eval {name}: {len(ds):,} sentences")

    model = SentenceARDecoder(cfg).to(device)
    print(f"params: {model.num_parameters():,}")

    print("fitting input normalization...")
    model.cpu()
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
        train_ds.seed = epoch + 1
        model.train()
        running, seen, swipes, t_epoch = 0.0, 0, 0, time.time()

        for batch in train_loader:
            x, sent_id, word_id, tgt_in, tgt_out, tok_word, n_words, _ = \
                (b.to(device, non_blocking=True) for b in batch)
            loss = sentence_loss(model, x, sent_id, word_id, tgt_in, tgt_out,
                                 tok_word, n_words)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += float(loss.detach())
            seen += 1
            swipes += x.shape[0]

            if step % args.log_every == 0:
                rate = swipes / (time.time() - t_epoch)
                print(f"  e{epoch} step {step}/{total_steps} "
                      f"loss {running / seen:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"{rate:.0f} swipes/s", flush=True)

        train_loss = running / max(seen, 1)
        row = {"epoch": epoch, "train_loss": train_loss,
               "secs": round(time.time() - t_epoch, 1)}
        for name, loader in evals.items():
            m = evaluate(model, loader, device, kb.letters,
                         eval_corpora[name], args.eval_batches)
            row[name] = {"loss": round(m["loss"], 4),
                         "tok_acc": round(m["tok_acc"], 4),
                         "wacc": round(m["wacc"], 4), "n_words": m["n_words"]}
        history.append(row)
        print(f"epoch {epoch}: loss {train_loss:.4f}  " + "  ".join(
            f"{k} loss={v['loss']:.3f} tok={v['tok_acc']:.3f} "
            f"wacc={v['wacc']:.3f}"
            for k, v in row.items() if isinstance(v, dict)
        ), flush=True)

        selector = row.get("futo/val", {}).get("loss", train_loss)
        if selector < best:
            best = selector
            torch.save(
                {"model": model.state_dict(), "cfg": vars(cfg),
                 "alphabet": kb.letters, "args": vars(args)},
                out / "sent_ar.pt",
            )
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nsaved {out / 'sent_ar.pt'}")


if __name__ == "__main__":
    main()
