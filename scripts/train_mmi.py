#!/usr/bin/env python3
"""MMI (discriminative) fine-tune: train the emissions for the search.

The head-to-head against the released FUTO Swipe Model encoder (#25) showed
their encoder is weak standalone (48.2% greedy on our validation) but strong
under lexicon-constrained search (90.1%) — they train emissions *for the
decoder they ship*. Our encoder was trained with plain CTC, which maximizes
P(truth|gesture) but never sees the confusions the beam actually surfaces.
This closes that loop:

    loss = -[ log P_ctc(truth) - logsumexp_j log P_ctc(candidate_j) ]

where the candidates are real first-pass n-best lists dumped by
``dump_nbest.py`` from the very checkpoint being fine-tuned. Classic MMI over
an n-best lattice: acoustic mass is pushed onto the truth *relative to the
rivals the search actually produces*, which is what top-1 measures. Truth is
injected where the beam missed it, so the denominator always contains the
numerator and the loss is bounded.

Expect greedy accuracy to *drop* while everything under search rises — that
trade is the point, and it is the same signature the FUTO encoder shows.
Measured (one epoch, LR 5e-5, beam-128/top-16 lists from 150k train swipes):
greedy 80.1 -> 79.0, but beam-64 top-1 92.0 -> 92.3, hit@8 97.0 -> 97.4, and
the full stack 93.81 -> ~94.0 on validation. Cross-corpus held: How We Swipe
zero-shot 80.5 -> 80.6. Zero inference cost — same model, same beam.

Model selection note: greedy validation accuracy is the wrong criterion here
(it moves the other way); checkpoints are saved per epoch and must be judged
by beam top-1 via ``eval_decoder.py``.

Usage:
    python scripts/dump_nbest.py --checkpoint runs/full20/encoder.pt \\
        --split futo/train --limit 150000 --beam-width 128 --top-k 16 \\
        --out runs/rescorer128k16
    python scripts/train_mmi.py --out runs/mmi
    python scripts/eval_decoder.py --checkpoint runs/mmi/encoder_ep0.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import (  # noqa: E402
    SwipeCorpus,
    SwipeDataset,
    decode,
    make_loader,
)

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import load_model, pick_device  # noqa: E402


def encode_words(words, char_index, max_len: int):
    """(N, max_len) int targets + lengths; empty words get length 0."""
    n = len(words)
    ids = np.zeros((n, max_len), dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    for i, w in enumerate(words):
        w = w[:max_len]
        if not w:
            continue
        ids[i, :len(w)] = [char_index[c] for c in w]
        lens[i] = len(w)
    return ids, lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full20/encoder.pt")
    ap.add_argument("--nbest", default="runs/rescorer128k16/futo_train.npz",
                    help="n-best lists dumped from --checkpoint")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="runs/mmi")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-word-len", type=int, default=24)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into an --out that already has "
                         "checkpoints (a rerun is NOT bit-identical: dropout "
                         "RNG differs, so overwriting orphans any numbers "
                         "measured against the old files)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not args.overwrite and any(out_dir.glob("encoder_ep*.pt")):
        sys.exit(f"{out_dir} already holds checkpoints; pass --overwrite "
                 f"or choose a fresh --out")

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    model.train()
    blank = model.cfg.blank
    char_index = {c: i for i, c in enumerate(alphabet)}
    K_MAX = args.max_word_len

    data = np.load(args.nbest, allow_pickle=False)
    cands = data["candidates"].astype(str).copy()
    valid = data["valid"].copy()
    target = data["target"].astype(np.int64).copy()
    words = data["words"].astype(str)
    n, K = cands.shape
    cands[~valid] = ""          # invalid slots are stored as the string "0"

    # Inject truth where the beam missed it, into the last slot.
    missing = target < 0
    cands[missing, K - 1] = [w[:K_MAX] for w in words[missing]]
    valid[missing, K - 1] = True
    target[missing] = K - 1
    print(f"n={n:,}  K={K}  truth injected for {int(missing.sum()):,}")

    cand_ids, cand_lens = encode_words(cands.reshape(-1), char_index, K_MAX)
    cand_ids = cand_ids.reshape(n, K, K_MAX)
    cand_lens = cand_lens.reshape(n, K)
    # A padded slot can't go through ctc_loss with length 0: give it a 1-char
    # dummy and mask its score out instead.
    dummy = cand_lens == 0
    cand_lens[dummy] = 1
    valid = valid & ~dummy

    root = Path(args.cache)
    corpus = SwipeCorpus.load(root / "futo/train", alphabet, limit=n)
    assert list(corpus.words[:100]) == list(words[:100]), \
        "cache order does not match the n-best dump"
    kb = KeyboardLayout.qwerty()
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2)

    val_corpus = SwipeCorpus.load(root / "futo/validation", alphabet,
                                  limit=5000)
    val_ds = SwipeDataset(val_corpus, kb, augment_cfg=None,
                          resample_mode=mode, key_units=key_units)
    val_loader = make_loader(val_ds, batch_size=512, shuffle=False,
                             num_workers=2)

    @torch.no_grad()
    def val_greedy():
        model.eval()
        preds, refs = [], []
        for x, t, ln in val_loader:
            lp = model(x.to(device)).float().cpu()
            preds.extend(decode.greedy_decode(lp, blank, alphabet))
            refs.extend(decode.target_strings(t, ln, alphabet))
        model.train()
        return decode.score(preds, refs)

    m0 = val_greedy()
    print(f"before: greedy val CER {m0['cer']:.4f}  wacc {m0['wacc']:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t_ids = torch.from_numpy(cand_ids)
    t_lens = torch.from_numpy(cand_lens)
    t_valid = torch.from_numpy(valid)
    t_target = torch.from_numpy(target)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    for epoch in range(args.epochs):
        run, seen = 0.0, 0
        for bi, (x, _, _) in enumerate(loader):
            lo = bi * args.batch_size
            hi = min(lo + args.batch_size, n)
            b = hi - lo
            if b <= 0:
                break
            lp = model(x.to(device))                    # (b, T, C) log-probs
            T = lp.shape[1]
            # (T, b*K, C): every candidate of every sample through ctc_loss.
            # MPS has no CTC kernel, so score on CPU; autograd carries the
            # gradient back across the device copy.
            lp_flat = (lp.unsqueeze(1).expand(b, K, T, lp.shape[2])
                       .reshape(b * K, T, -1).transpose(0, 1))
            ids = t_ids[lo:hi].reshape(b * K, K_MAX)
            lens = t_lens[lo:hi].reshape(b * K)
            in_lens = torch.full((b * K,), T, dtype=torch.long)
            nll = F.ctc_loss(lp_flat.cpu().float(), ids, in_lens, lens,
                             blank=blank, reduction="none",
                             zero_infinity=True)
            scores = (-nll).reshape(b, K)
            scores = scores.masked_fill(~t_valid[lo:hi], -1e4)
            loss = F.cross_entropy(scores.to(device),
                                   t_target[lo:hi].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * b
            seen += b
            if (bi + 1) % 200 == 0:
                print(f"  {seen:,}/{n:,}  mmi loss {run / seen:.4f}",
                      flush=True)
        m = val_greedy()
        print(f"epoch {epoch}: mmi {run / max(seen, 1):.4f}  "
              f"greedy val CER {m['cer']:.4f}  wacc {m['wacc']:.4f}")
        ckpt["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(ckpt, out_dir / f"encoder_ep{epoch}.pt")

    print("done. select the epoch by beam top-1 (eval_decoder.py), "
          "not greedy — MMI trades greedy away by design.")


if __name__ == "__main__":
    main()
