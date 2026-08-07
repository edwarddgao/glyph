#!/usr/bin/env python3
"""Greedy CTC vs trie-constrained beam search.

The error analysis said 85% of greedy errors are not words and ~half sit one
edit from an in-vocabulary word. This measures what a lexicon actually recovers.

The lexicon is the FUTO training vocabulary with observed counts as a unigram
prior -- the realistic setup, where a keyboard ships with a fixed vocabulary.
That imposes a hard ceiling: any evaluation word outside the lexicon cannot be
produced, so the ceiling is reported alongside every number.

Usage:
    python scripts/eval_decoder.py --checkpoint runs/full/encoder.pt --limit 5000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (  # noqa: E402
    EncoderConfig,
    SwipeCorpus,
    SwipeDataset,
    SwipeEncoder,
    decode,
    make_loader,
)
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402
from swipe_typing.model.lexicon import Lexicon  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["dilations"] = tuple(cfg_dict["dilations"])
    model = SwipeEncoder(EncoderConfig(**cfg_dict))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    a = ckpt.get("args", {}) or {}
    return (model, ckpt.get("alphabet", ALPHABET),
            not a.get("no_key_units", False), a.get("resample_mode", "time"))


@torch.no_grad()
def run_encoder(model, loader, device, alphabet):
    chunks, refs = [], []
    for x, targets, lengths in loader:
        chunks.append(model(x.to(device)).float().cpu().numpy())
        refs.extend(decode.target_strings(targets, lengths, alphabet))
    return np.concatenate(chunks), refs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--splits", nargs="+",
                    default=["futo/validation", "how_we_swipe/test"])
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--beam-widths", nargs="+", type=int, default=[16])
    ap.add_argument("--alpha", type=float, default=0.4, help="unigram weight")
    ap.add_argument("--beta", type=float, default=0.6, help="length bonus")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)

    train_words = SwipeCorpus.load(root / "futo/train", alphabet).words
    lexicon = Lexicon.from_words(train_words)
    print(f"lexicon: {len(lexicon):,} words from the training vocabulary\n")

    for split in args.splits:
        path = root / split
        if not path.exists():
            print(f"[skip] {split}")
            continue
        corpus = SwipeCorpus.load(path, alphabet, limit=args.limit)
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                          key_units=key_units)
        loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)
        log_probs, refs = run_encoder(model, loader, device, alphabet)

        in_lex = sum(r in lexicon for r in refs)
        ceiling = in_lex / len(refs)

        greedy = decode.greedy_decode(torch.from_numpy(log_probs),
                                      model.cfg.blank, alphabet)
        gm = decode.score(greedy, refs)

        print(f"== {split}  n={len(refs):,} ==")
        print(f"  lexicon ceiling (eval words in vocab): {ceiling:6.1%}")
        print(f"  {'decoder':<22}{'CER':>8}{'top-1':>9}{'top-k':>9}{'sec':>8}")
        print(f"  {'greedy (no lexicon)':<22}{gm['cer']:>8.3f}"
              f"{gm['wacc']:>9.3f}{'-':>9}{'-':>8}")

        for width in args.beam_widths:
            cfg = BeamConfig(beam_width=width, alpha=args.alpha,
                             beta=args.beta, top_k=max(args.top_k, 1))
            t0 = time.time()
            top1, topk_hit = [], 0
            for i, item in enumerate(log_probs):
                hyps = beam_search(item, lexicon, alphabet, cfg)
                words = [w for w, _ in hyps]
                top1.append(words[0] if words else greedy[i])
                if refs[i] in words:
                    topk_hit += 1
            dt = time.time() - t0
            bm = decode.score(top1, refs)
            print(f"  {'beam ' + str(width):<22}{bm['cer']:>8.3f}"
                  f"{bm['wacc']:>9.3f}{topk_hit / len(refs):>9.3f}{dt:>8.1f}")

        # What is left after the lexicon has done its work?
        best = top1
        wrong = [(p, r) for p, r in zip(best, refs) if p != r]
        oov = sum(1 for _, r in wrong if r not in lexicon)
        print(f"\n  remaining errors: {len(wrong):,}")
        print(f"    out-of-vocabulary (unreachable): {oov / max(len(wrong), 1):6.1%}")
        print(f"    real-word confusions:            "
              f"{(len(wrong) - oov) / max(len(wrong), 1):6.1%}")
        print("  examples:")
        for p, r in wrong[:6]:
            tag = "OOV" if r not in lexicon else "confusion"
            print(f"    {r:<16} -> {p:<16} [{tag}]")
        print()


if __name__ == "__main__":
    main()
