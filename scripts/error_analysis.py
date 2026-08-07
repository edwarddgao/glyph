#!/usr/bin/env python3
"""Where does the encoder actually fail?

The point is to decide whether the encoder is still the bottleneck. Two failure
modes have opposite implications:

  - The prediction is a near-miss of the true word (edit distance 1) or is not a
    real word at all. A lexicon-constrained decoder recovers these. The encoder
    is fine; the decoding layer is missing.

  - The prediction is a *different valid word* that the gesture genuinely
    resembles ("im" -> "in"). No lexicon helps; only a language model or a
    better encoder can separate these -- and for very short words the gestures
    may be near-identical, in which case nothing can.

Usage:
    python scripts/error_analysis.py --checkpoint runs/full/encoder.pt
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
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
def collect(model, loader, device, alphabet):
    preds, refs = [], []
    for x, targets, lengths in loader:
        lp = model(x.to(device))
        preds.extend(decode.greedy_decode(lp, model.cfg.blank, alphabet))
        refs.extend(decode.target_strings(targets, lengths, alphabet))
    return preds, refs


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    --"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)

    # Lexicon = everything the training set ever asked for.
    train_words = set(
        SwipeCorpus.load(root / "futo/train", alphabet, limit=None).words
    )
    print(f"lexicon: {len(train_words):,} training words\n")

    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2)
    preds, refs = collect(model, loader, device, alphabet)

    overall = decode.score(preds, refs)
    n = len(refs)
    wrong = [(p, r) for p, r in zip(preds, refs) if p != r]
    print(f"== {args.split}: n={n:,} cer={overall['cer']:.4f} "
          f"wacc={overall['wacc']:.4f}  ({len(wrong):,} errors) ==\n")

    # --- is a lexicon enough? ------------------------------------------------
    d1 = sum(1 for p, r in wrong if decode.edit_distance(p, r) == 1)
    d2 = sum(1 for p, r in wrong if decode.edit_distance(p, r) == 2)
    not_word = sum(1 for p, _ in wrong if p not in train_words)
    real_word_conf = sum(1 for p, r in wrong
                         if p in train_words and decode.edit_distance(p, r) >= 1)
    # Would a lexicon see the right answer at all? (true word in vocabulary
    # AND within one edit of what the encoder emitted)
    recoverable = sum(
        1 for p, r in wrong
        if r in train_words and r in decode.edits1(p, alphabet)
    )
    print("lexicon leverage (share of errors):")
    print(f"  prediction is not a known word   {pct(not_word, len(wrong))}"
          "   a lexicon rejects it outright")
    print(f"  true word 1 edit away            {pct(d1, len(wrong))}")
    print(f"  true word 2 edits away           {pct(d2, len(wrong))}")
    print(f"  in-vocab AND 1 edit away         {pct(recoverable, len(wrong))}"
          "   <- lexicon+beam very likely recovers")
    print(f"  predicted a different real word  {pct(real_word_conf, len(wrong))}"
          "   needs an LM, not a lexicon")
    print(f"  true word absent from lexicon    "
          f"{pct(sum(1 for _, r in wrong if r not in train_words), len(wrong))}"
          "   a lexicon cannot help")

    # --- where are the errors concentrated? ---------------------------------
    print("\nby word length:")
    by_len = decode.confusion_by_length(preds, refs)
    err_by_len = Counter(len(r) for _, r in wrong)
    print(f"  {'len':>4}{'n':>8}{'wacc':>8}{'CER':>8}{'share of all errors':>21}")
    for ln, m in by_len.items():
        if m["n"] < 20:
            continue
        print(f"  {ln:>4}{m['n']:>8,}{m['wacc']:>8.3f}{m['cer']:>8.3f}"
              f"{pct(err_by_len[ln], len(wrong)):>21}")

    short = sum(v for k, v in err_by_len.items() if k <= 3)
    print(f"\n  words of length <= 3 account for {pct(short, len(wrong)).strip()} "
          f"of all errors")

    # --- seen vs unseen vocabulary ------------------------------------------
    seen = [(p, r) for p, r in zip(preds, refs) if r in train_words]
    unseen = [(p, r) for p, r in zip(preds, refs) if r not in train_words]
    for label, group in (("in training vocab", seen), ("unseen word", unseen)):
        if group:
            m = decode.score([p for p, _ in group], [r for _, r in group])
            print(f"  {label:<20} n={m['n']:>7,}  wacc={m['wacc']:.3f}  "
                  f"cer={m['cer']:.3f}")

    # --- doubled letters -----------------------------------------------------
    def has_double(w):
        return any(a == b for a, b in zip(w, w[1:]))

    dbl = [(p, r) for p, r in zip(preds, refs) if has_double(r)]
    nodbl = [(p, r) for p, r in zip(preds, refs) if not has_double(r)]
    print("\ndoubled letters (CTC collapses repeats -- needs a blank between):")
    for label, group in (("has doubled letter", dbl), ("no doubled letter", nodbl)):
        if group:
            m = decode.score([p for p, _ in group], [r for _, r in group])
            print(f"  {label:<20} n={m['n']:>7,}  wacc={m['wacc']:.3f}")

    # --- which letters, and are they neighbours? -----------------------------
    subs = Counter()
    op_counts = Counter()
    for p, r in wrong:
        for op, pc, rc in decode.align_ops(p, r):
            op_counts[op] += 1
            if op == "sub":
                subs[(rc, pc)] += 1

    total_ops = sum(op_counts[k] for k in ("sub", "ins", "del"))
    print(f"\nedit ops across errors: " + "  ".join(
        f"{k} {pct(op_counts[k], total_ops).strip()}" for k in ("sub", "ins", "del")
    ))

    key_w, key_h = 1 / 10, 1 / 3
    adjacent = 0
    for (rc, pc), c in subs.items():
        if rc in alphabet and pc in alphabet:
            d = np.abs(np.array(kb.center(rc)) - np.array(kb.center(pc)))
            if d[0] <= key_w * 1.5 and d[1] <= key_h * 1.5:
                adjacent += c
    n_subs = sum(subs.values())
    print(f"substitutions between keyboard-adjacent keys: "
          f"{pct(adjacent, n_subs).strip()}")

    print("\ntop substitutions (true -> predicted):")
    for (rc, pc), c in subs.most_common(12):
        d = float(np.linalg.norm(np.array(kb.center(rc)) - np.array(kb.center(pc))))
        print(f"  {rc} -> {pc}   {c:>5}   key distance {d / key_w:.1f} key-widths")

    print("\nmost-missed words:")
    for (p, r), c in Counter(wrong).most_common(12):
        print(f"  {r:<14} -> {p:<14} {c:>4}x"
              f"{'  [pred not a word]' if p not in train_words else ''}")


if __name__ == "__main__":
    main()
