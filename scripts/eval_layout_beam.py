#!/usr/bin/env python3
"""Layout transfer WITH the trie-constrained beam.

The layout-transfer table in the README's encoder section is lexicon-free
greedy decode, by design: it isolates the encoder. But nothing ships greedy,
and the FUTO paper's held-out-layout numbers (96.84% clearflow, 97.68% kasroz)
are beam-100 + a 162k-word trie, so comparing them against our greedy numbers
confounds decode mode with model quality.

This script scores the same held-out swipe-5 layouts through the full first
pass — trie-constrained beam + blended lexicon — so the comparison is
decoder-vs-decoder. Result (20-epoch encoder, beam 100, train+wf320k):

    layout       greedy   beam 100   vs swipe-5 qwerty
    qwerty        72.8%      84.3%    (same-corpus baseline)
    qwertz        68.0%      86.1%    +1.8
    clearflow     49.0%      83.4%    -0.9
    kasroz        53.5%      84.0%    -0.3
    dvorak        40.4%      79.4%    -4.9

The 20-30 point "geometry gap" at greedy was mostly a decode-mode artifact:
the encoder's logits on 5-row grids are noisier, but the information survives
and the trie recovers it. What remains real: dvorak (-4.9) and the ~8-point
corpus shift between swipe-5 and futo/validation donors.

Usage:
    python scripts/eval_layout_beam.py --checkpoint runs/full20/encoder.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, decode, make_loader  # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402
from swipe_typing.sources import futo  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device, run_encoder  # noqa: E402


def group_by_layout(path, language: str, alphabet: str):
    """Bucket swipe-5 records by the layout they were performed on."""
    letters = set(alphabet)
    buckets = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if language and rec.get("language") != language:
                continue
            sw = futo.parse_record(rec, split="test", source="futo/swipe-5")
            if sw is None or not letters.issuperset(sw.word):
                continue
            buckets[rec.get("layout") or "?"].append(sw)
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full20/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min-swipes", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--beam-widths", nargs="+", type=int, default=[32, 100])
    ap.add_argument("--alpha", type=float, default=0.8, help="unigram weight")
    ap.add_argument("--beta", type=float, default=1.2, help="length bonus")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    print(f"checkpoint {args.checkpoint}  ({model.num_parameters():,} params)")
    print(f"features: key_units={key_units} resample={mode}")

    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    print(f"lexicon '{args.lexicon}': {len(lexicon):,} words\n")

    print("loading swipe-5 ...")
    buckets = group_by_layout(
        futo.download("swipe-5", "train"), args.language, alphabet)

    for name, swipes in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(swipes) < args.min_swipes:
            continue
        try:
            kb = KeyboardLayout.from_futo_json(futo.download_layout(name))
            kb = kb.reindex(alphabet)
        except (KeyError, OSError) as exc:
            print(f"  [skip] {name}: {exc}")
            continue

        corpus = SwipeCorpus.from_swipes(swipes, alphabet, origin=name)
        ds = SwipeDataset(corpus, kb, augment_cfg=None,
                          resample_mode=mode, key_units=key_units)
        loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)
        log_probs, refs = run_encoder(model, loader, device, alphabet)

        greedy = decode.greedy_decode(torch.from_numpy(log_probs),
                                      model.cfg.blank, alphabet)
        gm = decode.score(greedy, refs)
        ceiling = sum(r in lexicon for r in refs) / len(refs)

        print(f"== {name}  n={len(refs):,}  lexicon ceiling {ceiling:6.1%} ==")
        print(f"  {'decoder':<22}{'CER':>8}{'top-1':>9}{'hit@k':>9}{'sec':>8}")
        print(f"  {'greedy (no lexicon)':<22}{gm['cer']:>8.3f}"
              f"{gm['wacc']:>9.3f}{'-':>9}{'-':>8}")

        for width in args.beam_widths:
            cfg = BeamConfig(beam_width=width, alpha=args.alpha,
                             beta=args.beta, top_k=max(args.top_k, 1))
            t0 = time.time()
            top1, hit = [], 0
            for i, item in enumerate(log_probs):
                hyps = beam_search(item, lexicon, alphabet, cfg)
                words = [w for w, _ in hyps]
                top1.append(words[0] if words else greedy[i])
                hit += refs[i] in words
            dt = time.time() - t0
            bm = decode.score(top1, refs)
            print(f"  {'beam ' + str(width):<22}{bm['cer']:>8.3f}"
                  f"{bm['wacc']:>9.3f}{hit / len(refs):>9.3f}{dt:>8.1f}")
        print()


if __name__ == "__main__":
    main()
