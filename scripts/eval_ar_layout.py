#!/usr/bin/env python3
"""Layout transfer for the AR decoder: swipe-5's held-out layouts, AR beam.

    python scripts/eval_ar_layout.py --checkpoint runs/ar_full/ar_decoder.pt

The #22/#38 exam ported to the fused-era first pass. Real gestures performed
on layouts the model never trained on (qwertz, clearflow, kasroz, dvorak…),
featurized against each layout's own key geometry, decoded by the
trie-constrained AR beam with the honest blended lexicon. The mechanism under
test is the same as #38's: affinity is layout-relative, so transfer holds
exactly to the degree the model avoided memorizing qwerty-conditional
letter statistics — which for the AR head includes its own decoder prior.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, decode, make_loader  # noqa: E402
from swipe_typing.model.ar import FlatTrie, ar_beam, greedy_decode  # noqa: E402
from swipe_typing.sources import futo  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, pick_device  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from eval_layout_beam import group_by_layout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_full/ar_decoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min-swipes", type=int, default=200)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, mode = load_ar(args.checkpoint, device)
    print(f"checkpoint {args.checkpoint}  ({model.num_parameters():,} params)")

    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    trie = FlatTrie(lexicon, alphabet)
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
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                          shape_only=model.cfg.shape_only)
        loader = make_loader(ds, batch_size=256, shuffle=False, num_workers=2)
        refs = list(corpus.words)

        feats = []
        preds = []
        with torch.no_grad():
            for x, _, _ in loader:
                feats.append(x)
                preds.extend(greedy_decode(model, x.to(device), alphabet))
        gm = decode.score(preds, refs)
        feats = torch.cat(feats)
        ceiling = sum(r in lexicon for r in refs) / len(refs)

        t0 = time.time()
        top1, hit = [], 0
        for s in range(0, len(feats), args.chunk):
            lists = ar_beam(model, feats[s:s + args.chunk].to(device), trie,
                            alphabet, beam_width=args.beam_width)
            for j, cl in enumerate(lists):
                i = s + j
                ranked = sorted(
                    cl, key=lambda c: c[1] + args.alpha * c[2]
                    + args.beta * c[3], reverse=True)[:args.top_k]
                words = [w for w, *_ in ranked]
                top1.append(words[0] if words else preds[i])
                hit += refs[i] in words
        dt = time.time() - t0
        bm = decode.score(top1, refs)
        print(f"== {name}  n={len(refs):,}  lexicon ceiling {ceiling:6.1%} ==")
        print(f"  greedy {gm['wacc']:.3f}  beam {bm['wacc']:.3f}  "
              f"hit@{args.top_k} {hit / len(refs):.3f}  ({dt:.0f}s)")
        print()


if __name__ == "__main__":
    main()
