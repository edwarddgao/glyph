#!/usr/bin/env python3
"""Export deep AR candidate lists + ILM scores for the fused-search harness.

    python scripts/export_fused_bundle.py --out fused_bundle.pkl

The fused sentence-beam (scripts/modal_fused_search.py) needs, per swipe,
*all* AR beam survivors — not the top-8 the stack currently truncates to —
with their score components kept separate so the GPU side can recombine:

    per candidate: (word, ar_logp, unigram_logp, length)
    per unique word: ILM score under zero and mean memory ablation

Everything textual, so the bundle stays a few MB and the GPU containers need
no repo, corpus, or encoder — the modal_rerank.py pattern.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, pick_device  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from eval_reranker import order_by_sentence  # noqa: E402
from probe_ilm_fusion import ilm_scores  # noqa: E402
from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.ar import FlatTrie, ar_beam  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_mmi/ar_decoder_ep0.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--max-cands", type=int, default=64)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", default="fused_bundle.pkl")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, mode = load_ar(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    trie = FlatTrie(lexicon, alphabet)

    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=4)
    feats = torch.cat([x for x, _, _ in loader])
    n = len(corpus)
    print(f"{args.split}: {n:,} swipes")

    lists: list[list[tuple[str, float, float, int]]] = []
    t0 = time.time()
    for s in range(0, n, args.chunk):
        x = feats[s:s + args.chunk].to(device)
        out = ar_beam(model, x, trie, alphabet, beam_width=args.beam_width)
        lists.extend(cl[:args.max_cands] for cl in out)
        if (s // args.chunk) % 50 == 0:
            rate = (s + len(out)) / (time.time() - t0)
            print(f"  {s + len(out):,}/{n:,}  {rate:.0f}/s", flush=True)

    depth = np.array([len(cl) for cl in lists])
    inlist = sum(any(w == corpus.words[i] for w, *_ in cl)
                 for i, cl in enumerate(lists)) / n
    print(f"  candidates/swipe: mean {depth.mean():.1f} max {depth.max()}  "
          f"truth-in-list {inlist:.4f}")

    uniq = sorted({w for cl in lists for w, *_ in cl})
    print(f"  unique candidates: {len(uniq):,}; scoring ILM...")
    T, d = 64, model.cfg.d_model
    zero_mem = torch.zeros(1, T, d, device=device)
    mean_mem = model.encode(feats[:2048].to(device)).mean(dim=0, keepdim=True)
    ilm = {
        "zero": ilm_scores(model, uniq, alphabet, device, zero_mem),
        "mean": ilm_scores(model, uniq, alphabet, device, mean_mem),
    }

    bundle = {
        "checkpoint": args.checkpoint,
        "lists": lists,
        "ilm": ilm,
        "refs": list(corpus.words),
        "sentences": list(corpus.sentences),
        "word_idx": [int(i) for i in corpus.word_idx],
        "groups": [list(map(int, g)) for g in order_by_sentence(corpus)],
    }
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"wrote {args.out} "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
