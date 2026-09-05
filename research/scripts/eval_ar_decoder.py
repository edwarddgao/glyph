#!/usr/bin/env python3
"""Evaluate the AR decoder: greedy, trie-constrained beam, coupling probe.

    python scripts/eval_ar_decoder.py --checkpoint runs/ar_shape10/ar_decoder.pt

Mirrors eval_decoder.py so the numbers are directly comparable to the CTC
path. The beam exposes (ar_logp, unigram_logp, length) per candidate, so
alpha/beta are swept in memory. Note the AR decoder is trained with
cross-entropy on corpus words, so unlike CTC it has an *explicit* route to the
unigram prior; expect the alpha optimum near zero.

The probe re-runs #44's congruence test with sequence-level scores: a decoder
that can express coupling should rank the coherent twin (od) above the
incoherent cross-terms (id, os) on `is` gestures.
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
from swipe_typing.model import SwipeCorpus, SwipeDataset, decode, make_loader  # noqa: E402
from swipe_typing.model.ar import (  # noqa: E402
    ARConfig,
    ARSwipeDecoder,
    FlatTrie,
    ar_beam,
    greedy_decode,
    score_words,
)

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, pick_device  # noqa: E402

PROBES = {
    "is": ["od", "id", "os", "of"],
    "on": ["in"],
    "he": ["gw", "ge", "hw"],
}


def load_ar(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["dilations"] = tuple(cfg_dict["dilations"])
    model = ARSwipeDecoder(ARConfig(**cfg_dict))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    a = ckpt.get("args", {}) or {}
    return model, ckpt.get("alphabet", ALPHABET), a.get("resample_mode", "time")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_shape10/ar_decoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--splits", nargs="+",
                    default=["futo/validation", "how_we_swipe/test"])
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=64,
                    help="swipes decoded per beam batch (memory-bound)")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.2, 0.4, 0.8])
    ap.add_argument("--betas", nargs="+", type=float,
                    default=[0.0, 0.6, 1.2])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--save-preds", default=None,
                    help="npz prefix; writes <prefix>_<split>.npz with refs, "
                         "greedy and beam top-1 words in compare_hyps.py's "
                         "format (use --lag beam / --lag greedy)")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, mode = load_ar(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)

    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    print(f"lexicon '{args.lexicon}': {len(lexicon):,} words")
    t0 = time.time()
    trie = FlatTrie(lexicon, alphabet)
    print(f"flat trie: {len(trie.is_word):,} nodes ({time.time() - t0:.1f}s)\n")

    for split in args.splits:
        path = root / split
        if not path.exists():
            print(f"[skip] {split}")
            continue
        corpus = SwipeCorpus.load(path, alphabet, limit=args.limit)
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                          n_points=model.cfg.n_frames,
                          shape_only=model.cfg.shape_only)
        loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)
        refs = list(corpus.words)

        greedy_preds = []
        feats = []
        with torch.no_grad():
            for x, _, _ in loader:
                feats.append(x)
                greedy_preds.extend(greedy_decode(model, x.to(device), alphabet))
        gm = decode.score(greedy_preds, refs)

        in_lex = sum(r in lexicon for r in refs)
        print(f"== {split}  n={len(refs):,} ==")
        print(f"  lexicon ceiling: {in_lex / len(refs):6.1%}")
        print(f"  greedy (no lexicon)   CER {gm['cer']:.3f}  top-1 {gm['wacc']:.3f}")

        feats = torch.cat(feats)
        t0 = time.time()
        cands: list[list[tuple[str, float, float, int]]] = []
        for s in range(0, len(feats), args.chunk):
            cands.extend(ar_beam(model, feats[s:s + args.chunk].to(device),
                                 trie, alphabet, beam_width=args.beam_width))
        dt = time.time() - t0
        surv = sum(any(w == refs[i] for w, *_ in c)
                   for i, c in enumerate(cands)) / len(refs)
        print(f"  beam {args.beam_width}: {dt:.0f}s, "
              f"truth among surviving candidates: {surv:.4f}")

        best = (0.0, None, None, None)
        for a in args.alphas:
            for b in args.betas:
                top1 = 0
                for i, cl in enumerate(cands):
                    if not cl:
                        continue
                    w = max(cl, key=lambda c: c[1] + a * c[2] + b * c[3])[0]
                    top1 += w == refs[i]
                if top1 / len(refs) > best[0]:
                    best = (top1 / len(refs), a, b, None)
        t1, a, b, _ = best
        ranks = []
        top1_words = []
        for i, cl in enumerate(cands):
            ranked = sorted(cl, key=lambda c: c[1] + a * c[2] + b * c[3],
                            reverse=True)[:args.top_k]
            words = [w for w, *_ in ranked]
            top1_words.append(words[0] if words else greedy_preds[i])
            ranks.append(words.index(refs[i]) if refs[i] in words else -1)
        bm = decode.score(top1_words, refs)
        print(f"  beam top-1 (alpha={a}, beta={b}): {bm['wacc']:.4f}  "
              f"CER {bm['cer']:.3f}")
        cells = "  ".join(
            f"@{k}:{sum(0 <= r < k for r in ranks) / len(ranks):.3f}"
            for k in (1, 2, 4, 8) if k <= args.top_k
        )
        print(f"  oracle over n-best:  {cells}")

        if args.save_preds:
            tag = split.replace("/", "_")
            np.savez(f"{args.save_preds}_{tag}.npz",
                     refs=np.array(refs, dtype=object),
                     greedy=np.array(greedy_preds, dtype=object),
                     beam=np.array(top1_words, dtype=object),
                     ranks=np.array(ranks, dtype=np.int64),
                     alpha=a, beta=b)
        wrong = [(p, r) for p, r in zip(top1_words, refs) if p != r]
        oov = sum(1 for _, r in wrong if r not in lexicon)
        print(f"  remaining errors: {len(wrong):,}  "
              f"(OOV {oov / max(len(wrong), 1):.1%})")
        for p, r in wrong[:6]:
            print(f"    {r:<16} -> {p}")
        print()

        if split == "futo/validation":
            print("== coupling probe (sequence scores, mean dLogP truth-cand) ==")
            idx_by_word = {w: [] for w in PROBES}
            for i, w in enumerate(refs):
                if w in idx_by_word and len(idx_by_word[w]) < 100:
                    idx_by_word[w].append(i)
            for w, cands_p in PROBES.items():
                x = feats[idx_by_word[w]].to(device)
                st = score_words(model, x, [w] * len(x), alphabet)
                for cand in cands_p:
                    sc = score_words(model, x, [cand] * len(x), alphabet)
                    d = (st - sc).cpu().numpy()
                    print(f"  {w:<6}{cand:<6}{d.mean():>10.1f} nats   "
                          f"cand wins {(d < 0).mean():5.1%}   n={len(d)}")
            print()


if __name__ == "__main__":
    main()
