#!/usr/bin/env python3
"""Lexicon escape: detect when the dictionary is the problem.

The trie can only emit lexicon words, so when a user swipes a word the
lexicon lacks, the search silently substitutes a wrong real word — the one
error class no lexicon growth, encoder, or LM can touch (#36's "unfixable"
bucket). The tell is cheap: score the unconstrained greedy string and the
constrained beam winner by full CTC forward and take the gap

    s = logP_ctc(greedy) - logP_ctc(beam top-1)

When the lexicon contains what the user swiped the two agree; when the
encoder is confidently tracing a spelling the trie won't allow, the
constraint pays a visible cost.

Measured (#42, frozen MMI encoder, 20k per split): the *detector* is
excellent — AUC 0.974 on futo/validation, 0.924 on How We Swipe for
"truth is out-of-lexicon". The *replacement policy* (emit greedy when
s > tau) is dead: out-of-lexicon truths are 1.2%/1.7% of swipes and the
greedy string matches them exactly only 22%/14% (the permutation+MMI
recipe lifts hws to 18%), so the best net top-1 across every threshold is
+0.02/−0.00 and false escapes outnumber recoveries at any useful
sensitivity. The signal's real consumers are a secondary raw-spelling
suggestion and the add-to-dictionary prompt, where a false alarm costs a
candidate slot rather than a correct word — and transcription corpora
understate real out-of-lexicon prevalence (names, slang).

Usage:
    python scripts/eval_lexicon_escape.py --limit 20000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/mmi/encoder_ep0.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--splits", nargs="+",
                    default=["futo/validation", "how_we_swipe/test"])
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[2, 4, 6, 8, 10, 14, 20])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    blank = len(alphabet)
    c2i = {c: i for i, c in enumerate(alphabet)}

    def ctc_score(lp: np.ndarray, word: str) -> float:
        if not word:
            return -1e9
        x = torch.from_numpy(lp).unsqueeze(1)
        tgt = torch.tensor([[c2i[c] for c in word]])
        return -F.ctc_loss(x, tgt, torch.tensor([lp.shape[0]]),
                           torch.tensor([len(word)]), blank=blank,
                           reduction="none").item()

    for split in args.splits:
        corpus = SwipeCorpus.load(root / split, alphabet, limit=args.limit)
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                          key_units=key_units)
        loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)
        chunks = []
        with torch.no_grad():
            for x, _, _ in loader:
                chunks.append(model(x.to(device)).float().cpu().numpy())
        lp = np.concatenate(chunks)
        words = corpus.words
        n = len(words)

        greedy = decode.greedy_decode(torch.from_numpy(lp), blank, alphabet)
        cfg = BeamConfig(top_k=1)
        t0 = time.time()
        beam_top, sig = [], []
        for i in range(n):
            hyps = beam_search(lp[i], lexicon, alphabet, cfg)
            w = hyps[0][0] if hyps else ""
            g = greedy[i]
            beam_top.append(w)
            sig.append(0.0 if g == w
                       else ctc_score(lp[i], g) - ctc_score(lp[i], w))
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{n} ({(i+1)/(time.time()-t0):.0f}/s)",
                      flush=True)

        truth = np.array(words)
        beam = np.array(beam_top)
        grd = np.array(greedy)
        s = np.array(sig)
        ool = np.array([t not in lexicon for t in truth])
        base = (beam == truth).mean()

        print(f"\n=== {split} ===  n={n:,}")
        print(f"baseline beam top-1: {base:.4f}   truth out-of-lexicon: "
              f"{ool.mean():.4f} ({ool.sum()})")
        print(f"greedy exactly right on OOL words: "
              f"{(grd[ool] == truth[ool]).mean():.3f}  <- mechanism ceiling")

        order = np.argsort(s)
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        auc = ((ranks[ool].sum() - ool.sum() * (ool.sum() - 1) / 2)
               / (ool.sum() * (~ool).sum()))
        print(f"signal AUC for 'truth is out-of-lexicon': {auc:.3f}")

        print(f"{'tau':>6} {'escapes':>8} {'recovered':>10} "
              f"{'collateral':>11} {'net top-1':>10} {'delta':>8}")
        for tau in args.taus:
            esc = s > tau
            final = np.where(esc, grd, beam)
            acc = (final == truth).mean()
            rec = (esc & (grd == truth) & (beam != truth)).sum()
            col = (esc & (beam == truth) & (grd != truth)).sum()
            print(f"{tau:>6g} {esc.sum():>8,} {rec:>10,} {col:>11,} "
                  f"{acc:>10.4f} {acc-base:>+8.4f}")


if __name__ == "__main__":
    main()
