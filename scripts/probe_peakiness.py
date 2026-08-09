#!/usr/bin/env python3
"""Does frame-level overconfidence cost the search anything?

The encoder's posteriors are one-hot (81% of frames above 0.999, #16/#39).
This prices what that costs, in two parts, on cached emissions — no
retraining:

1. Post-hoc softening sweep: temperature-scale (or probability-floor) the
   emissions and re-run the trie beam. If no setting raises truth-survival,
   softness is not what the search is missing.

2. Autopsy of in-lexicon misses: score the true word by full CTC forward
   (every alignment, no pruning) against the beam winner, raw and with a
   probability floor. The fraction of misses the floor flips is the damage
   attributable to isolated write-offs — the only part retraining for
   humility could reclaim.

Measured (canonical encoder, 20k futo/validation): softening moves
truth-survival ≤ +0.08 while top-1 falls up to 1.2 points — sharpness is
load-bearing for ranking. The truth loses its 268 in-lexicon misses by a
median 7.9 nats (≈2700×) and a 1e-3 floor flips 12% of them: write-off
damage ≈ 0.16% of swipes, ≈ 0.05 points through the stack. Overconfidence
is behaviorally harmless to the search; confidence work belongs post-hoc.

Usage:
    python scripts/probe_peakiness.py --limit 20000
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
from swipe_typing.model import SwipeCorpus, SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_candidates  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402


def renorm(lp: np.ndarray) -> np.ndarray:
    return lp - torch.logsumexp(torch.from_numpy(lp), dim=-1,
                                keepdim=True).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--temperatures", nargs="+", type=float,
                    default=[1.25, 1.5, 2.0, 3.0])
    ap.add_argument("--floors", nargs="+", type=float, default=[1e-3, 1e-4])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units)
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)

    chunks = []
    with torch.no_grad():
        for x, _, _ in loader:
            chunks.append(model(x.to(device)).float().cpu().numpy())
    lp0 = np.concatenate(chunks)
    words = corpus.words
    n = len(words)
    print(f"emissions {lp0.shape}", flush=True)

    def sweep(name: str, lp: np.ndarray):
        cfg = BeamConfig()
        in_pool = hit8 = top1 = 0
        pools = []
        t0 = time.time()
        for i in range(n):
            cands = beam_candidates(lp[i], lexicon, alphabet, cfg)
            pools.append(cands)
            ranked = sorted(
                cands,
                key=lambda c: c[1] + args.alpha * c[2] + args.beta * c[3],
                reverse=True)
            in_pool += words[i] in {w for w, _, _, _ in cands}
            hit8 += words[i] in {w for w, _, _, _ in ranked[:8]}
            top1 += bool(ranked) and ranked[0][0] == words[i]
        print(f"{name:>14}: in-pool {in_pool/n:.4f}   hit@8 {hit8/n:.4f}   "
              f"top-1 {top1/n:.4f}   ({time.time()-t0:.0f}s)", flush=True)
        return pools

    print("\n== post-hoc softening sweep ==", flush=True)
    pools_base = sweep("T=1 baseline", lp0)
    for t in args.temperatures:
        sweep(f"T={t}", renorm(lp0 / t))
    floored = {p: renorm(np.logaddexp(lp0, np.log(p))) for p in args.floors}
    for p, lp in floored.items():
        sweep(f"floor {p:g}", lp)

    print("\n== autopsy: in-lexicon words the search never surfaced ==",
          flush=True)
    blank = len(alphabet)
    c2i = {c: i for i, c in enumerate(alphabet)}

    def ctc_score(lp: np.ndarray, word: str) -> float:
        x = torch.from_numpy(lp).unsqueeze(1)
        tgt = torch.tensor([[c2i[c] for c in word]])
        return -F.ctc_loss(x, tgt, torch.tensor([lp.shape[0]]),
                           torch.tensor([len(word)]), blank=blank,
                           reduction="none").item()

    miss_idx = [i for i, cands in enumerate(pools_base)
                if words[i] not in {w for w, _, _, _ in cands}
                and words[i] in lexicon]
    print(f"misses with truth in lexicon: {len(miss_idx)} of {n}", flush=True)

    gaps = {"raw": []}
    flips = {p: 0 for p in args.floors}
    for p in args.floors:
        gaps[f"floor {p:g}"] = []
    for i in miss_idx:
        cands = pools_base[i]
        if not cands:
            continue
        winner = max(cands,
                     key=lambda c: c[1] + args.alpha * c[2] + args.beta * c[3])[0]
        gaps["raw"].append(ctc_score(lp0[i], winner)
                           - ctc_score(lp0[i], words[i]))
        for p in args.floors:
            g = (ctc_score(floored[p][i], winner)
                 - ctc_score(floored[p][i], words[i]))
            gaps[f"floor {p:g}"].append(g)
            flips[p] += g <= 0

    m = len(gaps["raw"])
    print("\nwinner-minus-truth gap (full CTC forward, nats):")
    for name, g in gaps.items():
        g = np.array(g)
        print(f"  {name:>12}: median {np.median(g):7.1f}   "
              f"p25 {np.percentile(g, 25):7.1f}   "
              f"p75 {np.percentile(g, 75):7.1f}")
    for p in args.floors:
        print(f"truth overtakes winner with floor {p:g}: {flips[p]}/{m} "
              f"({100*flips[p]/m:.1f}%)")


if __name__ == "__main__":
    main()
