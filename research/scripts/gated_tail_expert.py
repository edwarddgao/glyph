#!/usr/bin/env python3
"""Harvest the tail fine-tune's coverage gain without its mid-frequency tax:
a count-gated two-expert first pass over the saved deep lists.

    python scripts/gated_tail_expert.py --split hws
    python scripts/gated_tail_expert.py --split val

#77's sweep showed that every fine-tuning recipe that adds synthetic tail
gestures pays on words with 6-500 real training gestures -- part forgetting
(fixable by replay), part new competitors, part plain continued-training
drift cross-corpus -- and that no external prior refunds it. The tail gain
itself is real and large where the tail is in-lexicon (HWS zero-count words
+9.6). The two effects live on disjoint word sets, so gate by the word's real
training count instead of blending the models:

    score(w) = ar_base(w)             if count(w) >= K
             = ar_tail(w) + delta     otherwise
             + BETA*len + alpha*uni - lam*ilm_model(w)

over the union of both models' candidate lists. Words above the gate are
scored only by the base model (its list, its ILM); words below it only by
the tail model. delta calibrates the two models' score scales; it and K are
picked on the even-indexed sentences and the result is read on the odd ones.
Costs two encoder passes per swipe; this is the ceiling test, not the
deployable design.
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

BETA = 1.2
ORDER = ["0", "1-5", "6-50", "51-500", "500+"]


def bucket(c: int) -> str:
    return ("0" if c == 0 else "1-5" if c <= 5 else "6-50" if c <= 50
            else "51-500" if c <= 500 else "500+")


def decode_single(b, alpha, lam):
    ilm = b["ilm"]["mean"]
    out = []
    for L in b["lists"]:
        best, bs = None, -np.inf
        for w, ar, uni, ln in L:
            s = ar + BETA * ln + alpha * uni - lam * ilm.get(w, 0.0)
            if s > bs:
                best, bs = w, s
        out.append(best)
    return out


def decode_gated(B, T, cnt, K, delta, alpha, lam):
    ib, it = B["ilm"]["mean"], T["ilm"]["mean"]
    out = []
    for Lb, Lt in zip(B["lists"], T["lists"]):
        best, bs = None, -np.inf
        for w, ar, uni, ln in Lb:
            if cnt[w] >= K:
                s = ar + BETA * ln + alpha * uni - lam * ib.get(w, 0.0)
                if s > bs:
                    best, bs = w, s
        for w, ar, uni, ln in Lt:
            if cnt[w] < K:
                s = ar + delta + BETA * ln + alpha * uni - lam * it.get(w, 0.0)
                if s > bs:
                    best, bs = w, s
        out.append(best)
    return out


def score(hyps, refs, cnt, idx):
    tot, ok = Counter(), Counter()
    for i in idx:
        b = bucket(cnt[refs[i]])
        tot[b] += 1
        ok[b] += hyps[i] == refs[i]
    return 100 * sum(ok.values()) / len(idx), {b: 100 * ok[b] / max(tot[b], 1) for b in ORDER}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="hws", choices=["val", "hws"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--tail", default=None)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--gates", default="1,3,6,11,21,51")
    ap.add_argument("--deltas", default="-3,-2,-1,-0.5,0,0.5,1,2,3")
    args = ap.parse_args()

    cnt = pickle.load(open("runs/futo_train_counts.pkl", "rb"))
    B = pickle.load(open(args.base or f"fused_base_{args.split}.pkl", "rb"))
    T = pickle.load(open(args.tail or f"fused_tailft_{args.split}.pkl", "rb"))
    refs = B["refs"]
    assert refs == T["refs"]
    # sentence-level split: even groups tune, odd groups test
    groups = B["groups"]
    dev = [i for g in groups[0::2] for i in g]
    test = [i for g in groups[1::2] for i in g]
    print(f"split {args.split}: {len(refs)} words, {len(groups)} sentences; "
          f"dev {len(dev)} / test {len(test)} words; alpha={args.alpha} lam={args.lam}")

    hb = decode_single(B, args.alpha, args.lam)
    ht = decode_single(T, args.alpha, args.lam)
    for name, h in [("base", hb), ("tail-ft", ht)]:
        d, _ = score(h, refs, cnt, dev)
        t, per = score(h, refs, cnt, test)
        print(f"{name:>8}: dev {d:.2f}  test {t:.2f} | " + " ".join(f"{per[b]:>6.2f}" for b in ORDER))

    print(f"\ngated: dev top-1 by gate K (rows) and delta (cols)")
    gates = [int(k) for k in args.gates.split(",")]
    deltas = [float(d) for d in args.deltas.split(",")]
    print(f"{'K':>4} | " + " ".join(f"{d:>6.1f}" for d in deltas))
    best = (-1, None, None)
    for K in gates:
        row = []
        for delta in deltas:
            h = decode_gated(B, T, cnt, K, delta, args.alpha, args.lam)
            d, _ = score(h, refs, cnt, dev)
            row.append(d)
            if d > best[0]:
                best = (d, K, delta)
        print(f"{K:>4} | " + " ".join(f"{v:>6.2f}" for v in row))
    _, K, delta = best
    print(f"\ndev-best gate K={K} (tail model scores words with <{K} real gestures), delta={delta}")
    h = decode_gated(B, T, cnt, K, delta, args.alpha, args.lam)
    t, per = score(h, refs, cnt, test)
    tb, pb = score(hb, refs, cnt, test)
    tt, pt = score(ht, refs, cnt, test)
    print(f"\nTEST (odd sentences, n={len(test)})")
    print(f"{'arm':>8} {'top-1':>6} | " + " ".join(f"{b:>7}" for b in ORDER))
    print(f"{'base':>8} {tb:>6.2f} | " + " ".join(f"{pb[b]:>7.2f}" for b in ORDER))
    print(f"{'tail-ft':>8} {tt:>6.2f} | " + " ".join(f"{pt[b]:>7.2f}" for b in ORDER))
    print(f"{'gated':>8} {t:>6.2f} | " + " ".join(f"{per[b]:>7.2f}" for b in ORDER))
    print(f"{'g-b':>8} {t - tb:>+6.2f} | " + " ".join(f"{per[b] - pb[b]:>+7.2f}" for b in ORDER))
    # paired
    wins = sum(1 for i in test if h[i] == refs[i] and hb[i] != refs[i])
    losses = sum(1 for i in test if h[i] != refs[i] and hb[i] == refs[i])
    from math import comb
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"gated vs base on test: {wins} wins / {losses} losses, McNemar p={p:.2g}")


if __name__ == "__main__":
    main()
