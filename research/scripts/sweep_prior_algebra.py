#!/usr/bin/env python3
"""Can an external prior cancel the tail fine-tune's head tax? First-pass sweep.

    python scripts/sweep_prior_algebra.py --split val
    python scripts/sweep_prior_algebra.py --split hws

#60 fine-tuned the AR encoder on synthetic gestures for 269k rare lexicon
words and bought +7 to +10 on unseen words at a 1-4 point cost on every
frequent-word bucket; #61 found the cost survives the fused stack and that
raising alpha (the external unigram weight) refunds about half of it. #76
then showed the frequent-word confusions are not separable from the gesture
even by dedicated classifiers, so the tax cannot be lost acoustic
discrimination on the head -- it can only be the encoder's implicit frequency
prior having been flattened. If that is the whole story, the prior can be
put back from outside: a stronger external unigram (alpha), or subtracting
the encoder's own internal-LM estimate (lambda x ilm, the #49 prior algebra)
and replacing it with the corpus unigram.

This sweeps the acoustic score

    ar + BETA*len + alpha*uni - lam*ilm[w]

over the saved deep lists (M=64 candidates) of the base and tail-ft arms,
reporting first-pass top-1 overall and by the truth's FUTO-train gesture
count, so the head/tail trade is visible per cell. It is the cheap proxy: the
fused stack adds the LM on top, but #61 showed the bucket structure passes
through fusion essentially unchanged.
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


def train_counts(cache: Path, alphabet: str) -> Counter:
    cp = Path("runs/futo_train_counts.pkl")
    if cp.exists():
        return pickle.load(open(cp, "rb"))
    from swipe_typing.model import SwipeCorpus
    c = Counter(SwipeCorpus.load(cache / "futo/train", alphabet).words)
    pickle.dump(c, open(cp, "wb"))
    return c


def decode(bundle, alpha, lam, ilm_mode, beta=BETA):
    ilm = bundle["ilm"].get(ilm_mode, {}) if lam else {}
    hyps = []
    for L in bundle["lists"]:
        best, bs = None, -np.inf
        for w, ar, uni, ln in L:
            s = ar + beta * ln + alpha * uni - (lam * ilm.get(w, 0.0) if lam else 0.0)
            if s > bs:
                best, bs = w, s
        hyps.append(best)
    return hyps


def by_bucket(hyps, refs, cnt):
    tot, ok = Counter(), Counter()
    for h, r in zip(hyps, refs):
        b = bucket(cnt[r])
        tot[b] += 1
        ok[b] += h == r
    return {b: 100 * ok[b] / max(tot[b], 1) for b in ORDER}, 100 * sum(ok.values()) / len(refs), tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "hws"])
    ap.add_argument("--arms", default=None,
                    help="comma list name=bundle.pkl; default base/tailft "
                         "bundles for --split. First arm is the reference.")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--alphas", default="0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0")
    ap.add_argument("--lams", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--ilm", default="mean", choices=["zero", "mean"])
    args = ap.parse_args()

    from swipe_typing.layout import KeyboardLayout
    kb = KeyboardLayout.qwerty()
    cnt = train_counts(Path(args.cache), kb.letters)
    if args.arms:
        arms = {kv.split("=")[0]: pickle.load(open(kv.split("=")[1], "rb"))
                for kv in args.arms.split(",")}
    else:
        arms = {"base": pickle.load(open(f"fused_base_{args.split}.pkl", "rb")),
                "tailft": pickle.load(open(f"fused_tailft_{args.split}.pkl", "rb"))}
    ref_arm = next(iter(arms))
    refs = arms[ref_arm]["refs"]
    assert all(b["refs"] == refs for b in arms.values())
    alphas = [float(a) for a in args.alphas.split(",")]
    lams = [float(l) for l in args.lams.split(",")]

    _, _, tot = by_bucket(refs, refs, cnt)
    print(f"split {args.split}: n={len(refs)}  bucket shares: " +
          ", ".join(f"{b} {100 * tot[b] / len(refs):.1f}%" for b in ORDER))
    print(f"acoustic = ar + {BETA}*len + alpha*uni - lam*ilm[{args.ilm}]\n")
    hdr = f"{'arm':>6} {'alpha':>5} {'lam':>4} | {'top-1':>6} | " + " ".join(f"{b:>7}" for b in ORDER)
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for lam in lams:
        for alpha in alphas:
            for arm, b in arms.items():
                per, top, _ = by_bucket(decode(b, alpha, lam, args.ilm), refs, cnt)
                rows[(arm, alpha, lam)] = (top, per)
                print(f"{arm:>6} {alpha:>5.2f} {lam:>4.2f} | {top:>6.2f} | " +
                      " ".join(f"{per[k]:>7.2f}" for k in ORDER))
        print()

    # the question: which tail-ft cell keeps the tail gain and refunds the head?
    best = {a: max(((k, v) for k, v in rows.items() if k[0] == a), key=lambda kv: kv[1][0])
            for a in arms}
    for a, (k, v) in best.items():
        print(f"{a:>10} best: alpha={k[1]} lam={k[2]} top-1 {v[0]:.2f} | " +
              " ".join(f"{v[1][b]:>7.2f}" for b in ORDER))
    bp, btop = best[ref_arm][1][1], best[ref_arm][1][0]
    for a in arms:
        if a == ref_arm:
            continue
        print(f"\n{a} minus {ref_arm}(best), per bucket, at each {a} cell (positive = {a} ahead):")
        print(f"{'alpha':>5} {'lam':>4} | {'top-1':>6} | " + " ".join(f"{b:>7}" for b in ORDER))
        for (arm, alpha, lam), (top, per) in sorted(rows.items()):
            if arm != a:
                continue
            print(f"{alpha:>5.2f} {lam:>4.2f} | {top - btop:>+6.2f} | " +
                  " ".join(f"{per[k] - bp[k]:>+7.2f}" for k in ORDER))


if __name__ == "__main__":
    main()
