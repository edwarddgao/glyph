#!/usr/bin/env python3
"""Paired comparison between two fused-search runs' saved hypotheses.

    python scripts/compare_hyps.py runs/hyps_g2.npz runs/hyps_xl.npz --lag joint

Every arm of an LM ladder decodes the *same* n-best lists for the same words,
so the arms agree on the overwhelming majority of them and the unpaired
standard error (0.16 pts at n=20k) badly overstates the bar a difference has
to clear. What matters is the discordant pairs: words one arm gets right and
the other gets wrong. McNemar's exact test reads exactly those.
"""

from __future__ import annotations

import argparse
from math import comb

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for b wins vs c losses among discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--lag", default="joint")
    ap.add_argument("--examples", type=int, default=0,
                    help="print this many words the second arm flips to right")
    args = ap.parse_args()

    A, B = np.load(args.a, allow_pickle=True), np.load(args.b, allow_pickle=True)
    refs = A["refs"]
    assert (refs == B["refs"]).all(), "different bundles"
    ok_a, ok_b = A[args.lag] == refs, B[args.lag] == refs
    n = len(refs)

    win = int((~ok_a & ok_b).sum())      # b fixes what a missed
    loss = int((ok_a & ~ok_b).sum())
    p = mcnemar_exact(win, loss)
    print(f"n={n}  {args.lag}")
    print(f"  {args.a}: {ok_a.mean():.4f}")
    print(f"  {args.b}: {ok_b.mean():.4f}")
    print(f"  delta {(ok_b.mean() - ok_a.mean()) * 100:+.2f} pts  "
          f"({win} fixed, {loss} broken, {win + loss} discordant of {n})")
    print(f"  McNemar exact p = {p:.2g}")
    # The paired SE of the difference is what the delta should be read against.
    se = np.sqrt(win + loss) / n
    print(f"  paired 1 SE = {se * 100:.2f} pts  "
          f"({abs(ok_b.mean() - ok_a.mean()) / se:.1f} SE)")

    if args.examples:
        flips = np.where(~ok_a & ok_b)[0][:args.examples]
        for i in flips:
            print(f"    {A[args.lag][i]!r} -> {B[args.lag][i]!r}  "
                  f"(truth {refs[i]!r})")


if __name__ == "__main__":
    main()
