#!/usr/bin/env python3
"""Score the keyboard benchmark (Block C): QuickPath vs Gboard vs Swipe.

Every `bench_*.json` in data/ is one sentence typed with one keyboard: the
prompted sentence, what the keyboard committed, time from first input to
"next", and the number of deletion events. Latest upload per (session, set,
keyboard, sentence) wins, so a redo replaces the earlier attempt.

Words are aligned to the reference with word-level Levenshtein (the same
normalization as the corpora: letters only, lowercase). Accuracy is reported
per keyboard overall and by tag, then every pair of keyboards is compared
*paired* over the words both typed — McNemar's exact test over the discordant
words — because the sentences are shared and the unpaired SE would overstate
the uncertainty. The deliberate OOV probe (`istg`, set 7) is reported apart.

Usage:  .venv/bin/python iphone/benchmark_keyboards.py [--data iphone/data] [--session edg]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

OOV_PROBES = {"istg"}


def norm(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def word_align(ref: list[str], hyp: list[str]) -> list[bool]:
    """Per reference word: did an aligned hypothesis word match it?"""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    correct = [False] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            correct[i - 1] = ref[i - 1] == hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return correct


def load(data: Path, session: str | None, source: str | None = None):
    latest: dict[tuple, dict] = {}
    for f in data.glob("bench_*.json"):
        p = json.loads(f.read_text())
        if session and p.get("session") != session:
            continue
        if source and not p.get("session", "").startswith(f"replay-{source}"):
            continue
        k = (p.get("session", "anon"), p.get("set", 0), p["keyboard"], p["sentence"])
        if k not in latest or p["ts"] > latest[k]["ts"]:
            latest[k] = p
    return latest


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """(a right & b wrong, a wrong & b right, exact two-sided p)."""
    ab = sum(1 for x, y in zip(a, b) if x and not y)
    ba = sum(1 for x, y in zip(a, b) if y and not x)
    p = binomtest(ab, ab + ba, 0.5).pvalue if ab + ba else 1.0
    return ab, ba, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--session", default=None, help="restrict to one user's initials")
    ap.add_argument("--source", default=None, help="replay source filter: capture | futo (matches session prefix replay-<source>)")
    ap.add_argument("--examples", type=int, default=8, help="error examples per keyboard")
    a = ap.parse_args()
    rows = load(Path(a.data), a.session, a.source)
    if not rows:
        print("no bench_*.json found")
        return

    # per (session, set, sentence): keyboard -> per-word correctness
    by_sentence: dict[tuple, dict[str, dict]] = defaultdict(dict)
    keyboards = sorted({k[2] for k in rows})
    for (sess, st, kb, sent), p in rows.items():
        ref = [norm(w) for w in sent.split()]
        hyp = [norm(w) for w in p["typed"].split() if norm(w)]
        by_sentence[(sess, st, sent)][kb] = {
            "correct": word_align(ref, hyp), "ref": ref, "hyp": hyp, "tag": p.get("tag", ""),
            "ms": p.get("ms"), "deletions": p.get("deletions", 0),
        }

    print(f"{len(rows)} sentence uploads, {len(by_sentence)} distinct (session, set, sentence), "
          f"keyboards: {', '.join(keyboards)}\n")

    # ---- per-keyboard accuracy, overall and by tag ----
    print("== word accuracy (top-1 committed, no corrections) ==")
    print(f"{'keyboard':<11} {'overall':>16} {'everyday':>16} {'tail':>16} {'first word':>16} {'later words':>16} {'s/word':>7} {'OOV':>6}")
    for kb in keyboards:
        tot = defaultdict(lambda: [0, 0])
        oov = [0, 0]
        ms, words_t, dels, nsent = 0.0, 0, 0, 0
        for r in by_sentence.values():
            if kb not in r:
                continue
            e = r[kb]
            nsent += 1
            dels += e["deletions"] or 0
            if e["ms"]:
                ms += e["ms"]; words_t += len(e["ref"])
            for pos, (w, ok) in enumerate(zip(e["ref"], e["correct"])):
                if w in OOV_PROBES:
                    oov[0] += ok; oov[1] += 1
                    continue
                tot["overall"][0] += ok; tot["overall"][1] += 1
                tot[e["tag"]][0] += ok; tot[e["tag"]][1] += 1
                tot["first" if pos == 0 else "later"][0] += ok; tot["first" if pos == 0 else "later"][1] += 1
        def fmt(k):
            c, n = tot[k]
            return f"{c}/{n} = {c / n:.1%}" if n else "—"
        spw = f"{ms / 1000 / words_t:.2f}" if words_t else "—"
        print(f"{kb:<11} {fmt('overall'):>16} {fmt('everyday'):>16} {fmt('tail'):>16} {fmt('first'):>16} {fmt('later'):>16} "
              f"{spw:>7} {(f'{oov[0]}/{oov[1]}' if oov[1] else '—'):>6}")

    # ---- paired comparisons over sentences both keyboards typed ----
    print("\n== paired (McNemar exact, over words both keyboards typed; OOV probe excluded) ==")
    for ka, kb_ in combinations(keyboards, 2):
        A, B, tags, firsts = [], [], [], []
        for r in by_sentence.values():
            if ka in r and kb_ in r:
                for pos, (w, x, y) in enumerate(zip(r[ka]["ref"], r[ka]["correct"], r[kb_]["correct"])):
                    if w in OOV_PROBES:
                        continue
                    A.append(x); B.append(y); tags.append(r[ka]["tag"]); firsts.append(pos == 0)
        if not A:
            continue
        for label, mask in [("overall", [True] * len(A)),
                            ("everyday", [t == "everyday" for t in tags]),
                            ("tail", [t == "tail" for t in tags]),
                            ("first wd", firsts)]:
            aa = [x for x, m in zip(A, mask) if m]; bb = [y for y, m in zip(B, mask) if m]
            if not aa:
                continue
            ab, ba, p = mcnemar(aa, bb)
            n = len(aa)
            delta = (sum(aa) - sum(bb)) / n * 100
            print(f"  {ka} vs {kb_} [{label:<8}] n={n:<4} {ka} {sum(aa) / n:.1%}  {kb_} {sum(bb) / n:.1%}  "
                  f"delta {delta:+.1f} pts  discordant {ab} vs {ba}  p={p:.3f}")

    # ---- power note ----
    if keyboards:
        any_pair = next(iter(combinations(keyboards, 2)), None)
        if any_pair:
            ka, kb_ = any_pair
            A = [x for r in by_sentence.values() if ka in r and kb_ in r
                 for w, x in zip(r[ka]["ref"], r[ka]["correct"]) if w not in OOV_PROBES]
            B = [y for r in by_sentence.values() if ka in r and kb_ in r
                 for w, y in zip(r[kb_]["ref"], r[kb_]["correct"]) if w not in OOV_PROBES]
            if A:
                disc = sum(1 for x, y in zip(A, B) if x != y) / len(A)
                n = len(A)
                # detectable delta at ~80% power, alpha 0.05: d = 2.8 * sqrt(r / n)
                print(f"\npower: {n} paired words, discordance {disc:.1%} → smallest detectable difference "
                      f"≈ {2.8 * np.sqrt(max(disc, 1e-6) / n) * 100:.1f} pts (80% power)")

    # ---- error examples ----
    for kb in keyboards:
        errs = []
        for r in by_sentence.values():
            if kb in r:
                e = r[kb]
                bad = [(w, h) for w, ok, h in zip(e["ref"], e["correct"], e["hyp"] + [""] * 20) if not ok]
                if bad:
                    errs.append((" ".join(e["ref"]), " ".join(e["hyp"])))
        if errs:
            print(f"\n== {kb}: {len(errs)} sentences with errors, e.g. ==")
            for ref, hyp in errs[:a.examples]:
                print(f"   {ref}\n → {hyp}")


if __name__ == "__main__":
    main()
