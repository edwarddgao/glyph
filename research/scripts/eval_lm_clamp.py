#!/usr/bin/env python3
"""Does clamping the LM bonus fix the surname flips without costing accuracy?

    .venv/bin/python scripts/eval_lm_clamp.py [--lm-device mps]

The fused score adds mu·(logP_LM(w|ctx) − marginal(w)). The marginal is averaged
over eight mid-sentence contexts, where proper nouns are rare, so at a sentence
start a surname's conditional sits far above its marginal and the bonus turns a
correct first pass ("hes", "about") into "hess", "scott" (seen in the practice
records, 2026-09-05). Candidates tried on the shipped stack (AR `ar_mixed_s1`
first pass: ar + 0.6·uni + 1.2·len − 0.25·ilm, beam 32; distilgpt2, mu 0.8,
lookahead 1, top-8) over the two replay sets:

  clamp c      bonus = mu·min(delta, c)          c ∈ {∞, 3, 2, 1, 0.5, 0}
  first-off    no LM term on the first word of a sentence (delta = 0 at t = 0)
  both         first-off + clamp 2

Reports word accuracy overall / everyday / tail / first word on the 542 real
iPhone words and the 1,337 FUTO words, and how often each variant changes a
first-pass-correct word (the failure this is about).
"""
from __future__ import annotations

import argparse, collections, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "iphone"))
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeDataset  # noqa: E402
from swipe_typing.model.ar import FlatTrie, ar_beam  # noqa: E402
from swipe_typing.model.data import SwipeCorpus  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from probe_ilm_fusion import ilm_scores  # noqa: E402
import fused_rescore as fr  # noqa: E402

ALPHA, BETA, LAM, MU, BEAM, M = 0.6, 1.2, 0.25, 0.8, 8, 8


def decode(slots, lm, clamp=None, first_off=False, lag=1):
    """slots: [(ref, [(word, acoustic)])] -> decoded words (lookahead-1 commitment)."""
    states = [((), 0.0)]
    for t, (_ref, cands) in enumerate(slots):
        ctxs = [" ".join(s) for s, _ in states]
        lm.fill([(c, cw) for c in ctxs for cw, _ in cands])
        priors = {cw: lm.prior(cw) for cw, _ in cands}
        exp = {}
        for (words, cum), ctx in zip(states, ctxs):
            for cw, ac in cands:
                delta = lm.cache[(ctx, cw)] - priors[cw]
                if clamp is not None: delta = min(delta, clamp)
                if first_off and t == 0: delta = 0.0
                sc = ac + MU * delta
                wt = words + (cw,)
                if wt not in exp or cum + sc > exp[wt]: exp[wt] = cum + sc
        states = sorted(exp.items(), key=lambda kv: -kv[1])[:BEAM]
        if lag is not None and t - lag >= 0:
            j = t - lag; keep = states[0][0][j]
            states = [s for s in states if s[0][j] == keep] or states[:1]
    return list(states[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device("cpu"); kb = KeyboardLayout.qwerty()
    lex = build_lexicon("train+wf320k", ROOT / "data/canonical", ALPHABET, 1.0); trie = FlatTrie(lex, ALPHABET)
    m, alphabet, mode = load_ar(str(ROOT / "runs/ar_mixed_s1/ar_decoder.pt"), device)
    bench = json.load(open(ROOT.parent / "keyboard/Resources/bench_gestures.json"))["sentences"]
    from swipe_typing.schema import Swipe  # noqa: E402

    # mean memory for the ILM table, as export_ilm.py
    val = SwipeCorpus.load(ROOT / "data/canonical/futo/validation", alphabet, limit=2000)
    with torch.no_grad():
        mean_mem = m.encode(torch.cat([x[None] for x, _ in SwipeDataset(val, kb, augment_cfg=None, resample_mode=mode, key_units=True)])).mean(0, keepdim=True)

    domains = {}
    for source in ("capture", "futo"):
        sents = [s for s in bench if s["source"] == source]
        xs, ys, ts, off, words, aspects, sids, widx = [], [], [], [0], [], [], [], []
        for si, s in enumerate(sents):
            for j, (w, g) in enumerate(zip(s["words"], s["gestures"])):
                xs.extend(g["x"]); ys.extend(g["y"]); ts.extend(g["t"]); off.append(len(xs)); words.append(w); aspects.append(2.44); sids.append(si); widx.append(j)
        corpus = SwipeCorpus(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ts, np.int32), np.asarray(off), words, np.asarray(aspects, np.float32))
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode, key_units=True)
        feats = torch.cat([x[None] for x, _ in ds])
        t0 = time.time()
        with torch.no_grad():
            cands = ar_beam(m, feats, trie, alphabet, beam_width=32)
        allw = sorted({cw for c in cands for cw, *_ in c})
        ilm = ilm_scores(m, [w for w in allw if len(w) <= m.cfg.max_word_len], alphabet, device, mean_mem, batch=1024)
        slots = collections.defaultdict(list)
        for i, c in enumerate(cands):
            ranked = sorted(((cw, ar + ALPHA * u + BETA * n - LAM * ilm.get(cw, 0.0)) for cw, ar, u, n in c), key=lambda t: -t[1])[:M]
            slots[sids[i]].append((words[i], ranked))
        domains[source] = ([slots[k] for k in sorted(slots)], [s["tag"] for s in sents])
        print(f"{source}: {len(cands)} words, first pass {sum(1 for i, c in enumerate(cands) if c and sorted(c, key=lambda t: -(t[1] + ALPHA * t[2] + BETA * t[3] - LAM * ilm.get(t[0], 0.0)))[0][0] == words[i]) / len(cands) * 100:.1f}% ({time.time() - t0:.0f}s)", flush=True)

    lm = fr.LMScorer("distilgpt2", torch.device(a.lm_device))
    if a.lm_device == "cpu": lm.model = lm.model.float()
    variants = [("current", dict()), ("clamp 3", dict(clamp=3.0)), ("clamp 2", dict(clamp=2.0)), ("clamp 1", dict(clamp=1.0)), ("clamp 0.5", dict(clamp=0.5)),
                ("clamp 0", dict(clamp=0.0)), ("first-off", dict(first_off=True)), ("first-off+clamp 2", dict(first_off=True, clamp=2.0))]
    print(f"\n{'variant':<20} {'capture':>8} {'everyday':>9} {'tail':>6} {'1st wd':>7}  | {'futo':>6} {'1st wd':>7}  | fp-correct flipped (cap / futo)")
    for name, kw in variants:
        row = []
        for source in ("capture", "futo"):
            slots_list, tags = domains[source]
            ok = n = ev = evn = tl = tln = f1 = f1n = flipped = 0
            for sl, tag in zip(slots_list, tags):
                out = decode(sl, lm, **kw)
                for j, ((ref, cands), o) in enumerate(zip(sl, out)):
                    n += 1; ok += o == ref
                    if cands and cands[0][0] == ref and o != ref: flipped += 1
                    if j == 0: f1n += 1; f1 += o == ref
                    if source == "capture":
                        if tag == "everyday": evn += 1; ev += o == ref
                        else: tln += 1; tl += o == ref
            row.append((ok / n * 100, ev / max(evn, 1) * 100, tl / max(tln, 1) * 100, f1 / max(f1n, 1) * 100, flipped))
        c, f = row
        print(f"{name:<20} {c[0]:8.1f} {c[1]:9.1f} {c[2]:6.1f} {c[3]:7.1f}  | {f[0]:6.1f} {f[3]:7.1f}  | {c[4]} / {f[4]}", flush=True)


if __name__ == "__main__":
    main()
