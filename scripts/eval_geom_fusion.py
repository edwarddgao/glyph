#!/usr/bin/env python3
"""Fused sentence-beam with the classical geometry channel back in (#73).

The decode is run_fused_local.py's (parity-verified word-for-word against
runs/hyps_base_hws.npz at gamma=0) with one change: every candidate's
acoustic score gains a training-free GestureDP alignment term,

    acoustic(w) = ar + beta*len + alpha*uni - gamma*geom_cost(w)

Why this pays: within a swipe's candidate list the AR score and the
alignment cost are nearly independent (corr -0.17 on hws), so the analytic
channel carries evidence the trained one lacks — #10/#11's rule applied to
a channel the neural stack had discarded. On hws it is worth +2.1 eval
(80.90 -> 83.04 at gamma=0.5, McNemar p=3e-33), concentrated in unseen
(+9.4) and rare (+7.1) words with the head flat-positive; optional
proposals from gen_geom_proposals.py add a further +0.16 (p=1e-4).

    python scripts/eval_geom_fusion.py --bundle fused_base_hws.pkl \
        --data data/canonical/how_we_swipe/test --gamma 0.5 \
        --proposals geom_props_hws.pkl --baseline runs/hyps_base_hws.npz
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from gen_geom_proposals import calibrate_offset  # noqa: E402
from swipe_typing.geomllm import GeomConfig, GestureDP  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus  # noqa: E402

BETA = 1.2
BEAM = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="fused_base_hws.pkl")
    ap.add_argument("--data", default="data/canonical/how_we_swipe/test")
    ap.add_argument("--lm", default="gpt2")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="geometry-channel weight; 0 reproduces the "
                         "gamma-free fused decode exactly")
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--proposals", default=None,
                    help="pkl from gen_geom_proposals.py; adds out-of-list "
                         "candidates carrying both channel scores")
    ap.add_argument("--p", type=int, default=8,
                    help="proposals appended per swipe")
    ap.add_argument("--calibrate", type=int, default=2000)
    ap.add_argument("--limit-groups", type=int, default=0)
    ap.add_argument("--dev-groups", type=int, default=1000,
                    help="leading sentence groups reported as the tuning "
                         "slice, the rest as eval")
    ap.add_argument("--baseline", default=None,
                    help="hyps npz to diff against (fixed/broken, McNemar)")
    ap.add_argument("--buckets", action="store_true",
                    help="report accuracy by futo-train word count")
    ap.add_argument("--save-hyps", default=None)
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    lists, refs, groups = bundle["lists"], bundle["refs"], bundle["groups"]
    n = len(refs)
    corpus = SwipeCorpus.load(args.data, ALPHABET, limit=n)
    assert list(corpus.words) == list(refs), "bundle/corpus order mismatch"
    kb = KeyboardLayout.qwerty()
    offset = calibrate_offset(corpus, kb, args.calibrate)
    gcfg = GeomConfig(offset=offset)

    print("scoring list candidates under the geometry channel...")
    t0 = time.time()
    aug: list[list[tuple[str, float, float, int, float]]] = []
    for i in range(n):
        dp = GestureDP(corpus.points(i), corpus.times(i), kb, gcfg)
        aug.append([(w, ar, uni, ln, float(dp.word_cost(w)))
                    for w, ar, uni, ln in lists[i][:args.m]])
    print(f"  {n} swipes in {time.time() - t0:.0f}s")
    if args.proposals:
        with open(args.proposals, "rb") as f:
            props = pickle.load(f)
        for i in range(n):
            # truncation ranked by the AR-acoustic score, matching the
            # measured +0.16 cell (the fused search re-ranks with gamma)
            ranked = sorted(props[i], key=lambda c: -(c[1] + BETA * c[3]
                                                      + args.alpha * c[2]))
            aug[i] += ranked[:args.p]

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    tok = AutoTokenizer.from_pretrained(args.lm)
    tok.pad_token = tok.eos_token
    lm = (AutoModelForCausalLM.from_pretrained(args.lm, dtype=torch.float16)
          .to(device).eval())
    bos = tok.bos_token_id or tok.eos_token_id
    pad = tok.eos_token_id
    cache: dict[tuple[str, str], float] = {}

    def ctx_ids(ctx: str) -> torch.Tensor:
        if ctx.strip():
            return tok(ctx, return_tensors="pt").input_ids[0][-64:]
        return torch.tensor([bos])

    @torch.no_grad()
    def score_one(ctx: str, words: list[str]) -> None:
        ids = ctx_ids(ctx).to(device)[None, :]
        c_len = ids.shape[1]
        tails = [tok(" " + w, return_tensors="pt").input_ids[0].to(device)
                 for w in words]
        inp = torch.full((len(words), c_len + max(len(t) for t in tails)),
                         pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(inp)
        inp[:, :c_len] = ids
        mask[:, :c_len] = 1
        for i, t in enumerate(tails):
            inp[i, c_len:c_len + len(t)] = t
            mask[i, c_len:c_len + len(t)] = 1
        logits = lm(input_ids=inp, attention_mask=mask).logits
        lp = F.log_softmax(logits.float(), dim=-1)
        for i, t in enumerate(tails):
            step = lp[i, c_len - 1:c_len - 1 + len(t)]
            cache[(ctx, words[i])] = float(step.gather(1, t[:, None]).sum())

    @torch.no_grad()
    def score_flat(ctxs: list[str], words: list[str]) -> None:
        cids = [ctx_ids(c) for c in ctxs]
        tails = [tok(" " + w, return_tensors="pt").input_ids[0]
                 for w in words]
        rows = [(s, w) for s in range(len(ctxs)) for w in range(len(words))]
        R = len(rows)
        L = max(len(cids[s]) + len(tails[w]) for s, w in rows)
        T = max(len(t) for t in tails)
        inp = torch.full((R, L), pad, dtype=torch.long)
        mask = torch.zeros((R, L), dtype=torch.long)
        read = torch.zeros((R, T), dtype=torch.long)
        want = torch.zeros((R, T), dtype=torch.long)
        live = torch.zeros((R, T), dtype=torch.bool)
        for r, (s, w) in enumerate(rows):
            c, t = cids[s], tails[w]
            inp[r, :len(c) + len(t)] = torch.cat([c, t])
            mask[r, :len(c) + len(t)] = 1
            read[r] = len(c) - 1 + torch.arange(T).clamp(max=len(t) - 1)
            want[r, :len(t)] = t
            live[r, :len(t)] = True
        inp, mask = inp.to(device), mask.to(device)
        read, want = read.to(device), want.to(device)
        live = live.to(device)
        h = lm.base_model(input_ids=inp,
                          attention_mask=mask).last_hidden_state
        h = h.gather(1, read[:, :, None].expand(-1, -1, h.shape[-1]))
        lp = F.log_softmax(lm.get_output_embeddings()(h).float(), dim=-1)
        got = (lp.gather(2, want[:, :, None])[:, :, 0] * live).sum(1)
        for r, (s, w) in enumerate(rows):
            cache[(ctxs[s], words[w])] = float(got[r])

    check_ctxs = ["", "is that", "im on a"]
    check_words = ["ok", "the", "plane", "networks"]
    for c in check_ctxs:
        score_one(c, check_words)
    ref_scores = dict(cache)
    cache.clear()
    score_flat(check_ctxs, check_words)
    worst = max(abs(ref_scores[k] - cache[k]) for k in ref_scores)
    print(f"scorer flat: max |delta| vs per-state = {worst:.4f} nats")
    scorer = score_flat if worst <= 0.10 else score_one
    cache.clear()

    def lm_fill(ctxs: list[str], words: list[str]) -> None:
        todo = [c for c in dict.fromkeys(ctxs)
                if any((c, w) not in cache for w in words)]
        if not todo:
            return
        if scorer is score_one:
            for c in todo:
                score_one(c, [w for w in words if (c, w) not in cache])
            return
        chunk = max(1, args.rows // len(words))
        for i in range(0, len(todo), chunk):
            score_flat(todo[i:i + chunk], words)

    def decode(idx: list[int]) -> list[str]:
        states = [((), 0.0)]
        n_words = len(idx)
        for t, i in enumerate(idx):
            cl = [(w, ar + BETA * ln + args.alpha * uni - args.gamma * geom)
                  for w, ar, uni, ln, geom in aug[i]]
            if cl:
                cands = [w for w, _ in cl]
                ctxs = [" ".join(words) for words, _ in states]
                lm_fill(ctxs + [""], cands)
                unc = [cache[("", w)] for w in cands]
                expansions: dict[tuple, float] = {}
                for (words, cum), ctx in zip(states, ctxs):
                    for k, (w, ac) in enumerate(cl):
                        wt = words + (w,)
                        sc = cum + ac + args.mu * (cache[(ctx, w)] - unc[k])
                        if wt not in expansions or sc > expansions[wt]:
                            expansions[wt] = sc
                states = sorted(expansions.items(),
                                key=lambda kv: -kv[1])[:BEAM]
            else:
                states = [(words + ("",), cum) for words, cum in states]
        return list(states[0][0]) + [""] * (n_words - len(states[0][0]))

    run_groups = groups[:args.limit_groups] if args.limit_groups else groups
    dev = np.zeros(n, dtype=bool)
    for g in groups[:args.dev_groups]:
        for i in g:
            dev[i] = True

    hyps = [""] * n
    t0 = time.time()
    for gi, idx in enumerate(run_groups):
        out = decode(idx)
        for w, i in zip(out, idx):
            hyps[i] = w
        if gi % 500 == 0 and gi:
            rate = gi / (time.time() - t0)
            print(f"  {gi}/{len(run_groups)} sentences "
                  f"(eta {(len(run_groups) - gi) / rate / 60:.0f}m)",
                  flush=True)

    done = [i for g in run_groups for i in g]
    for name, keep in [("all", lambda i: True), ("dev", lambda i: dev[i]),
                       ("eval", lambda i: not dev[i])]:
        idx = [i for i in done if keep(i)]
        if idx:
            acc = np.mean([hyps[i] == refs[i] for i in idx])
            print(f"[{name}] n={len(idx)}  top-1 {acc:.4f}")

    if args.baseline:
        base = np.load(args.baseline, allow_pickle=True)["joint"]
        for name, keep in [("dev", lambda i: dev[i]),
                           ("eval", lambda i: not dev[i])]:
            idx = [i for i in done if keep(i)]
            fixed = sum(base[i] != refs[i] == hyps[i] for i in idx)
            broken = sum(hyps[i] != refs[i] == base[i] for i in idx)
            m = fixed + broken
            # two-sided exact binomial (McNemar) without a scipy dependency
            p = (sum(math.comb(m, k) for k in
                     range(min(fixed, broken) + 1)) +
                 sum(math.comb(m, k) for k in
                     range(max(fixed, broken), m + 1))) / 2 ** m if m else 1.0
            b_acc = np.mean([base[i] == refs[i] for i in idx])
            h_acc = np.mean([hyps[i] == refs[i] for i in idx])
            print(f"[{name}] {b_acc:.4f} -> {h_acc:.4f} "
                  f"({(h_acc - b_acc) * 100:+.2f})  fixed {fixed} / "
                  f"broken {broken}  McNemar p={min(1.0, p):.2g}")

    if args.buckets:
        from collections import Counter
        counts = Counter(SwipeCorpus.load("data/canonical/futo/train",
                                          ALPHABET).words)
        base = (np.load(args.baseline, allow_pickle=True)["joint"]
                if args.baseline else None)
        edges = [("unseen", 0, 0), ("1-5", 1, 5), ("6-50", 6, 50),
                 (">50", 51, 10 ** 9)]
        ev = [i for i in done if not dev[i]]
        for name, lo, hi in edges:
            sel = [i for i in ev if lo <= counts.get(refs[i], 0) <= hi]
            if not sel:
                continue
            h = np.mean([hyps[i] == refs[i] for i in sel])
            line = f"  {name:>6}: n={len(sel):5d}  {h:.4f}"
            if base is not None:
                b = np.mean([base[i] == refs[i] for i in sel])
                line += f"  (baseline {b:.4f}, {(h - b) * 100:+.2f})"
            print(line)

    if args.save_hyps:
        np.savez_compressed(args.save_hyps,
                            refs=np.array(refs, dtype=object),
                            joint=np.array(hyps, dtype=object))
        print(f"wrote {args.save_hyps}")


if __name__ == "__main__":
    main()
