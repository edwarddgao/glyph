#!/usr/bin/env python3
"""Run one fused-search config locally (no Modal, no timeout).

    python scripts/run_fused_local.py --bundle fused_bundle_shape.pkl \
        --lm gpt2-xl --mu 0.8 --m 8 --delta

Same decode as modal_fused_search.fused_search. Exists because a single
config is not worth a GPU container: the search makes ~160k small batch-8
LM forwards (one per beam state per swipe), so it is kernel-launch-bound and
a big cloud GPU buys almost nothing over local. Modal pays for parallel
configs, not faster single cells.
"""

from __future__ import annotations

import argparse
import pickle
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BETA = 1.2
ALPHA = 0.4
BEAM = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="fused_bundle.pkl")
    ap.add_argument("--lm", default="gpt2")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--mode", default="uni")
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--delta", action="store_true")
    args = ap.parse_args()

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    lists = bundle["lists"]
    ilm = bundle["ilm"].get(args.mode, {})
    refs = bundle["refs"]
    groups = bundle["groups"]
    n = len(refs)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.lm)
    tok.pad_token = tok.eos_token
    lm = (AutoModelForCausalLM.from_pretrained(args.lm, dtype=torch.float16)
          .to(device).eval())
    bos = tok.bos_token_id or tok.eos_token_id

    cache: dict[tuple[str, str], float] = {}

    @torch.no_grad()
    def lm_scores(ctx: str, words: list[str]) -> list[float]:
        todo = [w for w in words if (ctx, w) not in cache]
        if todo:
            if ctx.strip():
                ids = tok(ctx, return_tensors="pt").input_ids.to(device)
                ids = ids[:, -64:]
            else:
                ids = torch.tensor([[bos]], device=device)
            c_len = ids.shape[1]
            tails = [tok(" " + w, return_tensors="pt").input_ids[0].to(device)
                     for w in todo]
            max_t = max(len(t) for t in tails)
            m = len(todo)
            inp = torch.full((m, c_len + max_t), tok.eos_token_id,
                             dtype=torch.long, device=device)
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
                cache[(ctx, todo[i])] = float(step.gather(1, t[:, None]).sum())
        return [cache[(ctx, w)] for w in words]

    def acoustic(cl):
        out = []
        for w, ar, uni, ln in cl:
            prior = ALPHA * uni if args.lam == 0.0 else -args.lam * ilm.get(w, 0.0)
            out.append((w, ar + BETA * ln + prior))
        return out

    def decode(idx, lag):
        states = [((), 0.0)]
        n_words = len(idx)
        for t, i in enumerate(idx):
            cl = acoustic(lists[i][:args.m])
            if cl:
                expansions: dict[tuple, float] = {}
                for words, cum in states:
                    ctx = " ".join(words)
                    ls = lm_scores(ctx, [w for w, _ in cl])
                    if args.delta:
                        unc = lm_scores("", [w for w, _ in cl])
                        ls = [a - u for a, u in zip(ls, unc)]
                    for (w, ac), l in zip(cl, ls):
                        wt = words + (w,)
                        sc = cum + ac + args.mu * l
                        if wt not in expansions or sc > expansions[wt]:
                            expansions[wt] = sc
                states = sorted(expansions.items(), key=lambda kv: -kv[1])[:BEAM]
            else:
                states = [(words + ("",), cum) for words, cum in states]
            if lag is not None and t - lag >= 0:
                j = t - lag
                w_commit = states[0][0][j]
                states = [s for s in states if s[0][j] == w_commit] or states[:1]
        return list(states[0][0]) + [""] * (n_words - len(states[0][0]))

    t0 = time.time()
    for lag, name in [(0, "streaming"), (1, "lookahead1"), (None, "joint")]:
        hit = 0
        for gi, idx in enumerate(groups):
            out = decode(idx, lag)
            hit += sum(w == refs[i] for w, i in zip(out, idx))
            if name == "streaming" and gi % 400 == 0 and gi:
                rate = gi / (time.time() - t0)
                print(f"  {gi}/{len(groups)} sentences "
                      f"({len(cache):,} cached, "
                      f"eta this pass {(len(groups) - gi) / rate / 60:.0f}m)",
                      flush=True)
        print(f"{name}: {hit / n:.4f}  ({time.time() - t0:.0f}s, "
              f"{len(cache):,} lm scores)", flush=True)

    ceiling = sum(refs[i] in [w for w, *_ in lists[i][:args.m]]
                  for i in range(n)) / n
    print(f"ceiling@{args.m}: {ceiling:.4f}")


if __name__ == "__main__":
    main()
