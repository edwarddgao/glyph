#!/usr/bin/env python3
"""Run fused-search configs locally (no Modal, no timeout).

    python scripts/run_fused_local.py --bundle fused_bundle_shape.pkl \
        --lm gpt2-xl --mu 0.8 --m 8 --delta

Same decode as modal_fused_search.fused_search. Exists because a single
config is not worth a GPU container: the search makes one LM forward per beam
state per swipe, so it is kernel-launch-bound and a big cloud GPU buys almost
nothing over local. Modal pays for parallel configs, not faster single cells.

Three amortizations make the LM ladder above gpt2-xl affordable (#63); none
of them changes the search, and `--scorer per-state` still runs the original
path for comparison:

  * every (live state, candidate) pair goes through ONE forward instead of
    one forward per state — up to 8x fewer launches, which is what this
    regime is bound by.
  * only the handful of positions whose logits are actually read get
    projected to vocabulary. On a 151k-vocab model the full (rows, len,
    vocab) log-softmax is the single most expensive thing in the pass.
  * `--mus` runs several LM weights in one process, sharing the (ctx, word)
    score cache. The beams diverge, so reuse is partial, but the second mu
    costs roughly half the first.
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
    ap.add_argument("--dtype", default="float16",
                    help="float16 | bfloat16 | float32. fp16 even for "
                         "bf16-native checkpoints: scoring is a forward pass "
                         "over <70 tokens, and bf16's 8-bit mantissa costs "
                         "0.15 nats of batch-order noise on MPS against "
                         "fp16's 0.02 (fp32 is exactly batch-invariant)")
    ap.add_argument("--check-fp32", action="store_true",
                    help="also score the self-check pairs in fp32 on CPU — "
                         "proves fp16 is not distorting this model's scores")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--mode", default="uni")
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--mus", default=None,
                    help="comma list of LM weights to run in one process, "
                         "sharing the LM score cache (overrides --mu)")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="external unigram weight in the acoustic score; "
                         "dilution-trained first passes want more of it")
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--lags", default="0,1,joint",
                    help="comma list of commitment lags to run "
                         "(0=streaming, 1=lookahead1, joint)")
    ap.add_argument("--rows", type=int, default=64,
                    help="max (state x candidate) rows per LM forward")
    ap.add_argument("--scorer", default="auto",
                    choices=["auto", "flat", "per-state"],
                    help="flat = every (state, candidate) row in one "
                         "forward; per-state is the pre-#63 reference path, "
                         "one forward per context. auto takes flat if it "
                         "agrees with per-state on this model")
    ap.add_argument("--save-hyps", default=None,
                    help="npz path; per-lag hypothesis arrays aligned to "
                         "bundle order, for slice analyses. With --mus the "
                         "mu is appended before the suffix")
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
    dtype = getattr(torch, args.dtype)
    lm = (AutoModelForCausalLM.from_pretrained(args.lm, dtype=dtype)
          .to(device).eval())
    bos = tok.bos_token_id or tok.eos_token_id
    pad = tok.eos_token_id
    print(f"{args.lm}  {sum(p.numel() for p in lm.parameters()) / 1e9:.2f}B "
          f"params  {dtype}", flush=True)

    cache: dict[tuple[str, str], float] = {}

    def ctx_ids(ctx: str) -> torch.Tensor:
        if ctx.strip():
            return tok(ctx, return_tensors="pt").input_ids[0][-64:]
        return torch.tensor([bos])

    @torch.no_grad()
    def score_one(ctx: str, words: list[str]) -> None:
        """Pre-#63 path: one forward, context repeated per candidate."""
        ids = ctx_ids(ctx).to(device)[None, :]
        c_len = ids.shape[1]
        tails = [tok(" " + w, return_tensors="pt").input_ids[0].to(device)
                 for w in words]
        max_t = max(len(t) for t in tails)
        m = len(words)
        inp = torch.full((m, c_len + max_t), pad, dtype=torch.long,
                         device=device)
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
        """One forward over every (state, candidate) row, right-padded.

        Works for any architecture. Every row starts at position 0 with its
        own context, so nothing shifts and the padding sits strictly after
        the tokens whose logits are read — the padding cannot reach them
        through causal attention or through a recurrent scan. Left padding
        with position_ids, the obvious alternative, scores 0.27 nats off on
        Qwen3.5, whose linear-attention layers absorb pad tokens into their
        state; the scorer self-check exists because that is invisible until
        measured.
        """
        cids = [ctx_ids(c) for c in ctxs]
        tails = [tok(" " + w, return_tensors="pt").input_ids[0] for w in words]
        rows = [(s, w) for s in range(len(ctxs)) for w in range(len(words))]
        R = len(rows)
        L = max(len(cids[s]) + len(tails[w]) for s, w in rows)
        T = max(len(t) for t in tails)
        inp = torch.full((R, L), pad, dtype=torch.long)
        mask = torch.zeros((R, L), dtype=torch.long)
        # read[r, j] is the position whose logits predict tail token j;
        # want[r, j] is that token. Rows with short tails repeat their last
        # read position and are zeroed out by `live`.
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
        read, want, live = read.to(device), want.to(device), live.to(device)

        # Only T positions per row are ever read, so project only those: the
        # full (R, L, vocab) logits are the expensive part of a forward at
        # this shape — 1.5 GB of fp32 traffic per call on a 151k vocab.
        h = lm.base_model(input_ids=inp, attention_mask=mask).last_hidden_state
        h = h.gather(1, read[:, :, None].expand(-1, -1, h.shape[-1]))
        lp = F.log_softmax(lm.get_output_embeddings()(h).float(), dim=-1)
        got = (lp.gather(2, want[:, :, None])[:, :, 0] * live).sum(1)
        for r, (s, w) in enumerate(rows):
            cache[(ctxs[s], words[w])] = float(got[r])

    PATHS = {"flat": score_flat, "per-state": score_one}

    def lm_fill(ctxs: list[str], words: list[str]) -> None:
        """Score every missing (ctx, word) pair among these contexts."""
        # dict.fromkeys: the empty context is both a state and delta's
        # unconditional row on the first swipe of every sentence.
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
            scorer(todo[i:i + chunk], words)

    CHECK_CTXS = ["", "is that", "im on a",
                  "the extensive crack networks that"]
    CHECK_WORDS = ["ok", "the", "plane", "form", "i", "networks"]

    def check(path) -> float:
        """Worst disagreement with per-state scoring, in nats.

        Batching and padding are load-bearing, and how a given architecture
        treats them is not knowable from the config — prove it per model.
        Anything much above fp16/bf16 batch noise means the fast path is
        not scoring the same thing.
        """
        ctxs, words = CHECK_CTXS, CHECK_WORDS
        cache.clear()
        for c in ctxs:
            score_one(c, words)
        ref = dict(cache)
        cache.clear()
        if path is not score_one:
            path(ctxs, words)
        worst = max(abs(ref[k] - cache.get(k, ref[k])) for k in ref)
        cache.clear()
        return worst

    order = ([args.scorer] if args.scorer != "auto"
             else ["flat", "per-state"])
    scorer = None
    for name in order:
        try:
            worst = check(PATHS[name])
        except (AttributeError, NotImplementedError, TypeError) as e:
            print(f"  scorer {name}: unavailable ({type(e).__name__}: {e})",
                  flush=True)
            continue
        # fp16 batch-order noise measures 0.02-0.06 nats; the bugs this is
        # here to catch (bf16 mantissa, left padding through a recurrent
        # state) measure 0.27 and up.
        ok = worst <= 0.10 or args.scorer != "auto"
        print(f"  scorer {name}: max |delta| vs per-state = {worst:.4f} nats"
              f"{'' if ok else '  — REJECTED'}", flush=True)
        if ok:
            scorer = PATHS[name]
            break
    if scorer is None:
        raise SystemExit("no scoring path agrees with per-state scoring")

    if args.check_fp32:
        lm_fill(CHECK_CTXS, CHECK_WORDS)
        fast = dict(cache)
        cache.clear()
        ref32 = (AutoModelForCausalLM.from_pretrained(args.lm,
                                                      dtype=torch.float32)
                 .to("cpu").eval())
        lm, device, keep = ref32, torch.device("cpu"), lm
        for c in CHECK_CTXS:
            score_one(c, CHECK_WORDS)
        worst = max(abs(fast[k] - cache[k]) for k in fast)
        print(f"  {args.dtype} vs fp32: max |delta| = {worst:.4f} nats "
              f"over {len(fast)} pairs", flush=True)
        del ref32
        lm, device = keep, torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        cache.clear()

    def acoustic(cl):
        out = []
        for w, ar, uni, ln in cl:
            prior = (args.alpha * uni if args.lam == 0.0
                     else -args.lam * ilm.get(w, 0.0))
            out.append((w, ar + BETA * ln + prior))
        return out

    def decode(idx, lag, mu):
        states = [((), 0.0)]
        n_words = len(idx)
        for t, i in enumerate(idx):
            cl = acoustic(lists[i][:args.m])
            if cl:
                cands = [w for w, _ in cl]
                ctxs = [" ".join(words) for words, _ in states]
                lm_fill(ctxs + ([""] if args.delta else []), cands)
                unc = ([cache[("", w)] for w in cands] if args.delta
                       else [0.0] * len(cands))
                expansions: dict[tuple, float] = {}
                for (words, cum), ctx in zip(states, ctxs):
                    for k, (w, ac) in enumerate(cl):
                        wt = words + (w,)
                        sc = cum + ac + mu * (cache[(ctx, w)] - unc[k])
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

    all_lags = {"0": (0, "streaming"), "1": (1, "lookahead1"),
                "joint": (None, "joint")}
    mus = ([float(x) for x in args.mus.split(",")] if args.mus else [args.mu])
    t0 = time.time()
    first_pass = True
    for mu in mus:
        hyps_out = {}
        for key in args.lags.split(","):
            lag, name = all_lags[key.strip()]
            hit = 0
            hyps = [""] * n
            for gi, idx in enumerate(groups):
                out = decode(idx, lag, mu)
                for w, i in zip(out, idx):
                    hyps[i] = w
                    hit += w == refs[i]
                if first_pass and gi % 400 == 0 and gi:
                    rate = gi / (time.time() - t0)
                    print(f"  {gi}/{len(groups)} sentences "
                          f"({len(cache):,} cached, "
                          f"eta this pass {(len(groups) - gi) / rate / 60:.0f}m)",
                          flush=True)
            first_pass = False
            hyps_out[name] = hyps
            print(f"mu={mu} {name}: {hit / n:.4f}  ({time.time() - t0:.0f}s, "
                  f"{len(cache):,} lm scores)", flush=True)

        if args.save_hyps:
            import numpy as np

            path = args.save_hyps
            if len(mus) > 1:
                stem, _, suf = path.rpartition(".")
                path = f"{stem}_mu{mu}.{suf}"
            np.savez_compressed(
                path, refs=np.array(refs, dtype=object),
                **{k: np.array(v, dtype=object) for k, v in hyps_out.items()})
            print(f"wrote {path}", flush=True)

    ceiling = sum(refs[i] in [w for w, *_ in lists[i][:args.m]]
                  for i in range(n)) / n
    print(f"ceiling@{args.m}: {ceiling:.4f}")


if __name__ == "__main__":
    main()
