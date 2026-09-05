#!/usr/bin/env python3
"""HAT-style internal-LM subtraction: fix the stack's double-counted prior.

    python scripts/probe_ilm_fusion.py --nbest data/nbest_ar_mmi/futo_validation.npz

The AR decoder emits P(word | gesture), which carries its internalized corpus
prior; the stack then adds a unigram and a context LM on top, so the prior is
counted two-to-three times and alpha exists to fudge it back out (its optimum
drifted 0.8 -> 0.4 when the AR head replaced CTC). The density-ratio fix
scores candidates as

    s(w) = log P_AR(w | gesture) - lambda * log P_ILM(w) + mu * log P_LM(w | ctx)

where P_ILM is the same decoder run with the gesture memory ablated -- the
model can then condition on nothing but letter history, which is exactly its
internal LM. Two ablations are probed: zeroed memory and dataset-mean memory.

The lambda x mu surface is swept with a cheap proxy (GPT-2 scoring candidates
under the *oracle left* context, one pass, cached per (ctx, word)); the
winner should then be confirmed through eval_deferred_commit on an adjusted
dump (--write-adjusted).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, pick_device  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from finetune_ar_mmi import batch_scores  # noqa: E402
from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus  # noqa: E402


@torch.no_grad()
def ilm_scores(model, words: list[str], alphabet: str, device,
               memory_row: torch.Tensor, batch: int = 512) -> dict[str, float]:
    """log P(word) under a constant (gesture-free) memory."""
    out = {}
    for s in range(0, len(words), batch):
        chunk = words[s:s + batch]
        mem = memory_row.expand(len(chunk), -1, -1)
        sc = batch_scores(model, mem, chunk, alphabet, device)
        for w, v in zip(chunk, sc.cpu().tolist()):
            out[w] = v
    return out


@torch.no_grad()
def oracle_left_lm(swipe_ctx: list[str | None], cand_lists: list[list[str]],
                   device, lm_name: str = "gpt2") -> list[list[float]]:
    """log P(candidate | oracle left context), cached per (ctx, word)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(lm_name)
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        lm_name, dtype=torch.float16).to(device).eval()

    need = sorted({(c, w) for c, cl in zip(swipe_ctx, cand_lists)
                   if c is not None for w in cl})
    print(f"  lm pairs to score: {len(need):,}")
    cache: dict[tuple[str, str], float] = {}
    bos = tok.eos_token  # gpt2 has no BOS; eos as sentence start is standard
    B = 96
    for s in range(0, len(need), B):
        batch = need[s:s + B]
        texts = [bos + ((c + " " + w) if c else " " + w) for c, w in batch]
        ctx_lens = [len(tok(bos + c).input_ids) if c else 1
                    for c, _ in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="right")
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device)
        logits = lm(ids, attention_mask=mask).logits.float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        got = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
        for b, (c, w) in enumerate(batch):
            start = max(ctx_lens[b] - 1, 0)
            n_tok = int(mask[b].sum()) - 1
            cache[(c, w)] = float(got[b, start:n_tok].sum())
        if (s // B) % 100 == 0:
            print(f"  scored {min(s + B, len(need)):,}/{len(need):,}",
                  flush=True)
    return [[cache.get((c, w), 0.0) if c is not None else 0.0 for w in cl]
            for c, cl in zip(swipe_ctx, cand_lists)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_mmi/ar_decoder_ep0.pt")
    ap.add_argument("--nbest", default="data/nbest_ar_mmi/futo_validation.npz")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--dump-alpha", type=float, default=0.4,
                    help="alpha the dump's scores were combined with")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.45, 0.6])
    ap.add_argument("--mus", nargs="+", type=float,
                    default=[0.0, 0.4, 0.8, 1.2])
    ap.add_argument("--write-adjusted", default="",
                    help="write an adjusted npz (scores - dump_alpha*unigram "
                         "- lambda*ilm) for eval_deferred_commit; format: "
                         "mode:lambda:outpath")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, _ = load_ar(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    lexicon = build_lexicon(args.lexicon, Path(args.cache), alphabet, 1.0)

    npz = np.load(args.nbest, allow_pickle=False)
    n = min(args.limit, len(npz["words"]))
    feats = npz["features"]
    cands, valid = npz["candidates"][:n], npz["valid"][:n]
    scores, words = npz["scores"][:n], npz["words"][:n]

    cand_lists = [[str(w) for w, v in zip(cands[i], valid[i]) if v]
                  for i in range(n)]
    uniq = sorted({w for cl in cand_lists for w in cl})
    print(f"{n:,} swipes, {len(uniq):,} unique candidates")

    # Remove the dump's unigram term so the prior enters exactly once below.
    uni = {w: lexicon.logp(w) for w in uniq}

    # Internal-LM estimates under two ablations.
    model.eval()
    T, d = 64, model.cfg.d_model
    zero_mem = torch.zeros(1, T, d, device=device)
    sample = torch.from_numpy(feats[:2048]).to(device)
    mean_mem = model.encode(sample).mean(dim=0, keepdim=True)
    ilm = {
        "zero": ilm_scores(model, uniq, alphabet, device, zero_mem),
        "mean": ilm_scores(model, uniq, alphabet, device, mean_mem),
    }
    for mode, m in ilm.items():
        top = sorted(uniq, key=lambda w: -m[w])[:8]
        print(f"  ILM({mode}) most probable candidates: {top}")

    # Oracle-left contexts from the corpus (rows align with the dump).
    corpus = SwipeCorpus.load(Path(args.cache) / args.split, alphabet,
                              limit=len(npz["words"]))
    ctx: list[str | None] = []
    for i in range(n):
        sent, wi = corpus.sentences[i], int(corpus.word_idx[i])
        toks = sent.split() if sent else []
        ok = bool(sent) and 0 <= wi < len(toks) and toks[wi] == str(words[i])
        ctx.append(" ".join(toks[:wi]) if ok else None)
    print(f"  swipes with usable left context: "
          f"{sum(c is not None for c in ctx):,}/{n:,}")

    lm_sc = oracle_left_lm(ctx, cand_lists, device)

    print("\n  top-1 by (lambda, mu), oracle-left proxy:")
    for mode in ("zero", "mean"):
        print(f"  -- ILM ablation: {mode} --")
        header = "  lam\\mu " + "".join(f"{m:>9.1f}" for m in args.mus)
        print(header)
        for lam in args.lambdas:
            row = f"  {lam:>6.2f}"
            for mu in args.mus:
                correct = 0
                for i, cl in enumerate(cand_lists):
                    if not cl:
                        continue
                    base = scores[i]
                    best_w, best_s = None, -np.inf
                    for k, w in enumerate(cl):
                        s = (float(base[k]) - args.dump_alpha * uni[w]
                             - lam * ilm[mode][w] + mu * lm_sc[i][k])
                        if s > best_s:
                            best_w, best_s = w, s
                    correct += best_w == str(words[i])
                row += f"{correct / n:>9.4f}"
            print(row)

    if args.write_adjusted:
        mode, lam, outpath = args.write_adjusted.split(":")
        lam = float(lam)
        full_n = len(npz["words"])
        adj = npz["scores"].copy()
        cands_all, valid_all = npz["candidates"], npz["valid"]
        # Score any candidates beyond the probe subset too.
        uniq_all = sorted({str(w) for i in range(full_n)
                           for w, v in zip(cands_all[i], valid_all[i]) if v})
        missing = [w for w in uniq_all if w not in ilm[mode]]
        if missing:
            mem = zero_mem if mode == "zero" else mean_mem
            ilm[mode].update(ilm_scores(model, missing, alphabet, device, mem))
        for i in range(full_n):
            for k, (w, v) in enumerate(zip(cands_all[i], valid_all[i])):
                if v:
                    w = str(w)
                    adj[i, k] -= (args.dump_alpha * lexicon.logp(w)
                                  + lam * ilm[mode][w])
        np.savez_compressed(
            outpath, features=npz["features"], candidates=cands_all,
            scores=adj, valid=valid_all, target=npz["target"],
            words=npz["words"], sessions=npz["sessions"],
        )
        print(f"\n  wrote {outpath} (mode={mode}, lambda={lam})")


if __name__ == "__main__":
    main()
