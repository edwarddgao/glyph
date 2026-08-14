#!/usr/bin/env python3
"""Does the LM that ranks best in-search also *model* the eval text best?

    python scripts/probe_lm_fit.py --bundle fused_bundle_val38.pkl \
        --lms gpt2,gpt2-xl,Qwen/Qwen3.5-2B-Base,Qwen/Qwen3.5-9B-Base

#66 found 2019's gpt2-xl at the top of an in-search LM ladder that runs to 9B
of modern pretraining, which is surprising enough to deserve a direct check
rather than an explanation. Perplexity over the reference sentences — the
lowercase, unpunctuated Common Voice prompts the corpus actually contains —
separates the two readings: if the modern model fits this text better and
still ranks worse, the deficit is about ranking rather than language
modelling; if it fits worse, the ladder is measuring distribution match, as
#34 argued from the other side.

Reports per-token and per-word NLL. Per-word is the comparable one across
tokenizers (50k vs 151k vocabularies segment the same words differently);
per-token is shown only to make that difference visible.

--check-fp32 re-scores a sample in fp32 on CPU, which prices the other
candidate artifact: the ladder runs bf16-native checkpoints in fp16.
"""

from __future__ import annotations

import argparse
import pickle

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="fused_bundle_val38.pkl")
    ap.add_argument("--lms", default="gpt2,gpt2-xl")
    ap.add_argument("--limit", type=int, default=400,
                    help="sentences to score")
    ap.add_argument("--check-fp32", action="store_true")
    ap.add_argument("--format", default="raw", choices=["raw", "cased"],
                    help="raw = the corpus's own lowercase unpunctuated form; "
                         "cased = sentence-cased with a period, to price how "
                         "much of a model's misfit is surface form")
    args = ap.parse_args()

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    refs, groups = bundle["refs"], bundle["groups"]
    sents = [" ".join(refs[i] for i in g if refs[i])
             for g in groups[:args.limit]]
    sents = [s for s in sents if s.strip()]
    if args.format == "cased":
        sents = [s[0].upper() + s[1:] + "." for s in sents]
    n_words = sum(len(s.split()) for s in sents)
    print(f"{len(sents)} sentences, {n_words} words\n")
    print(f"{'LM':<26}{'nll/word':>10}{'ppl/word':>10}"
          f"{'nll/token':>11}{'tok/word':>10}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for name in args.lms.split(","):
        tok = AutoTokenizer.from_pretrained(name)
        lm = (AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
              .to(device).eval())
        bos = tok.bos_token_id or tok.eos_token_id

        tot_nll, tot_tok = 0.0, 0
        with torch.no_grad():
            for s in sents:
                ids = tok(s, return_tensors="pt").input_ids[0].to(device)
                # score every token including the first, conditioned on bos,
                # so the comparison does not hand either model a free word
                inp = torch.cat([torch.tensor([bos], device=device), ids])[None]
                lp = F.log_softmax(lm(input_ids=inp).logits.float(), dim=-1)
                tot_nll -= float(
                    lp[0, :-1].gather(1, ids[:, None]).sum())
                tot_tok += len(ids)
        nll_w = tot_nll / n_words
        print(f"{name:<26}{nll_w:>10.3f}{torch.tensor(nll_w).exp():>10.1f}"
              f"{tot_nll / tot_tok:>11.3f}{tot_tok / n_words:>10.2f}")

        if args.check_fp32:
            probe = sents[:20]
            ref16 = []
            with torch.no_grad():
                for s in probe:
                    ids = tok(s, return_tensors="pt").input_ids[0].to(device)
                    inp = torch.cat([torch.tensor([bos], device=device),
                                     ids])[None]
                    lp = F.log_softmax(lm(input_ids=inp).logits.float(), -1)
                    ref16.append(float(lp[0, :-1].gather(1, ids[:, None]).sum()))
            lm32 = (AutoModelForCausalLM.from_pretrained(name,
                                                         dtype=torch.float32)
                    .to("cpu").eval())
            worst = 0.0
            with torch.no_grad():
                for s, r in zip(probe, ref16):
                    ids = tok(s, return_tensors="pt").input_ids[0]
                    inp = torch.cat([torch.tensor([bos]), ids])[None]
                    lp = F.log_softmax(lm32(input_ids=inp).logits.float(), -1)
                    v = float(lp[0, :-1].gather(1, ids[:, None]).sum())
                    worst = max(worst, abs(v - r) / max(1, len(ids)))
            print(f"{'':<26}fp16 vs fp32: {worst:.4f} nats/token worst case")
            del lm32
        del lm


if __name__ == "__main__":
    main()
