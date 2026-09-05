#!/usr/bin/env python3
"""Training-free rung 3: joint LLM-geometry beam, no lexicon, no encoder.

The LLM's next-token distribution is the candidate space (its vocabulary
replaces the trie) and an analytic alignment cost is the acoustic channel
(``geomllm.GestureDP`` replaces the CTC emissions). The corpus is used for
evaluation only; ``--lm-weight`` is the single knob.

    python scripts/eval_llm_beam.py --limit 200 --lm Qwen/Qwen3.5-2B-Base
    python scripts/eval_llm_beam.py --context oracle --lm-weight 0.6
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from swipe_typing.geomllm import (  # noqa: E402
    BeamConfig,
    GeomConfig,
    GestureDP,
    TokenLetterTable,
    context_cache_ok,
    decode_word,
)
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def left_context(corpus: SwipeCorpus, i: int, max_words: int = 8) -> str:
    sent, idx = corpus.sentences[i], int(corpus.word_idx[i])
    if not sent or idx <= 0:
        return ""
    return " ".join(sent.split()[:idx][-max_words:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/canonical/futo/validation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many swipes first (e.g. 50 to stay "
                         "disjoint from the width-tuning slice)")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the context KV cache even if the model "
                         "passes the parity probe")
    ap.add_argument("--lm", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--context", default="none", choices=["none", "oracle"])
    ap.add_argument("--prime", default="",
                    help="neutral text prepended before the (possibly empty) "
                         "context. Qwen3.5 has no real BOS, so its "
                         "unconditioned next-word distribution is flat "
                         "multilingual noise (#66); a short English prime "
                         "restores a usable prior for the cold start.")
    ap.add_argument("--lm-weight", type=float, default=0.5)
    ap.add_argument("--rescore-weight", type=float, default=1.0)
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--topk-lm", type=int, default=32)
    ap.add_argument("--topk-geom", type=int, default=24)
    ap.add_argument("--gate-letters", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=14)
    ap.add_argument("--n-points", type=int, default=96)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    device = pick_device(args.device)
    kb = KeyboardLayout.qwerty()
    corpus = SwipeCorpus.load(args.data, ALPHABET,
                              limit=args.offset + args.limit)

    tok = AutoTokenizer.from_pretrained(args.lm)
    lm = (AutoModelForCausalLM.from_pretrained(args.lm, dtype=torch.float16)
          .to(device).eval())
    table = TokenLetterTable(tok)
    bos = tok.bos_token_id or tok.eos_token_id

    bcfg = BeamConfig(beam=args.beam, topk_lm=args.topk_lm,
                      topk_geom=args.topk_geom, lm_weight=args.lm_weight,
                      rescore_weight=args.rescore_weight,
                      gate_letters=args.gate_letters,
                      max_tokens=args.max_tokens)
    gcfg = GeomConfig(n_points=args.n_points)

    use_cache = not args.no_cache and context_cache_ok(lm, tok)
    print(f"context cache: {'on' if use_cache else 'off'}")

    rows = []
    n_top1 = n_topk = 0
    t0 = time.time()
    idx = list(range(args.offset, len(corpus)))
    for k, i in enumerate(idx):
        dp = GestureDP(corpus.points(i), corpus.times(i), kb, gcfg)
        ctx = left_context(corpus, i) if args.context == "oracle" else ""
        ctx = " ".join(filter(None, [args.prime, ctx]))
        ids = [bos] + (tok.encode(ctx) if ctx else [])
        nbest = decode_word(lm, tok, table, dp, ids, bcfg,
                            use_cache=use_cache)
        ref = corpus.words[i]
        pred = nbest[0][0] if nbest else ""
        in_nbest = any(w == ref for w, _ in nbest)
        n_top1 += pred == ref
        n_topk += in_nbest
        rows.append((ref, pred, int(in_nbest),
                     " ".join(w for w, _ in nbest[:8])))
        if (k + 1) % 25 == 0 or k + 1 == len(idx):
            el = time.time() - t0
            print(f"  {k+1}/{len(idx)}  top-1 {n_top1/(k+1):.4f}  "
                  f"n-best {n_topk/(k+1):.4f}  ({(k+1)/el:.2f} swipes/s)",
                  flush=True)

    n = len(idx)
    print(f"\n{args.lm}  context={args.context}  lm_weight={args.lm_weight}"
          f"  beam={args.beam}  n={n}")
    print(f"top-1:          {n_top1/n:.4f}")
    print(f"n-best ceiling: {n_topk/n:.4f}")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write("word\tpred\tin_nbest\tnbest8\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
