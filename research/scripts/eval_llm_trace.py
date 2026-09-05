#!/usr/bin/env python3
"""Training-free rung 1: zero-shot LLM decode of nearest-key traces.

No lexicon, no trained encoder, no gesture data anywhere in the decoder: the
swipe is reduced to the string of keys under the finger (``trace.key_trace``)
and a base LLM decodes it few-shot. The few-shot examples are synthesized from
straight-line templates (``trace.template_trace``), so the prompt is built
from geometry alone; the corpus is touched only to be evaluated.

    python scripts/eval_llm_trace.py --limit 500 --lm Qwen/Qwen3.5-2B-Base
    python scripts/eval_llm_trace.py --context oracle   # + left context

Reports top-1, the strict-subsequence rate (the ceiling for any decoder that
reads the trace literally — the LLM may beat it on near-misses where the
finger cut a corner), and accuracy split by that flag.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus  # noqa: E402
from swipe_typing.trace import (  # noqa: E402
    collapse,
    is_subsequence,
    key_trace,
    template_trace,
)

# Few-shot support set: common words of varied length and shape, chosen by
# hand — NOT drawn from any gesture corpus. Context strings are equally
# synthetic. Template traces for these are computed from the layout.
SHOTS = [
    ("i think", "the"),
    ("she gave me", "an"),
    ("we can", "work"),
    ("that was a", "good"),
    ("he opened the", "door"),
    ("they live in a", "house"),
    ("please turn on the", "light"),
    ("i really", "appreciate"),
    ("the weather is", "nice"),
    ("call me", "tomorrow"),
    ("a very", "important"),
    ("this is my", "favorite"),
    ("did you", "understand"),
    ("we should", "probably"),
    ("thanks for", "everything"),
    ("i am", "hungry"),
]

HEADER = (
    "Swipe keyboard decoding. Each line shows the sequence of QWERTY keys "
    "crossed by a finger swiping one word, then the intended word. The "
    "FIRST key is the word's first letter and the LAST key is the word's "
    "last letter. The word's remaining letters appear in order within the "
    "key sequence, with extra keys the finger merely crossed in between; "
    "doubled letters appear once.\n"
)


def _spaced(trace: str) -> str:
    """One key per token: BPE turns a raw 30-letter string into opaque
    chunks, so the letters are spaced to keep them readable."""
    return " ".join(trace)


def build_prompt(trace: str, context: str, kb: KeyboardLayout,
                 use_context: bool) -> str:
    lines = [HEADER]
    for ctx, word in SHOTS:
        tr = _spaced(template_trace(word, kb))
        if use_context:
            lines.append(f"context: {ctx} | keys: {tr} -> {word}")
        else:
            lines.append(f"keys: {tr} -> {word}")
    if use_context:
        lines.append(f"context: {context or '<start>'} | "
                     f"keys: {_spaced(trace)} ->")
    else:
        lines.append(f"keys: {_spaced(trace)} ->")
    return "\n".join(lines)


def left_context(corpus: SwipeCorpus, i: int, max_words: int = 8) -> str:
    sent, idx = corpus.sentences[i], int(corpus.word_idx[i])
    if not sent or idx <= 0:
        return ""
    return " ".join(sent.split()[:idx][-max_words:])


def parse_word(text: str) -> str:
    """First alphabetic token of the completion."""
    for line in text.split("\n"):
        word = "".join(ch for ch in line.strip().split(" ")[0] if ch.isalpha())
        if word:
            return word.lower()
    return ""


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/canonical/futo/validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--lm", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--context", default="none", choices=["none", "oracle"])
    ap.add_argument("--trace-mode", default="collapsed",
                    choices=["collapsed", "dwell"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dump", default="",
                    help="optional path to write per-swipe TSV")
    args = ap.parse_args()

    device = pick_device(args.device)
    kb = KeyboardLayout.qwerty()
    corpus = SwipeCorpus.load(args.data, ALPHABET, limit=args.limit)
    use_ctx = args.context == "oracle"

    from swipe_typing.schema import Swipe
    import numpy as np

    traces, prompts, subseq = [], [], []
    for i in range(len(corpus)):
        pts = corpus.points(i)
        sw = Swipe(word=corpus.words[i], x=pts[:, 0], y=pts[:, 1],
                   t=corpus.times(i), aspect=float(corpus.aspects[i]),
                   session=corpus.sessions[i], source="eval")
        tr = key_trace(sw, kb, mode=args.trace_mode)
        traces.append(tr)
        subseq.append(is_subsequence(collapse(sw.word), tr))
        prompts.append(build_prompt(tr, left_context(corpus, i), kb, use_ctx))

    tok = AutoTokenizer.from_pretrained(args.lm)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = (AutoModelForCausalLM.from_pretrained(args.lm, dtype=torch.float16)
          .to(device).eval())

    preds: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(prompts), args.batch):
            chunk = prompts[b:b + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            out = lm.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            new = out[:, enc.input_ids.shape[1]:]
            preds.extend(parse_word(t) for t in tok.batch_decode(new))
            done = min(b + args.batch, len(prompts))
            if done % (args.batch * 8) < args.batch or done == len(prompts):
                print(f"  {done}/{len(prompts)}  "
                      f"({done / (time.time() - t0):.1f} swipes/s)",
                      flush=True)

    refs = corpus.words
    n = len(refs)
    acc = sum(p == r for p, r in zip(preds, refs)) / n
    sub_rate = sum(subseq) / n
    acc_in = (sum(p == r for p, r, s in zip(preds, refs, subseq) if s)
              / max(1, sum(subseq)))
    acc_out = (sum(p == r for p, r, s in zip(preds, refs, subseq) if not s)
               / max(1, n - sum(subseq)))

    print(f"\n{args.lm}  context={args.context}  trace={args.trace_mode}"
          f"  n={n}")
    print(f"top-1:                    {acc:.4f}")
    print(f"strict-subsequence rate:  {sub_rate:.4f}")
    print(f"top-1 | subsequence ok:   {acc_in:.4f}")
    print(f"top-1 | subsequence miss: {acc_out:.4f}")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write("word\tpred\tcorrect\tsubseq\ttrace\n")
            for r, p, s, tr in zip(refs, preds, subseq, traces):
                f.write(f"{r}\t{p}\t{int(p == r)}\t{int(s)}\t{tr}\n")
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
