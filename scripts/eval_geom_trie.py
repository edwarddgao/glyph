#!/usr/bin/env python3
"""Training-free geometry + lexicon trie: the cell that isolates the loss.

The joint LLM-geometry beam (#68) replaced BOTH trained stages at once: the
encoder with analytic geometry, and the lexicon with the LM's vocabulary.
Its 7-10 point deficit could sit in either substitution. This decoder keeps
the training-free first stage and puts the trie back: hypotheses are lexicon
words, scored by the same ``GestureDP`` alignment cost, with an optional
wordfreq unigram prior and an optional LLM rescore of the n-best — every
component still innocent of gesture data. If this lands near the trained
trie beam, the first-stage substitution is ~free and the entire #68 deficit
is enumeration.

    python scripts/eval_geom_trie.py --offset 50 --limit 150
    python scripts/eval_geom_trie.py --offset 50 --limit 150 \
        --lm gpt2-xl --context oracle
"""

from __future__ import annotations

import argparse
import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

from swipe_typing.geomllm import GeomConfig, GestureDP  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus  # noqa: E402
from swipe_typing.model.lexicon import english_counts  # noqa: E402


class Trie:
    __slots__ = ("children", "word")

    def __init__(self) -> None:
        self.children: dict[str, Trie] = {}
        self.word: str | None = None


def build_trie(words) -> Trie:
    root = Trie()
    for w in words:
        node = root
        for ch in w:
            node = node.children.setdefault(ch, Trie())
        node.word = w
    return root


def decode(dp: GestureDP, root: Trie, kb: KeyboardLayout, beam: int,
           max_depth: int = 24) -> list[tuple[str, float]]:
    """Geometric beam over the trie; returns (word, alignment cost)."""
    init = dp.init_all()                                  # (K, N)
    frontier = [(node, ch, init[kb.index(ch)])
                for ch, node in root.children.items()]
    out: dict[str, float] = {}
    for _ in range(max_depth):
        if not frontier:
            break
        nxt = []
        for node, prefix_last, row in frontier:
            if node.word is not None:
                g = dp.final(row, prefix_last)
                if node.word not in out or out[node.word] > g:
                    out[node.word] = g
            if node.children:
                ext = dp.extend_many(row, prefix_last)
                for ch, child in node.children.items():
                    nxt.append((child, ch, ext[kb.index(ch)]))
        nxt.sort(key=lambda t: float(np.min(t[2] + dp.tail_bound)))
        frontier = nxt[:beam]
    return sorted(out.items(), key=lambda kv: kv[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/canonical/futo/validation")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--lexicon", type=int, default=320_000,
                    help="top-N wordfreq words; no corpus vocabulary")
    ap.add_argument("--beam", type=int, default=2000)
    ap.add_argument("--unigram", type=float, default=0.5,
                    help="weight on log p_unigram(w)")
    ap.add_argument("--lm", default="",
                    help="optional LLM to rescore the geometric n-best")
    ap.add_argument("--lm-weight", type=float, default=1.0)
    ap.add_argument("--context", default="none", choices=["none", "oracle"])
    ap.add_argument("--prime", default="the following is a transcript of "
                                       "spoken english. she said")
    ap.add_argument("--rescore-top", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=500,
                    help="geometric candidates kept before the prior")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    corpus = SwipeCorpus.load(args.data, ALPHABET,
                              limit=args.offset + args.limit)
    counts = english_counts(args.lexicon, alphabet=ALPHABET)
    total = sum(counts.values())
    logp = {w: math.log(c / total) for w, c in counts.items()}
    root = build_trie(counts.keys())
    print(f"lexicon: {len(counts)} words")

    lm = tok = None
    bos = None
    if args.lm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.lm)
        device = ("mps" if torch.backends.mps.is_available() else "cpu")
        lm = (AutoModelForCausalLM.from_pretrained(args.lm,
                                                   dtype=torch.float16)
              .to(device).eval())
        bos = tok.bos_token_id or tok.eos_token_id

    def lm_scores(ctx_ids: list[int], words: list[str]) -> np.ndarray:
        """log P(word | ctx), canonical tokenization, right-padded batch."""
        import torch
        device = next(lm.parameters()).device
        seqs = [ctx_ids + tok.encode(" " + w) for w in words]
        width = max(len(s) for s in seqs)
        pad = tok.pad_token_id or tok.eos_token_id
        inp = torch.full((len(seqs), width), pad, dtype=torch.long)
        for r, s in enumerate(seqs):
            inp[r, :len(s)] = torch.tensor(s)
        with torch.no_grad():
            logits = lm(input_ids=inp.to(device)).logits
            lse = torch.stack([torch.logsumexp(logits[:, w].float(), -1)
                               for w in range(width)], 1)
            nexts = torch.zeros(len(seqs), width, dtype=torch.long)
            want = torch.zeros(len(seqs), width, dtype=torch.bool)
            for r, s in enumerate(seqs):
                for j in range(len(ctx_ids) - 1, len(s) - 1):
                    nexts[r, j] = s[j + 1]
                    want[r, j] = True
            tok_lp = (logits.gather(2, nexts.to(device)[:, :, None])[:, :, 0]
                      .float() - lse)
            return (tok_lp * want.to(device)).sum(1).cpu().numpy()

    def left_context(i: int) -> str:
        sent, idx = corpus.sentences[i], int(corpus.word_idx[i])
        if not sent or idx <= 0:
            return ""
        return " ".join(sent.split()[:idx][-8:])

    rows = []
    n_top1 = n_topk = n_oov = 0
    t0 = time.time()
    idx = list(range(args.offset, len(corpus)))
    floor = min(logp.values()) - 2.0
    for k, i in enumerate(idx):
        dp = GestureDP(corpus.points(i), corpus.times(i), kb, GeomConfig())
        cands = decode(dp, root, kb, args.beam)[:args.candidates]
        scored = [(w, -g + args.unigram * logp.get(w, floor))
                  for w, g in cands]
        scored.sort(key=lambda t: -t[1])
        if lm is not None:
            short = scored[:args.rescore_top]
            ctx = left_context(i) if args.context == "oracle" else ""
            ctx = " ".join(filter(None, [args.prime
                                         if args.context == "none" else "",
                                         ctx]))
            ctx_ids = [bos] + (tok.encode(ctx) if ctx else [])
            lps = lm_scores(ctx_ids, [w for w, _ in short])
            geom = dict(cands)
            scored = [(w, args.lm_weight * lps[j] - geom[w])
                      for j, (w, _) in enumerate(short)]
            scored.sort(key=lambda t: -t[1])
        ref = corpus.words[i]
        pred = scored[0][0] if scored else ""
        n_top1 += pred == ref
        in_k = any(w == ref for w, _ in scored[:8])
        n_topk += in_k
        n_oov += ref not in counts
        rows.append((ref, pred, int(in_k),
                     " ".join(w for w, _ in scored[:8])))
        if (k + 1) % 25 == 0 or k + 1 == len(idx):
            el = time.time() - t0
            print(f"  {k+1}/{len(idx)}  top-1 {n_top1/(k+1):.4f}  "
                  f"top-8 {n_topk/(k+1):.4f}  ({(k+1)/el:.2f} swipes/s)",
                  flush=True)

    n = len(idx)
    print(f"\ngeom+trie  lexicon={args.lexicon}  unigram={args.unigram}"
          f"  lm={args.lm or 'none'}  context={args.context}  n={n}")
    print(f"top-1:        {n_top1/n:.4f}")
    print(f"top-8:        {n_topk/n:.4f}")
    print(f"OOV refs:     {n_oov} (unreachable)")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write("word\tpred\tin_top8\ttop8\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
