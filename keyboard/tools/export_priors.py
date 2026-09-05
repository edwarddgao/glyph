#!/usr/bin/env python3
"""Precompute the LM's marginal log P(word) for every lexicon word.

The fused search's delta form subtracts the LM's own prior, estimated as the
mean of log P(word | ctx) over the eight neutral contexts (fused_rescore.py's
MARGINAL_CTXS). Computing that on the phone costs eight LM sequences per new
candidate word; this table (float32 per trie node, node order of lexicon.bin,
NaN for non-words) makes it a lookup.

    ../research/.venv/bin/python tools/export_priors.py --model gpt2
"""
from __future__ import annotations

import argparse, math, struct, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
KEYBOARD = HERE.parent
RESEARCH = KEYBOARD.parent / "research"
sys.path.insert(0, str(RESEARCH / "src")); sys.path.insert(0, str(RESEARCH / "scripts")); sys.path.insert(0, str(RESEARCH / "iphone"))
from swipe_typing.layout import ALPHABET  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402
from fused_rescore import MARGINAL_CTXS  # noqa: E402
OUT = KEYBOARD / "Resources"


def bfs_nodes(lex):
    nodes = [lex.root]
    for node in nodes:
        for ch in sorted(node.children):
            nodes.append(node.children[ch])
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N words")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if device.type == "mps" else torch.float32).to(device).eval()
    bos = tok.bos_token_id

    lex = build_lexicon("train+wf320k", RESEARCH / "data/canonical", ALPHABET, 1.0)
    nodes = bfs_nodes(lex)
    # word for each node (walk down; cheap enough once)
    words = [None] * len(nodes)
    index = {id(n): i for i, n in enumerate(nodes)}
    def walk(node, prefix):
        i = index[id(node)]
        if node.is_word: words[i] = prefix
        for ch, child in node.children.items(): walk(child, prefix + ch)
    walk(lex.root, "")
    word_ids = [i for i, w in enumerate(words) if w is not None]
    if a.limit: word_ids = word_ids[:a.limit]
    print(f"{len(word_ids):,} words x {len(MARGINAL_CTXS)} contexts")

    ctx_ids = [[bos] + (tok.encode(c) if c else []) for c in MARGINAL_CTXS]
    priors = np.full(len(nodes), np.nan, np.float32)
    t0 = time.time()
    # one sequence per (ctx, word); score tail tokens; mean over ctxs
    seqs, spans, owners = [], [], []
    def flush():
        if not seqs: return
        L = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), L), tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros_like(inp)
        for i, s in enumerate(seqs):
            inp[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
        with torch.no_grad():
            logits = lm(input_ids=inp.to(device), attention_mask=mask.to(device)).logits.float()
            lp = F.log_softmax(logits, dim=-1)
            tgt = inp.to(device)
            picked = lp[:, :-1].gather(2, tgt[:, 1:, None])[:, :, 0]  # (N, L-1): logp of token j given <j
        picked = picked.cpu().numpy()
        for i, ((s0, s1), (node, k)) in enumerate(zip(spans, owners)):
            acc[(node)][k] = float(picked[i, s0 - 1:s1 - 1].sum())
        seqs.clear(); spans.clear(); owners.clear()
    acc = {}
    for n, node in enumerate(word_ids):
        w = words[node]
        acc[node] = [0.0] * len(MARGINAL_CTXS)
        for k, (c, cids) in enumerate(zip(MARGINAL_CTXS, ctx_ids)):
            cont = tok.encode((" " if c else "") + w)
            seqs.append(cids + cont); spans.append((len(cids), len(cids) + len(cont))); owners.append((node, k))
        if len(seqs) >= a.batch:
            flush()
        if n % 20000 == 0 and n:
            print(f"  {n:,} words, {time.time() - t0:.0f}s", flush=True)
    flush()
    for node, vals in acc.items():
        priors[node] = sum(vals) / len(vals)
    path = OUT / "priors.bin"
    with open(path, "wb") as f:
        f.write(b"SWPR"); f.write(struct.pack("<II", 1, len(nodes))); f.write(priors.astype("<f4").tobytes())
    print(f"priors -> {path} ({path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
    # sanity against fused_rescore's own estimator on a few words
    import fused_rescore as fr
    ref = fr.LMScorer(a.model, torch.device("cpu")); ref.model = ref.model.float()
    for w in ["the", "priya", "downstairs", "ngl"]:
        i = words.index(w) if w in words else None
        if i is not None:
            print(f"  {w}: table {priors[i]:.3f}  fused_rescore fp32 {ref.prior(w):.3f}")


if __name__ == "__main__":
    main()
