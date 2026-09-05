#!/usr/bin/env python3
"""The context stage for the iPhone capture: delta-form LM over the beam's top-8.

Freeze-3-shaped stack: CTC trie beam candidates (whose scores already include
alpha*unigram + beta*length), then a sentence-level beam where each word adds
mu * (logP(w | decoded left context) - prior(w)), prior estimated over the
MARGINAL_CTXS neutral prefixes (#66/#72's corrected form). Streaming /
lookahead-1 / joint come from the commitment lag, as in run_fused_local.py.

Usage:  .venv/bin/python iphone/fused_rescore.py --lm gpt2-xl
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT.parent / "scripts"))
sys.path.insert(0, str(ROOT))

from swipe_typing.layout import ALPHABET, KeyboardLayout            # noqa: E402
from swipe_typing.model import SwipeDataset, make_loader            # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_search         # noqa: E402
from eval_decoder import build_lexicon, load_model, pick_device, run_encoder  # noqa: E402
from run_fused_local import MARGINAL_CTXS                           # noqa: E402
from decode_capture import build_corpus, latest_per_sentence        # noqa: E402

MU, BEAM, M = 0.8, 8, 8


class LMScorer:
    """logP(word | ctx) for a causal LM, batched, memoized."""

    def __init__(self, name: str, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = (AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16)
            .to(device).eval())
        self.device = device
        self.bos = self.tok.bos_token_id
        self.cache: dict[tuple[str, str], float] = {}

    @torch.no_grad()
    def fill(self, pairs: list[tuple[str, str]]) -> None:
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        if not todo:
            return
        seqs, spans = [], []
        for ctx, word in todo:
            ids = [self.bos] + (self.tok.encode(ctx) if ctx else [])
            cont = self.tok.encode((" " if ctx else "") + word)
            seqs.append(ids + cont)
            spans.append((len(ids), len(ids) + len(cont)))
        L = max(len(s) for s in seqs)
        input_ids = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        logits = self.model(input_ids.to(self.device),
                            attention_mask=mask.to(self.device)).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        for i, ((ctx, word), (a, b)) in enumerate(zip(todo, spans)):
            tot = sum(logp[i, j - 1, seqs[i][j]].item() for j in range(a, b))
            self.cache[(ctx, word)] = tot

    def prior(self, word: str) -> float:
        self.fill([(c, word) for c in MARGINAL_CTXS])
        return sum(self.cache[(c, word)] for c in MARGINAL_CTXS) / len(MARGINAL_CTXS)


def sentence_decode(slots, lm: LMScorer, lag):
    """slots: [(ref, [(word, acoustic)])] in word order. Returns hyp words."""
    states = [((), 0.0)]
    for t, (_ref, cands) in enumerate(slots):
        ctxs = [" ".join(w) for w, _ in states]
        lm.fill([(c, w) for c in ctxs for w, _ in cands])
        priors = {w: lm.prior(w) for w, _ in cands}
        expansions: dict[tuple, float] = {}
        for (words, cum), ctx in zip(states, ctxs):
            for w, ac in cands:
                sc = cum + ac + MU * (lm.cache[(ctx, w)] - priors[w])
                wt = words + (w,)
                if wt not in expansions or sc > expansions[wt]:
                    expansions[wt] = sc
        states = sorted(expansions.items(), key=lambda kv: -kv[1])[:BEAM]
        if lag is not None and t - lag >= 0:
            j = t - lag
            w_commit = states[0][0][j]
            states = [s for s in states if s[0][j] == w_commit] or states[:1]
    return list(states[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="gpt2-xl")
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    args = ap.parse_args()

    device = pick_device("auto")
    captures = latest_per_sentence("capture")
    corpus, meta = build_corpus(captures, ALPHABET)
    lexicon = build_lexicon("train+wf320k", ROOT.parent / "data/canonical",
                            ALPHABET, 1.0)

    model, alphabet, key_units, mode = load_model(
        str(ROOT.parent / args.checkpoint), device)
    ds = SwipeDataset(corpus, KeyboardLayout.qwerty(), augment_cfg=None,
                      resample_mode=mode, key_units=key_units,
                      shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=64, shuffle=False, num_workers=0)
    log_probs, refs = run_encoder(model, loader, device, alphabet)

    cfg = BeamConfig(beam_width=64, alpha=0.8, beta=1.2, top_k=M)
    cand_lists = [beam_search(lp, lexicon, alphabet, cfg)[:M] for lp in log_probs]

    # group into sentences, ordered by word_idx
    groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["key"]].append(i)
    for s in groups:
        groups[s].sort(key=lambda i: corpus.word_idx[i])

    lm = LMScorer(args.lm, device)
    print(f"LM {args.lm} on {device}; {len(groups)} sentences, {len(refs)} words")

    for lag, name in [(0, "streaming"), (1, "lookahead-1"), (None, "joint")]:
        hit = 0
        tag_hit: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        errs = []
        for s, idx in sorted(groups.items()):
            slots = [(refs[i], cand_lists[i]) for i in idx]
            out = sentence_decode(slots, lm, lag)
            for w, i in zip(out, idx):
                ok = w == refs[i]
                hit += ok
                tag_hit[meta[i]["tag"]][0] += ok
                tag_hit[meta[i]["tag"]][1] += 1
                if not ok:
                    errs.append((refs[i], w, meta[i]["tag"]))
        print(f"\n== fused {name}:  top-1 {hit / len(refs):.1%}  ({hit}/{len(refs)})")
        for tag, (c, n) in sorted(tag_hit.items()):
            print(f"   {tag:<9} {c}/{n} = {c / n:.1%}")
        if lag is None and errs:
            print("   joint errors:")
            for r, p, tag in errs:
                print(f"     {r:<12} -> {p:<12} [{tag}]")


if __name__ == "__main__":
    main()
