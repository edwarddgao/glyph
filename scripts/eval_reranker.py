#!/usr/bin/env python3
"""Rerank beam candidates with a context language model.

After lexicon-constrained decoding, 85% of what remains on futo/validation are
real-word confusions -- ``has``/``had``, ``im``/``in`` -- that a lexicon cannot
separate but the previous word can.

Two things this reports honestly:

*Context source.* Scoring against the ground-truth previous word is an upper
bound. A real keyboard conditions on what it already decoded, so errors
propagate. Both are measured; the gap between them is part of the result.

*Where it applies.* Bigram coverage is reported per split, because it bounds
what context can do. FUTO transcribes prose (55.7%). How We Swipe is a mix: 93%
of its *unique* sentences are random word lists ("prior simpson clearing
tencent"), but the 6% that are natural ("im on a plane", "is that ok") repeat far
more often and cover half its swipes, giving 40.3% overall. Judging that corpus
by sampling unique sentence strings badly understates the exploitable context.

Usage:
    python scripts/eval_reranker.py --limit 20000
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (  # noqa: E402
    SwipeCorpus,
    SwipeDataset,
    decode,
    make_loader,
)
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402
from swipe_typing.model.contextlm import BOS, ContextLM, rerank  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402


def normalize(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalpha())


def sentence_words(sentence: str) -> list[str]:
    return [normalize(w) for w in sentence.split()]


def order_by_sentence(corpus: SwipeCorpus) -> list[list[int]]:
    """Group swipe indices into sentences, ordered by position in the sentence.

    Decoded-context reranking has to walk a sentence left to right, so the
    grouping must be explicit rather than relying on cache order.
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(len(corpus)):
        groups[(corpus.sessions[i], corpus.sentences[i])].append(i)
    out = []
    for key, idx in groups.items():
        idx.sort(key=lambda i: (corpus.word_idx[i]
                                if corpus.word_idx[i] >= 0 else 0))
        out.append(idx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--splits", nargs="+", default=["futo/validation"])
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--lm-unigrams", default="data/lm/count_1w.txt")
    ap.add_argument("--lm-bigrams", default="data/lm/count_2w.txt")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    lm = ContextLM.from_norvig(args.lm_unigrams, args.lm_bigrams,
                               alphabet=alphabet)
    print(f"lexicon {len(lexicon):,} words; LM {len(lm):,} unigrams, "
          f"{lm.n_bigrams:,} bigrams (Google Web corpus, external to both "
          f"eval corpora)\n")

    for split in args.splits:
        corpus = SwipeCorpus.load(root / split, alphabet, limit=args.limit)
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                          key_units=key_units)
        loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)

        chunks = []
        with torch.no_grad():
            for x, _, _ in loader:
                chunks.append(model(x.to(device)).float().cpu().numpy())
        log_probs = np.concatenate(chunks)

        cfg = BeamConfig(beam_width=args.beam_width, top_k=args.top_k)
        nbest: list[list[tuple[str, float]]] = [
            beam_search(item, lexicon, alphabet, cfg) for item in log_probs
        ]

        refs = corpus.words
        groups = order_by_sentence(corpus)

        # Is there any language here to model?
        pairs = hits = 0
        for idx in groups:
            words = [refs[i] for i in idx]
            for a, b in zip(words, words[1:]):
                pairs += 1
                hits += (a, b) in lm.bigrams
        coverage = hits / max(pairs, 1)

        base = sum(bool(n) and n[0][0] == refs[i]
                   for i, n in enumerate(nbest)) / len(refs)
        ceiling = sum(refs[i] in [w for w, _ in n]
                      for i, n in enumerate(nbest)) / len(refs)

        print(f"== {split}  n={len(refs):,} ==")
        print(f"  LM bigram coverage of this corpus: {coverage:.1%}"
              + ("" if coverage > 0.25 else
                 "   <- little context to exploit"))
        print(f"  beam top-1 {base:.4f}   top-{args.top_k} ceiling {ceiling:.4f}")
        print(f"\n  {'':>7}{'--- gated (delta) ---':>25}{'--- raw logp ---':>25}")
        print(f"  {'weight':>7}{'gt-context':>12}{'decoded ctx':>13}"
              f"{'gt-context':>12}{'decoded ctx':>13}")

        n = len(refs)
        for weight in args.weights:
            row = f"  {weight:>7.2f}"
            for gated in (True, False):
                # (a) oracle context: the true previous word
                gt = 0
                for idx in groups:
                    words = [refs[i] for i in idx]
                    for pos, i in enumerate(idx):
                        if not nbest[i]:
                            continue
                        ctx = words[pos - 1] if pos else BOS
                        hyps = rerank(nbest[i], lm, ctx, weight, gated=gated)
                        gt += hyps[0][0] == refs[i]
                # (b) streaming: condition on what was actually decoded
                dec = 0
                for idx in groups:
                    prev = BOS
                    for i in idx:
                        if not nbest[i]:
                            prev = BOS
                            continue
                        hyps = rerank(nbest[i], lm, prev, weight, gated=gated)
                        dec += hyps[0][0] == refs[i]
                        prev = hyps[0][0]
                row += f"{gt / n:>12.4f}{dec / n:>13.4f}"
            print(row + ("  <- baseline" if weight == 0 else ""))
        print()


if __name__ == "__main__":
    main()
