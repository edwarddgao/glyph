#!/usr/bin/env python3
"""Export the frozen first pass into one pickle for off-machine LM rescoring.

The GPU side of #34 (`scripts/modal_rerank.py`) needs no repo, corpus, or
encoder — rescoring consumes only the n-best lists plus per-swipe sentence
context. This packages exactly that from a cache written by
`eval_neural_rerank.py --nbest-cache`, so every model in the ladder scores
byte-identical inputs.

Usage:
    python scripts/eval_neural_rerank.py --limit 5000 --nbest-cache nbest.pkl
    python scripts/export_rerank_bundle.py --nbest-cache nbest.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from swipe_typing.layout import ALPHABET
from swipe_typing.model import SwipeCorpus

sys.path.insert(0, str(Path(__file__).parent))
from eval_reranker import order_by_sentence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest-cache", required=True,
                    help="pickle from eval_neural_rerank.py --nbest-cache")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--out", default="rerank_bundle.pkl")
    args = ap.parse_args()

    with open(args.nbest_cache, "rb") as f:
        stored = pickle.load(f)
    nbest = stored["nbest"]
    # key = (checkpoint, split, limit, beam_width, top_k, lexicon)
    _, split, limit, *_ = stored["key"]
    print(f"nbest key: {stored['key']}")

    corpus = SwipeCorpus.load(Path(args.cache) / split, ALPHABET, limit=limit)
    assert len(corpus.words) == len(nbest), (len(corpus.words), len(nbest))

    bundle = {
        "key": stored["key"],
        "nbest": nbest,
        "refs": list(corpus.words),
        "sentences": list(corpus.sentences),
        "word_idx": [int(i) for i in corpus.word_idx],
        "groups": [list(map(int, g)) for g in order_by_sentence(corpus)],
    }
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    n = len(bundle["refs"])
    base = sum(bool(b) and b[0][0] == r
               for b, r in zip(nbest, bundle["refs"])) / n
    ceil = sum(r in [w for w, _ in b]
               for b, r in zip(nbest, bundle["refs"])) / n
    print(f"wrote {args.out}  n={n}  base={base:.4f}  ceiling={ceil:.4f}")


if __name__ == "__main__":
    main()
