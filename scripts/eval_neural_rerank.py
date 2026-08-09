#!/usr/bin/env python3
"""How much is left for *any* language model? A ceiling test.

The bigram reranker bought +0.4 points and was bound by sparsity: only 31.3% of
true bigrams appear in a 286k-entry table, though on that covered slice it picks
the right word 74.2% of the time. Extrapolating to perfect coverage suggests a
bigram-class model tops out near +1.3 -- well short of the ~4.8 points of n-best
headroom on futo/validation.

Rather than build a denser LM and find out, this rescores the *existing* n-best
lists with a pretrained neural LM. No training, and one number answers whether
the LM path is worth engineering at all or whether the remaining headroom is
acoustic.

Two details that matter:

*Casing.* Our labels are lowercase letters-only, which is out of distribution for
a GPT-2-class model ("is that ok" scores below "is that on"). The corpora keep
original sentence casing, so oracle context uses the true cased prefix. Decoded
context is necessarily lowercase, and the difference between the two is reported.

*Formulation.* Adding raw log P(w | ctx) re-applies the frequency prior already
in the beam score, which measurably hurt for bigrams. The delta form
log P(w|ctx) - log P(w) is measured alongside it.

Usage:
    python scripts/eval_neural_rerank.py --limit 5000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (  # noqa: E402
    SwipeCorpus,
    SwipeDataset,
    make_loader,
)
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402
from eval_reranker import order_by_sentence  # noqa: E402


class NeuralLM:
    """Sums token log-probabilities of a candidate word given a text prefix."""

    def __init__(self, name: str, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        # "auto" keeps GPT-2 in fp32 (matching the original run) and loads
        # modern checkpoints in their native bf16, which multi-B models need
        # to fit and to run at a usable speed on MPS.
        self.model = (AutoModelForCausalLM.from_pretrained(name, dtype="auto")
                      .to(device).eval())
        self.device = device
        # Qwen-style tokenizers have no BOS; eos is their document separator.
        self.bos = (self.tok.bos_token_id if self.tok.bos_token_id is not None
                    else self.tok.eos_token_id)
        self._uncond: dict[str, float] = {}

    def _ctx_ids(self, context: str) -> torch.Tensor:
        if not context.strip():
            return torch.tensor([[self.bos]], device=self.device)
        ids = self.tok(context, return_tensors="pt").input_ids.to(self.device)
        # Keep the prompt bounded; only recent words carry useful signal.
        return ids[:, -64:]

    @torch.no_grad()
    def score(self, context: str, candidates: list[str]) -> list[float]:
        """log P(candidate | context) for each candidate, one batched pass."""
        if not candidates:
            return []
        ctx = self._ctx_ids(context)
        c_len = ctx.shape[1]
        tails = [self.tok(" " + w, return_tensors="pt").input_ids[0].to(self.device)
                 for w in candidates]
        max_t = max(len(t) for t in tails)
        n = len(candidates)

        inp = torch.full((n, c_len + max_t), self.tok.eos_token_id or 0,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros((n, c_len + max_t), dtype=torch.long, device=self.device)
        inp[:, :c_len] = ctx
        mask[:, :c_len] = 1
        for i, t in enumerate(tails):
            inp[i, c_len:c_len + len(t)] = t
            mask[i, c_len:c_len + len(t)] = 1

        logits = self.model(input_ids=inp, attention_mask=mask).logits
        logprobs = F.log_softmax(logits.float(), dim=-1)
        out = []
        for i, t in enumerate(tails):
            # position c_len-1 predicts the first candidate token
            step = logprobs[i, c_len - 1:c_len - 1 + len(t)]
            out.append(float(step.gather(1, t[:, None]).sum()))
        return out

    def unconditional(self, words: list[str]) -> list[float]:
        todo = [w for w in words if w not in self._uncond]
        if todo:
            for w, s in zip(todo, self.score("", todo)):
                self._uncond[w] = s
        return [self._uncond[w] for w in words]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--lm", default="distilgpt2")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nbest-cache", default=None,
                    help="pickle path; reuse the first pass across LM runs")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)

    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)

    cache_key = (args.checkpoint, args.split, args.limit, args.beam_width,
                 args.top_k, args.lexicon)
    cache_path = Path(args.nbest_cache) if args.nbest_cache else None
    nbest = None
    if cache_path is not None and cache_path.exists():
        import pickle
        with open(cache_path, "rb") as f:
            stored = pickle.load(f)
        if stored["key"] == cache_key:
            nbest = stored["nbest"]
            print(f"  n-best loaded from {cache_path}")
        else:
            print(f"  cache key mismatch, recomputing ({cache_path})")
    if nbest is None:
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                          key_units=key_units)
        loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)
        chunks = []
        with torch.no_grad():
            for x, _, _ in loader:
                chunks.append(model(x.to(device)).float().cpu().numpy())
        log_probs = np.concatenate(chunks)

        cfg = BeamConfig(beam_width=args.beam_width, top_k=args.top_k)
        nbest = [beam_search(item, lexicon, alphabet, cfg) for item in log_probs]
        if cache_path is not None:
            import pickle
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"key": cache_key, "nbest": nbest}, f)
            print(f"  n-best written to {cache_path}")
    refs = corpus.words
    groups = order_by_sentence(corpus)
    n = len(refs)

    base = sum(bool(b) and b[0][0] == refs[i] for i, b in enumerate(nbest)) / n
    ceiling = sum(refs[i] in [w for w, _ in b] for i, b in enumerate(nbest)) / n
    print(f"== {args.split}  n={n:,}  LM={args.lm} ==")
    print(f"  beam top-1 {base:.4f}   top-{args.top_k} ceiling {ceiling:.4f}"
          f"   headroom {ceiling - base:.4f}\n")

    lm = NeuralLM(args.lm, device)

    # Oracle context: the true cased sentence prefix.
    # Decoded context: what the reranker itself emitted, necessarily lowercase.
    cond_gt: dict[int, list[float]] = {}
    uncond: dict[int, list[float]] = {}
    print("  scoring n-best with the neural LM ...")
    for idx in groups:
        raw_words = corpus.sentences[idx[0]].split() if idx else []
        for pos, i in enumerate(idx):
            cands = [w for w, _ in nbest[i]]
            if not cands:
                continue
            k = int(corpus.word_idx[i]) if corpus.word_idx[i] >= 0 else pos
            prefix = " ".join(raw_words[:k])
            cond_gt[i] = lm.score(prefix, cands)
            uncond[i] = lm.unconditional(cands)

    def sweep(use_delta: bool, weight: float) -> float:
        hit = 0
        for i, b in enumerate(nbest):
            if not b:
                continue
            lm_s = cond_gt[i]
            if use_delta:
                lm_s = [a - u for a, u in zip(lm_s, uncond[i])]
            j = max(range(len(b)), key=lambda k: b[k][1] + weight * lm_s[k])
            hit += b[j][0] == refs[i]
        return hit / n

    print(f"\n  oracle context (true cased sentence prefix)")
    print(f"  {'weight':>7}{'raw logP':>11}{'delta':>9}")
    best = (base, 0.0, False)
    for weight in args.weights:
        raw, dlt = sweep(False, weight), sweep(True, weight)
        for val, is_d in ((raw, False), (dlt, True)):
            if val > best[0]:
                best = (val, weight, is_d)
        print(f"  {weight:>7.2f}{raw:>11.4f}{dlt:>9.4f}"
              + ("  <- baseline" if weight == 0 else ""))

    val, weight, use_delta = best
    print(f"\n  best oracle: {val:.4f} at weight {weight} "
          f"({'delta' if use_delta else 'raw'})")
    if weight == 0.0:
        print("  no setting beats the baseline; skipping the streaming pass")
        return

    # Streaming: rebuild the prefix from what was actually decoded (lowercase).
    dec = 0
    for idx in groups:
        emitted: list[str] = []
        for i in idx:
            b = nbest[i]
            if not b:
                continue
            cands = [w for w, _ in b]
            s_ = lm.score(" ".join(emitted), cands)
            if use_delta:
                s_ = [a - u for a, u in zip(s_, uncond[i])]
            j = max(range(len(b)), key=lambda k: b[k][1] + weight * s_[k])
            dec += b[j][0] == refs[i]
            emitted.append(b[j][0])
    print(f"  decoded context at the same setting: {dec / n:.4f}")
    print(f"\n  baseline {base:.4f} -> {dec / n:.4f} "
          f"({dec / n - base:+.4f}), against {ceiling - base:+.4f} available")


if __name__ == "__main__":
    main()
