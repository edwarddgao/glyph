#!/usr/bin/env python3
"""Does deferring commitment across swipes buy accuracy? A speculative-decoding
style measurement.

Every context-LM number so far commits each word before the next swipe exists,
so the LM only ever sees *left* context. Speculative decoding's framing -- a
cheap draft accepted or revised by a stronger verifier over a window -- maps
directly onto this stack: the first pass drafts a word per swipe, and instead
of committing immediately we hold a small hypothesis set open until later
swipes arrive, letting word i+1's evidence veto word i. Concretely that is a
beam over the sentence's word lattice (n-best per swipe), scored with the same
delta-LM formulation the streaming reranker uses:

    total = sum_i [ first_pass(w_i) + lm_w * (log P(w_i|w_<i) - log P(w_i)) ]

Streaming commit is exactly this beam with width 1, so one script yields an
apples-to-apples comparison:

  draft       first-pass argmax, no LM
  streaming   width-1: commit each word with left context only (current stack)
  lookahead-1 width-B, but word i is frozen once word i+1 is processed
  joint       width-B over the whole sentence (unbounded lookahead)

Alongside top-1 it reports the numbers a real deferred-commit UI needs: how
often the verifier accepts the draft, how often an already-displayed word
would be revised, and how much of the joint gain survives when only
low-margin words are allowed to defer.

Usage:
    python scripts/dump_nbest.py --split futo/validation --limit 20000
    python scripts/eval_deferred_commit.py --limit 8000
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus  # noqa: E402
from swipe_typing.model.rescore import RescoreConfig, Rescorer  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import pick_device  # noqa: E402
from eval_full_stack import rescorer_logits  # noqa: E402
from eval_reranker import order_by_sentence  # noqa: E402
from train_rescorer import NBestSet  # noqa: E402


class BatchedDeltaLM:
    """Scores (context, word) pairs with mixed contexts in one forward pass.

    NeuralLM batches candidates under a single context; a lattice beam needs
    B contexts x K candidates per step, so this pads per-row contexts instead
    and caches every (context, word) score -- the width-1 and width-B passes
    share most of their queries.
    """

    def __init__(self, name: str, device: torch.device, ctx_tokens: int = 64,
                 truecase: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.device = device
        self.ctx_tokens = ctx_tokens
        self.truecase = truecase
        self.pad = self.tok.eos_token_id or 0
        self.bos = self.tok.bos_token_id
        self._cond: dict[tuple[str, str], float] = {}
        self._word_ids: dict[str, list[int]] = {}
        self._ctx_ids: dict[str, list[int]] = {}
        self.forwards = 0
        self.rows = 0

    def _word(self, w: str) -> list[int]:
        if w not in self._word_ids:
            self._word_ids[w] = self.tok(" " + w).input_ids
        return self._word_ids[w]

    def _ctx(self, c: str) -> list[int]:
        if c not in self._ctx_ids:
            if not c.strip():
                self._ctx_ids[c] = [self.bos]
            else:
                text = c
                if self.truecase:
                    # Decoded words are lowercase letters-only, which GPT-2
                    # never saw; restore the two recoverable conventions.
                    ws = text.split()
                    ws[0] = ws[0].capitalize()
                    text = " ".join("I" if w == "i" else w for w in ws)
                self._ctx_ids[c] = self.tok(text).input_ids[-self.ctx_tokens:]
        return self._ctx_ids[c]

    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """log P(word | context) per pair; empty context means unconditional."""
        todo = sorted({p for p in pairs if p not in self._cond})
        if todo:
            ctxs = [self._ctx(c) for c, _ in todo]
            tails = [self._word(w) for _, w in todo]
            width = max(len(c) + len(t) for c, t in zip(ctxs, tails))
            n = len(todo)
            inp = torch.full((n, width), self.pad, dtype=torch.long)
            mask = torch.zeros((n, width), dtype=torch.long)
            for i, (c, t) in enumerate(zip(ctxs, tails)):
                inp[i, :len(c)] = torch.tensor(c)
                inp[i, len(c):len(c) + len(t)] = torch.tensor(t)
                mask[i, :len(c) + len(t)] = 1
            inp, mask = inp.to(self.device), mask.to(self.device)
            logits = self.model(input_ids=inp, attention_mask=mask).logits
            logprobs = F.log_softmax(logits.float(), dim=-1)
            for i, (c, t) in enumerate(zip(ctxs, tails)):
                step = logprobs[i, len(c) - 1:len(c) - 1 + len(t)]
                tt = torch.tensor(t, device=self.device)
                self._cond[todo[i]] = float(step.gather(1, tt[:, None]).sum())
            self.forwards += 1
            self.rows += n
        return [self._cond[p] for p in pairs]

    def delta(self, pairs: list[tuple[str, str]]) -> list[float]:
        cond = self.score(pairs)
        uncond = self.score([("", w) for _, w in pairs])
        return [c - u for c, u in zip(cond, uncond)]


def beam_decode(sent: list[int], cands, scores, valid, lm: BatchedDeltaLM,
                lm_w: float, width: int, commit_lag: int | None) -> list[int]:
    """Return the chosen candidate slot per swipe in `sent`.

    commit_lag=None decodes the whole sentence before committing anything;
    commit_lag=L freezes word i once word i+L has been processed (L=0 with
    width 1 is streaming commit).
    """
    hyps: list[tuple[list[int], list[str], float]] = [([], [], 0.0)]
    for step, i in enumerate(sent):
        opts = [(j, str(cands[i][j]), float(scores[i][j]))
                for j in range(len(cands[i])) if valid[i][j]]
        if not opts:
            hyps = [(slots + [-1], words, s) for slots, words, s in hyps]
            continue
        if lm_w == 0.0:
            deltas = [0.0] * (len(hyps) * len(opts))
        else:
            pairs = [(" ".join(words), w)
                     for _, words, _ in hyps for _, w, _ in opts]
            deltas = lm.delta(pairs)
        nxt = []
        k = 0
        for slots, words, s in hyps:
            for j, w, fp in opts:
                nxt.append((slots + [j], words + [w], s + fp + lm_w * deltas[k]))
                k += 1
        nxt.sort(key=lambda h: -h[2])
        # Dedupe identical word sequences (distinct slots can hold one word).
        seen, hyps = set(), []
        for h in nxt:
            key = tuple(h[1])
            if key not in seen:
                seen.add(key)
                hyps.append(h)
            if len(hyps) == width:
                break
        if commit_lag is not None and step >= commit_lag:
            frozen = hyps[0][0][step - commit_lag]
            hyps = [h for h in hyps if h[0][step - commit_lag] == frozen] or hyps[:1]
    return hyps[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", default="data/nbest/futo_validation.npz")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--lm", default="gpt2")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--lm-weights", nargs="+", type=float, default=[0.8])
    ap.add_argument("--beams", nargs="+", type=int, default=[8])
    ap.add_argument("--rescorer", default="")
    ap.add_argument("--rescorer-weight", type=float, default=1.0)
    ap.add_argument("--max-sentence", type=int, default=32)
    ap.add_argument("--truecase", action="store_true",
                    help="capitalize sentence starts and 'i' in decoded "
                         "context before LM scoring")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    npz = np.load(args.nbest, allow_pickle=False)
    n = min(args.limit, len(npz["words"]))
    cands, scores = npz["candidates"][:n], npz["scores"][:n]
    valid, target, words = npz["valid"][:n], npz["target"][:n], npz["words"][:n]
    valid = valid & np.isfinite(scores)

    if args.rescorer:
        kb = KeyboardLayout.qwerty()
        nb = NBestSet(Path(args.nbest), kb)
        ckpt = torch.load(args.rescorer, map_location="cpu", weights_only=False)
        cfg_d = dict(ckpt["cfg"])
        cfg_d["dilations"] = tuple(cfg_d["dilations"])
        rmodel = Rescorer(RescoreConfig(**cfg_d))
        rmodel.load_state_dict(ckpt["model"])
        rmodel.to(device).eval()
        combined = rescorer_logits(rmodel, nb, device)
        rscale = float(rmodel.first_pass_scale.detach().cpu())
        residual = combined - rscale * nb.scores
        scores = scores + args.rescorer_weight * residual[:n]
        print(f"rescorer {args.rescorer} stacked at "
              f"weight {args.rescorer_weight}")

    corpus = SwipeCorpus.load(Path(args.cache) / args.split,
                              "abcdefghijklmnopqrstuvwxyz", limit=n)
    agree = np.mean([words[i] == corpus.words[i] for i in range(n)])
    assert agree > 0.999, f"n-best dump and corpus misaligned ({agree:.4f})"

    groups = [g[:args.max_sentence] for g in order_by_sentence(corpus)
              if all(i < n for i in g)]
    covered = [i for g in groups for i in g]
    lens = {i: len(g) for g in groups for i in g}
    pos = {i: p for g in groups for p, i in enumerate(g)}
    print(f"{args.split}: {n:,} swipes, {len(groups):,} sentences "
          f"(mean {np.mean([len(g) for g in groups]):.1f} words), "
          f"lm={args.lm} weights={args.lm_weights} beams={args.beams}")

    lm = BatchedDeltaLM(args.lm, device, truecase=args.truecase)
    modes = {"draft": dict(lm_w=0.0, width=1, commit_lag=0)}
    for lw in args.lm_weights:
        modes[f"streaming lm={lw}"] = dict(lm_w=lw, width=1, commit_lag=0)
        for b in args.beams:
            modes[f"look1 b={b} lm={lw}"] = dict(lm_w=lw, width=b,
                                                 commit_lag=1)
            modes[f"joint b={b} lm={lw}"] = dict(lm_w=lw, width=b,
                                                 commit_lag=None)
    choice = {m: np.full(n, -1, dtype=np.int64) for m in modes}
    t0 = time.time()
    for gi, g in enumerate(groups):
        for m, kw in modes.items():
            slots = beam_decode(g, cands, scores, valid, lm, **kw)
            for i, j in zip(g, slots):
                choice[m][i] = j
        if (gi + 1) % 200 == 0:
            done = sum(len(x) for x in groups[:gi + 1])
            rate = done / (time.time() - t0)
            print(f"  {gi + 1}/{len(groups)} sentences  "
                  f"({rate:.1f} swipes/s, {lm.forwards} forwards, "
                  f"cache {len(lm._cond):,})", flush=True)

    idx = np.array(covered)
    # A word whose truth never made the n-best is an error in every mode; keep
    # it in the denominator so numbers line up with the README convention.
    scored = idx

    def acc(m, sel):
        hit = (choice[m][sel] == target[sel]) & (target[sel] >= 0)
        return float(hit.mean())

    ceiling = float((target[idx] >= 0).mean())  # true word in list
    print(f"\nn = {len(scored):,} covered words   "
          f"n-best ceiling {ceiling:.4f}\n")
    inner = scored[np.array([pos[i] < lens[i] - 1 for i in scored])]
    last = scored[np.array([pos[i] == lens[i] - 1 for i in scored])]
    print(f"  {'mode':<22}{'top-1':<10}{'right ctx':<12}{'last':<10}"
          f"{'accept':<10}{'revise'}")
    for m in modes:
        lw = modes[m]["lm_w"]
        stream = f"streaming lm={lw}"
        row = f"  {m:<22}{acc(m, scored):<10.4f}"
        row += f"{acc(m, inner):<12.4f}" if len(inner) else f"{'--':<12}"
        row += f"{acc(m, last):<10.4f}" if len(last) else f"{'--':<10}"
        if m != "draft" and not m.startswith("streaming"):
            accept = float((choice[m][scored]
                            == choice["draft"][scored]).mean())
            revise = float((choice[m][scored]
                            != choice[stream][scored]).mean())
            row += f"{accept:<10.1%}{revise:.1%}"
        print(row)

    # Margin-gated deferral for the best joint config: only low-margin words
    # wait for more evidence, everything else commits as it streams.
    best = max((acc(m, scored), m) for m in modes if m.startswith("joint"))[1]
    stream = f"streaming lm={modes[best]['lm_w']}"
    top2 = np.sort(np.where(valid, scores, -np.inf), axis=1)
    margin = top2[:, -1] - top2[:, -2]
    print(f"\n  gated deferral, {best} vs {stream}:")
    print(f"  {'defer if margin <':<20}{'defer rate':<12}{'top-1'}")
    for tau in (0.0, 0.5, 1.0, 2.0, 4.0, np.inf):
        gate = margin[scored] < tau
        mixed = np.where(gate, choice[best][scored], choice[stream][scored])
        a = float(((mixed == target[scored])
                   & (target[scored] >= 0)).mean())
        print(f"  {tau:<20}{gate.mean():<12.3f}{a:.4f}")

    print(f"\n  LM cost: {lm.forwards:,} forwards, {lm.rows:,} rows, "
          f"{len(lm._cond):,} unique (ctx, word) scores, "
          f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
