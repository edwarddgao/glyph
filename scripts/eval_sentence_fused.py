#!/usr/bin/env python3
"""Price the horizon: the sentence-level AR decoder as the *only* scorer.

    python scripts/eval_sentence_fused.py --checkpoint runs/sent_ar/sent_ar.pt

Same sentence-beam skeleton and candidate lists as run_fused_local.py
(fused_bundle.pkl: 20k futo val swipes, deep AR-MMI lists, sentence groups),
but the score of extending a hypothesis with candidate w is a single number:

    log P_sent(w's letters + EOW | hypothesis letter history, gesture memory)

No alpha, no beta, no mu or lambda, no unigram, no external LM — the design
the ladder called "one decoder, two conditioning streams". Controls:

  --no-context   reset the token stream at each word boundary: the same
                 weights used word-level, isolating what sentence
                 self-attention adds
  seen/unseen    accuracy split by whether the sentence's text occurs in
                 futo/train (~7%% of val swipes): the memorization check
  --ar-weight    optionally mix the word-level AR score back in, testing
                 whether the monolith subsumes the separate acoustic score
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import pick_device  # noqa: E402
from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.sentence_ar import (  # noqa: E402
    SentenceARConfig, SentenceARDecoder,
)

BEAM = 8


def load_sent_ar(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_d = dict(ckpt["cfg"])
    cfg_d["dilations"] = tuple(cfg_d["dilations"])
    model = SentenceARDecoder(SentenceARConfig(**cfg_d))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    a = ckpt.get("args", {}) or {}
    return model, ckpt["alphabet"], a.get("resample_mode", "time")


class SentenceScorer:
    """Batched, cached log P(word + EOW | prefix, memories) for one sentence."""

    def __init__(self, model: SentenceARDecoder, alphabet: str,
                 device: torch.device, no_context: bool = False,
                 max_rows: int = 128):
        self.model = model
        self.cfg = model.cfg
        self.char = {c: i for i, c in enumerate(alphabet)}
        self.device = device
        self.no_context = no_context
        self.max_rows = max_rows
        self.mem_sent: torch.Tensor | None = None  # (W, T, d)
        self.cache: dict[tuple[tuple[str, ...], str], float] = {}
        self.forwards = 0
        self.rows = 0

    def start_sentence(self, mem_sent: torch.Tensor) -> None:
        self.mem_sent = mem_sent.to(self.device)
        self.cache.clear()

    def _tokens(self, prefix: tuple[str, ...], cand: str
                ) -> tuple[list[int], list[int], int]:
        """tin, tok_word, and the input index whose output is cand's first
        letter. Scored positions run from there through cand's EOW."""
        cfg = self.cfg
        tin, tword = [cfg.bos], []
        for wi, w in enumerate(prefix):
            n = len(w) + 1                       # letters + closing EOW
            tin.extend([self.char[c] for c in w] + [cfg.eow])
            tword.extend([wi] * n)
        t = len(prefix)
        start = len(tin) - 1
        tin.extend(self.char[c] for c in cand)
        # Positions predicting cand: its first letter (from BOS/EOW), each
        # following letter, and the closing EOW — all conditioned on word t.
        tword.extend([t] * (len(cand) + 1))
        return tin, tword, start

    def score_step(self, t: int, prefixes: list[tuple[str, ...]],
                   cands: list[str]) -> list[list[float]]:
        """Scores for every (prefix, candidate) pair at word position t.

        One forward for all uncached pairs across the whole beam — the search
        is kernel-launch-bound (#51), so per-hypothesis forwards would waste
        most of the wall clock.
        """
        keys = [(t, () if self.no_context else p, w)
                for p in prefixes for w in cands]
        todo = sorted({k for k in keys if k not in self.cache})
        for s in range(0, len(todo), self.max_rows):
            self._forward(todo[s:s + self.max_rows])
        it = iter(keys)
        return [[self.cache[next(it)] for _ in cands] for _ in prefixes]

    @torch.no_grad()
    def _forward(self, batch: list[tuple[int, tuple[str, ...], str]]) -> None:
        cfg = self.cfg
        rows = []
        for t, prefix, w in batch:
            tin, tword, start = self._tokens(prefix, w)
            rows.append((tin, tword, start))
        L = max(len(r[0]) for r in rows)
        n = len(rows)
        W = max(t for t, _, _ in batch) + 1
        T, d = self.mem_sent.shape[1], self.mem_sent.shape[2]
        tin_b = torch.zeros(n, L, dtype=torch.long)
        tword_b = torch.zeros(n, L, dtype=torch.long)
        memory = torch.zeros(n, W, T, d, device=self.device)
        for i, ((t, prefix, w), (tin, tword, _)) in enumerate(zip(batch, rows)):
            tin_b[i, :len(tin)] = torch.tensor(tin)
            # tword aligns with input positions (position i predicts output i).
            tword_b[i, :len(tword)] = torch.tensor(tword)
            if self.no_context:
                memory[i, 0] = self.mem_sent[t]
            else:
                memory[i, :t + 1] = self.mem_sent[:t + 1]
        tin_b, tword_b = tin_b.to(self.device), tword_b.to(self.device)
        logits = self.model.decode_step(memory, tin_b, tword_b)
        lp = F.log_softmax(logits.float(), dim=-1)
        for i, ((t, prefix, w), (tin, _, start)) in enumerate(zip(batch, rows)):
            ids = [self.char[c] for c in w] + [cfg.eow]
            step = lp[i, start:start + len(ids)]
            tgt = torch.tensor(ids, device=self.device)
            self.cache[(t, prefix, w)] = float(
                step.gather(1, tgt[:, None]).sum())
        self.forwards += 1
        self.rows += n


def decode_sentence(g, lists, scorer: SentenceScorer, m: int,
                    lag: int | None, ar_weight: float) -> list[str]:
    states: list[tuple[tuple[str, ...], float]] = [((), 0.0)]
    n_words = len(g)
    for t, i in enumerate(g):
        cl = lists[i][:m]
        if cl:
            words_c = [w for w, *_ in cl]
            ar_lp = {w: ar for w, ar, *_ in cl}
            all_scs = scorer.score_step(t, [words for words, _ in states],
                                        words_c)
            expansions: dict[tuple[str, ...], float] = {}
            for (words, cum), scs in zip(states, all_scs):
                for w, sc in zip(words_c, scs):
                    wt = words + (w,)
                    total = cum + sc + ar_weight * ar_lp[w]
                    if wt not in expansions or total > expansions[wt]:
                        expansions[wt] = total
            states = sorted(expansions.items(), key=lambda kv: -kv[1])[:BEAM]
        else:
            states = [(words + ("",), cum) for words, cum in states]
        if lag is not None and t - lag >= 0:
            j = t - lag
            w_commit = states[0][0][j]
            states = [s for s in states if s[0][j] == w_commit] or states[:1]
    return list(states[0][0]) + [""] * (n_words - len(states[0][0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/sent_ar/sent_ar.pt")
    ap.add_argument("--bundle", default="fused_bundle.pkl")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--ar-weight", type=float, default=0.0)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--limit-sentences", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, mode = load_sent_ar(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    lists, refs, groups = bundle["lists"], bundle["refs"], bundle["groups"]
    n = len(refs)

    corpus = SwipeCorpus.load(Path(args.cache) / args.split, alphabet, limit=n)
    assert list(corpus.words) == list(refs), "bundle/corpus misaligned"

    print("featurizing + encoding gesture memories...")
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=4)
    mems = []
    with torch.no_grad():
        for x, _, _ in loader:
            mems.append(model.encode(x.to(device)).cpu())
    mems = torch.cat(mems)
    print(f"memories: {tuple(mems.shape)}")

    train_sents = set()
    train_path = Path(args.cache) / "futo/train"
    if train_path.exists():
        tr = SwipeCorpus.load(train_path, alphabet)
        train_sents = set(tr.sentences)
        del tr
    seen_mask = np.array([s in train_sents for s in bundle["sentences"]])
    print(f"val swipes in train-seen sentences: {seen_mask.mean():.1%}")

    if args.limit_sentences:
        groups = groups[:args.limit_sentences]
    covered = [i for g in groups for i in g]
    scorer = SentenceScorer(model, alphabet, device,
                            no_context=args.no_context)

    hits = {name: np.zeros(n, dtype=bool) for name in
            ("streaming", "lookahead1", "joint")}
    chosen = {name: [""] * n for name in hits}
    t0 = time.time()
    for gi, g in enumerate(groups):
        mem_sent = mems[torch.tensor(g)]
        scorer.start_sentence(mem_sent)
        for lag, name in ((0, "streaming"), (1, "lookahead1"), (None, "joint")):
            out = decode_sentence(g, lists, scorer, args.m, lag,
                                  args.ar_weight)
            for w, i in zip(out, g):
                chosen[name][i] = w
                hits[name][i] = (w == refs[i])
        if (gi + 1) % 200 == 0:
            done = sum(len(x) for x in groups[:gi + 1])
            rate = done / (time.time() - t0)
            eta = (len(covered) - done) / rate / 60
            print(f"  {gi + 1}/{len(groups)} sentences  "
                  f"({rate:.0f} swipes/s x3 passes, eta {eta:.0f}m, "
                  f"{scorer.forwards:,} forwards)", flush=True)

    idx = np.array(covered)
    ceiling = np.mean([refs[i] in [w for w, *_ in lists[i][:args.m]]
                       for i in idx])
    print(f"\nn = {len(idx):,} swipes   ceiling@{args.m} {ceiling:.4f}   "
          f"ar_weight={args.ar_weight} no_context={args.no_context}")
    seen_idx = idx[seen_mask[idx]]
    unseen_idx = idx[~seen_mask[idx]]
    print(f"{'mode':<14}{'top-1':<10}{'seen-sent':<12}{'unseen-sent'}")
    for name in ("streaming", "lookahead1", "joint"):
        h = hits[name]
        row = f"{name:<14}{h[idx].mean():<10.4f}"
        row += (f"{h[seen_idx].mean():<12.4f}" if len(seen_idx) else f"{'--':<12}")
        row += (f"{h[unseen_idx].mean():.4f}" if len(unseen_idx) else "--")
        print(row)
    print(f"\n{scorer.forwards:,} forwards, {scorer.rows:,} rows, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
