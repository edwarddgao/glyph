#!/usr/bin/env python3
"""Score the iPhone capture study: our frozen stack vs the native keyboard.

Block A (capture_*.json): raw gestures on the canonical drawn keyboard,
decoded here with the frozen encoders + trie beam. Block B (native_*.json):
what the user's own iOS keyboard committed for the same sentences.

Dedup keeps the latest upload per (kind, sentence). Word alignment for the
native block is Levenshtein at the word level, letters-only lowercase on
both sides — the same normalization the corpora use.

Usage:  .venv/bin/python iphone/decode_capture.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent          # research/capture
sys.path.insert(0, str(ROOT.parent / "src"))

from swipe_typing.layout import ALPHABET, KeyboardLayout            # noqa: E402
from swipe_typing.model import SwipeDataset, decode, make_loader    # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_search         # noqa: E402
from swipe_typing.model.data import SwipeCorpus                     # noqa: E402

sys.path.insert(0, str(ROOT.parent / "scripts"))
from eval_decoder import build_lexicon, load_model, pick_device, run_encoder  # noqa: E402

DATA = ROOT / "data"


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def latest_per_sentence(kind: str) -> dict[tuple[str, str], dict]:
    """Latest upload per (session, sentence) — multi-user safe."""
    best: dict[tuple[str, str], dict] = {}
    for f in DATA.glob(f"{kind}_*.json"):
        p = json.loads(f.read_text())
        k = (p.get("session", "anon"), p["sentence"])
        if k not in best or p["ts"] > best[k]["ts"]:
            best[k] = p
    return best


def word_align(ref: list[str], hyp: list[str]):
    """Word-level Levenshtein; returns (#correct, #ref, aligned pairs)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            pairs.append((ref[i - 1], None)); i -= 1
        else:
            pairs.append((None, hyp[j - 1])); j -= 1
    pairs.reverse()
    correct = sum(1 for r, h in pairs if r is not None and r == h)
    return correct, n, pairs


def build_corpus(captures: dict[str, dict], alphabet: str) -> tuple[SwipeCorpus, list[dict]]:
    xs, ys, ts, offsets, words = [], [], [], [0], []
    aspects, sentences, word_idx, meta = [], [], [], []
    for p in captures.values():
        for g in p["gestures"]:
            w = norm_word(g["word"])
            if not w or any(c not in alphabet for c in w):
                continue
            xs.extend(g["x"]); ys.extend(g["y"]); ts.extend(g["t"])
            offsets.append(len(xs))
            words.append(w)
            aspects.append(g["aspect"])
            sentences.append(p["sentence"])
            word_idx.append(g["word_idx"])
            meta.append({"tag": p["tag"], "sentence": p["sentence"],
                         "session": p.get("session", "anon"),
                         "set": p.get("set", 1),
                         "key": p.get("session", "anon") + "|" + p["sentence"]})
    corpus = SwipeCorpus(
        np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
        np.asarray(ts, dtype=np.int32), np.asarray(offsets, dtype=np.int64),
        words, np.asarray(aspects, dtype=np.float32),
        sentences=sentences, word_idx=word_idx,
    )
    return corpus, meta


def decode_block_a(corpus, meta, lexicon, checkpoint: str, device):
    model, alphabet, key_units, mode = load_model(checkpoint, device)
    kb = KeyboardLayout.qwerty()
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units, shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=64, shuffle=False, num_workers=0)
    log_probs, refs = run_encoder(model, loader, device, alphabet)

    greedy = decode.greedy_decode(torch.from_numpy(log_probs),
                                  model.cfg.blank, alphabet)
    cfg = BeamConfig(beam_width=64, alpha=0.8, beta=1.2, top_k=8)
    top1, in8 = [], []
    for lp in log_probs:
        hyps = beam_search(lp, lexicon, alphabet, cfg)
        ws = [w for w, _ in hyps]
        top1.append(ws[0] if ws else "")
        in8.append(ws[:8])
    return refs, greedy, top1, in8


def report_block(name, refs, top1, in8, meta):
    n = len(refs)
    acc = sum(p == r for p, r in zip(top1, refs)) / n
    hit8 = sum(r in c for r, c in zip(refs, in8)) / n
    print(f"\n== {name}:  top-1 {acc:.1%}  ({sum(p == r for p, r in zip(top1, refs))}/{n})"
          f"   truth-in-top8 {hit8:.1%}")
    for tag in ("everyday", "tail"):
        idx = [i for i, m in enumerate(meta) if m["tag"] == tag]
        a = sum(top1[i] == refs[i] for i in idx) / len(idx)
        h = sum(refs[i] in in8[i] for i in idx) / len(idx)
        print(f"   {tag:<9} top-1 {a:.1%}  in-top8 {h:.1%}  (n={len(idx)})")
    errs = [(refs[i], top1[i], in8[i], meta[i]) for i in range(n) if top1[i] != refs[i]]
    if errs:
        print("   errors:")
        for r, p, c, m in errs:
            where = "in-top8" if r in c else "MISSED-LIST"
            print(f"     {r:<12} -> {p:<12} [{m['tag']}, {where}]")
    return acc


def main():
    device = pick_device("auto")
    captures = latest_per_sentence("capture")
    natives = latest_per_sentence("native")
    print(f"sentences: capture {len(captures)}, native {len(natives)}")

    alphabet = ALPHABET
    corpus, meta = build_corpus(captures, alphabet)
    print(f"gestures: {len(corpus)}  "
          f"({sum(1 for m in meta if m['tag'] == 'tail')} tail)")

    lexicon = build_lexicon("train+wf320k", ROOT.parent / "data/canonical",
                            alphabet, 1.0)
    oov = [w for w in corpus.words if w not in lexicon]
    print(f"lexicon {len(lexicon):,}; eval words outside it: {oov or 'none'}")

    results = {}
    for name, ckpt in [("canonical (runs/full)", "runs/full/encoder.pt"),
                       ("MMI frozen (runs/mmi)", "runs/mmi/encoder_ep0.pt")]:
        refs, greedy, top1, in8 = decode_block_a(
            corpus, meta, lexicon, str(ROOT.parent / ckpt), device)
        g = sum(p == r for p, r in zip(greedy, refs)) / len(refs)
        print(f"\n[{name}]  greedy {g:.1%}")
        results[name] = report_block(name, refs, top1, in8, meta)

    # ---- Block B: the native keyboard, word-aligned ----
    print("\n== native keyboard (QuickPath), aligned word-level ==")
    tot_c = tot_n = 0
    tag_c: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (_sess, s), p in sorted(natives.items()):
        ref = [norm_word(w) for w in s.split()]
        hyp = [norm_word(w) for w in p["typed"].split() if norm_word(w)]
        c, n, pairs = word_align(ref, hyp)
        tot_c += c; tot_n += n
        tag_c[p["tag"]][0] += c; tag_c[p["tag"]][1] += n
        bad = [(r, h) for r, h in pairs if r != h]
        flag = "  ".join(f"{r or '∅'}->{h or '∅'}" for r, h in bad)
        print(f"   {s:<42} {c}/{n}" + (f"   {flag}" if flag else ""))
    print(f"\n   native total: {tot_c}/{tot_n} = {tot_c / tot_n:.1%}")
    for tag, (c, n) in tag_c.items():
        print(f"   {tag:<9} {c}/{n} = {c / n:.1%}")

    print("\n== summary (same user, same 12 sentences) ==")
    for name, acc in results.items():
        print(f"   ours, {name:<24} {acc:.1%}  (first pass, no context LM)")
    print(f"   QuickPath (full stack, context on)   {tot_c / tot_n:.1%}")


if __name__ == "__main__":
    main()
