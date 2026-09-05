#!/usr/bin/env python3
"""The AR decoder's internal LM per lexicon word (research #78, "mean" ablation):
log P_ar(word | mean memory), scored with the decoder's memory replaced by the average
memory over real gestures — the prior the encoder carries about words regardless of
the gesture. The first pass subtracts lambda·ilm(w) (lambda 0.25).

    ../research/.venv/bin/python tools/export_ilm.py --checkpoint runs/ar_mixed_s1/ar_decoder.pt

Writes Resources/ilm.bin: "SWIL", u32 version, u32 N, float32[N] in lexicon.bin node order (NaN for non-words).
"""
from __future__ import annotations
import argparse, struct, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; KEYBOARD = HERE.parent; RESEARCH = KEYBOARD.parent / "research"
sys.path.insert(0, str(RESEARCH / "src")); sys.path.insert(0, str(RESEARCH / "scripts")); sys.path.insert(0, str(RESEARCH / "iphone"))
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeDataset  # noqa: E402
from swipe_typing.model.data import SwipeCorpus  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from probe_ilm_fusion import ilm_scores  # noqa: E402
from export_priors import bfs_nodes  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default="runs/ar_mixed_s1/ar_decoder.pt"); ap.add_argument("--mean-swipes", type=int, default=2000)
    a = ap.parse_args()
    device = torch.device("cpu"); kb = KeyboardLayout.qwerty()
    m, alphabet, mode = load_ar(str(RESEARCH / a.checkpoint), device)
    lex = build_lexicon("train+wf320k", RESEARCH / "data/canonical", ALPHABET, 1.0)
    nodes = bfs_nodes(lex); index = {id(n): i for i, n in enumerate(nodes)}
    words = [None] * len(nodes)
    def walk(node, prefix):
        if node.is_word: words[index[id(node)]] = prefix
        for ch, child in node.children.items(): walk(child, prefix + ch)
    walk(lex.root, "")
    corpus = SwipeCorpus.load(RESEARCH / "data/canonical/futo/validation", alphabet, limit=a.mean_swipes)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode, key_units=True)
    with torch.no_grad():
        feats = torch.cat([x[None] for x, _ in ds]); mean_mem = m.encode(feats).mean(0, keepdim=True)
    t0 = time.time()
    ids = [i for i, w in enumerate(words) if w and len(w) <= m.cfg.max_word_len]   # longer words cannot be decoded anyway
    table = ilm_scores(m, [words[i] for i in ids], alphabet, device, mean_mem, batch=1024)
    out = np.full(len(nodes), np.nan, np.float32)
    for i in ids: out[i] = table[words[i]]
    path = KEYBOARD / "Resources/ilm.bin"
    with open(path, "wb") as f:
        f.write(b"SWIL"); f.write(struct.pack("<II", 1, len(nodes))); f.write(out.astype("<f4").tobytes())
    print(f"ilm -> {path} ({len(ids):,} words, {time.time() - t0:.0f}s); the: {table['the']:.2f}  priya: {table.get('priya', float('nan')):.2f}")

if __name__ == "__main__":
    main()
