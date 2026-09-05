#!/usr/bin/env python3
"""Export the AR swipe decoder for the keyboard.

    ../research/.venv/bin/python tools/export_ar.py --checkpoint runs/ar_mixed_s1/ar_decoder.pt

Two Core ML models (fp32, CPU):
  SwipeAREncoder.mlpackage   features (1, 64, 32) -> memory (1, 64, 128)
  SwipeARStep.mlpackage      memory (K, 64, 128) + tokens (K, L) int32 -> log-probs (K, 28)
                             of the NEXT token after position L-1; L is an enumerated
                             shape 1..max_word_len+1, K fixed at the beam width
plus ar_meta.json and ar_goldens.json (capture gestures -> memory checksum and the
Python `ar_beam` candidate lists with (word, ar_logp, unigram, len)), for the Swift
port's tests. The step model reimplements nn.TransformerDecoder (norm_first, relu,
2 layers) with explicit matmuls so it traces cleanly; verified against the torch
module before conversion.
"""
from __future__ import annotations

import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
KEYBOARD = HERE.parent
RESEARCH = KEYBOARD.parent / "research"
sys.path.insert(0, str(RESEARCH / "src")); sys.path.insert(0, str(RESEARCH / "scripts")); sys.path.insert(0, str(RESEARCH / "iphone"))
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeDataset  # noqa: E402
from swipe_typing.model.ar import FlatTrie, ar_beam  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from decode_capture import build_corpus  # noqa: E402
OUT = KEYBOARD / "Resources"


class Encoder(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.encode(x)


class Step(nn.Module):
    """Explicit norm-first transformer decoder over a (K, L) prefix; returns log-probs at position L-1."""

    def __init__(s, m, K):
        super().__init__()
        c = m.cfg; s.K = K; s.d = c.d_model; s.h = c.dec_heads; s.dh = c.d_model // c.dec_heads
        s.tok_emb = m.tok_emb; s.tok_pos = m.tok_pos; s.head = m.head
        s.layers = nn.ModuleList()
        for layer in m.decoder.layers:
            s.layers.append(layer)

    def attn(s, q_in, kv_in, mha, causal):
        # No Python ints derived from shapes: the prefix length is a dynamic
        # (enumerated) dimension, so reshape with -1 and build masks with ones_like.
        d = s.d
        w, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(q_in, w[:d], b[:d]); k = F.linear(kv_in, w[d:2 * d], b[d:2 * d]); v = F.linear(kv_in, w[2 * d:], b[2 * d:])
        q = q.reshape(s.K, -1, s.h, s.dh).transpose(1, 2); k = k.reshape(s.K, -1, s.h, s.dh).transpose(1, 2); v = v.reshape(s.K, -1, s.h, s.dh).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(s.dh)
        if causal:
            a = a + torch.triu(torch.ones_like(a[0, 0]), 1) * -1e4
        a = F.softmax(a, dim=-1)
        y = (a @ v).transpose(1, 2).reshape(s.K, -1, d)
        return mha.out_proj(y)

    def forward(s, memory, tokens):
        pos = s.tok_pos[:, :tokens.shape[1]] if not torch.jit.is_tracing() else s.tok_pos.expand(s.K, -1, -1)[:, :tokens.shape[1]]
        x = s.tok_emb(tokens) + pos
        for ly in s.layers:
            x = x + s.attn(ly.norm1(x), ly.norm1(x), ly.self_attn, True)
            n2 = ly.norm2(x)
            x = x + s.attn(n2, memory, ly.multihead_attn, False)
            x = x + ly.linear2(F.relu(ly.linear1(ly.norm3(x))))
        logits = s.head(x[:, -1])
        return F.log_softmax(logits, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_mixed_s1/ar_decoder.pt")
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--goldens", type=int, default=24)
    a = ap.parse_args()
    import coremltools as ct
    device = torch.device("cpu")
    m, alphabet, mode = load_ar(str(RESEARCH / a.checkpoint), device)
    cfg = m.cfg; K = a.beam; Lmax = cfg.max_word_len + 1
    assert alphabet == ALPHABET and mode == "time"

    # ---- self-check of the explicit step against the torch module ----
    step = Step(m, K).eval()
    with torch.no_grad():
        x = torch.randn(K, 64, cfg.n_input); mem = m.encode(x)
        for L in (1, 5, Lmax):
            toks = torch.randint(0, cfg.n_keys, (K, L)); toks[:, 0] = cfg.bos
            ref = F.log_softmax(m.decode_step(mem, toks)[:, -1].float(), -1)
            got = step(mem, toks)
            d = float((ref - got).abs().max()); print(f"step L={L}: max |Δ| vs torch module {d:.2e}"); assert d < 1e-3

    # ---- encoder ----
    enc = Encoder(m).eval()
    x1 = torch.zeros(1, 64, cfg.n_input)
    ml_enc = ct.convert(torch.jit.trace(enc, x1), inputs=[ct.TensorType(name="features", shape=(1, 64, cfg.n_input), dtype=np.float32)],
                        outputs=[ct.TensorType(name="memory", dtype=np.float32)], minimum_deployment_target=ct.target.iOS17,
                        compute_precision=ct.precision.FLOAT32)
    ml_enc.save(str(OUT / "SwipeAREncoder.mlpackage"))
    # ---- step, enumerated prefix lengths ----
    mem0 = torch.zeros(K, 64, cfg.d_model); tok0 = torch.full((K, 3), cfg.bos, dtype=torch.long)
    traced = torch.jit.trace(step, (mem0, tok0))
    shapes = ct.EnumeratedShapes(shapes=[(K, L) for L in range(1, Lmax + 1)], default=(K, 3))
    ml_step = ct.convert(traced, inputs=[ct.TensorType(name="memory", shape=(K, 64, cfg.d_model), dtype=np.float32),
                                         ct.TensorType(name="tokens", shape=shapes, dtype=np.int32)],
                         outputs=[ct.TensorType(name="logp", dtype=np.float32)], minimum_deployment_target=ct.target.iOS17,
                         compute_precision=ct.precision.FLOAT32)
    ml_step.save(str(OUT / "SwipeARStep.mlpackage"))
    size = lambda p: sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) / 1e6
    print(f"encoder {size(OUT / 'SwipeAREncoder.mlpackage'):.1f} MB, step {size(OUT / 'SwipeARStep.mlpackage'):.1f} MB")

    # ---- goldens on capture gestures: memory + python ar_beam candidates ----
    lexicon = build_lexicon("train+wf320k", RESEARCH / "data/canonical", alphabet, 1.0)
    trie = FlatTrie(lexicon, alphabet)
    bench = [s for s in json.load(open(KEYBOARD / "Resources/bench_gestures.json"))["sentences"] if s["source"] == "capture"]
    rng = np.random.default_rng(0)
    gs = [(w, g) for s in bench for g, w in zip(s["gestures"], s["words"]) if len(g["x"]) >= 3]
    pick = [gs[i] for i in rng.choice(len(gs), a.goldens, replace=False)]
    caps = {"0": {"tag": "", "sentence": " ".join(w for w, _ in pick), "session": "g", "set": 1,
                  "gestures": [dict(g, word=w, word_idx=j, aspect=2.2) for j, (w, g) in enumerate(pick)]}}
    corpus, _ = build_corpus(caps, alphabet)
    ds = SwipeDataset(corpus, KeyboardLayout.qwerty(), augment_cfg=None, resample_mode=mode, key_units=True)
    feats = torch.cat([f[None] for f, _ in ds])
    with torch.no_grad():
        mem = m.encode(feats)
        cands = ar_beam(m, feats, trie, alphabet, beam_width=K)
        # Core ML encoder agreement
        got = np.stack([ml_enc.predict({"features": feats[i:i + 1].numpy()})["memory"][0] for i in range(len(feats))])
        print(f"coreml encoder vs torch: max |Δ| {np.abs(got - mem.numpy()).max():.2e}")
        # Core ML step agreement on a real prefix
        toks = torch.full((K, 4), cfg.eos, dtype=torch.long); toks[:, 0] = cfg.bos; toks[:, 1:] = torch.tensor([[alphabet.index(c) for c in "the"]])
        ref = step(mem[:1].expand(K, -1, -1).contiguous(), toks)
        gotp = ml_step.predict({"memory": mem[:1].expand(K, -1, -1).contiguous().numpy(), "tokens": toks.numpy().astype(np.int32)})["logp"]
        print(f"coreml step vs torch: max |Δ| {np.abs(gotp - ref.numpy()).max():.2e}")
    items = []
    for i, (w, g) in enumerate(pick):
        items.append({"word": w, "x": g["x"], "y": g["y"], "t": g["t"], "features": feats[i].tolist(),
                      "memory_checksum": float(mem[i].sum()),
                      "candidates": [{"word": cw, "ar": float(ar), "unigram": float(u), "length": int(n)} for cw, ar, u, n in cands[i]]})
    (OUT / "ar_goldens.json").write_text(json.dumps({"beam": K, "items": items}))
    (OUT / "ar_meta.json").write_text(json.dumps({"checkpoint": a.checkpoint, "beam": K, "max_word_len": cfg.max_word_len,
                                                  "vocab": cfg.vocab, "bos": cfg.bos, "eos": cfg.eos, "d_model": cfg.d_model}))
    top1 = sum(1 for it in items if it["candidates"] and it["candidates"][0]["word"] == it["word"])
    print(f"goldens: {len(items)} gestures, python ar top-1 (raw ar score) {top1}/{len(items)}; wrote ar_meta.json, ar_goldens.json")


if __name__ == "__main__":
    main()
