#!/usr/bin/env python3
"""Export a GPT-2-family LM for the keyboard's fused sentence search.

    ../research/.venv/bin/python tools/export_lm.py --model distilgpt2 --quant int8

Core ML model `SwipeLM.mlpackage`, fixed shapes:
  ids  (B, L) int32   token ids, right-padded with EOS
  pos  (B, P) int32   positions whose next-token distribution is read
  tgt  (B, P) int32   the token to read at each of those positions
  ->   logp (B, P)    log P(tgt | ids[:pos]) — sum over a word's tokens outside

Only the P read positions are projected onto the 50k vocabulary, so the call
moves a few hundred floats instead of B·L·50257. Also writes `lm_goldens.json`:
(ctx, word) -> log P(word | ctx) from the HF fp32 model, for the Swift tests,
and `lm_meta.json` with the shapes.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
KEYBOARD = HERE.parent
RESEARCH = KEYBOARD.parent / "research"
sys.path.insert(0, str(RESEARCH / "src"))
OUT = KEYBOARD / "Resources"


class Block(nn.Module):
    def __init__(s, d, h, L):
        super().__init__(); s.h = h; s.d = d; s.L = L
        s.ln1 = nn.LayerNorm(d); s.attn = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.ln2 = nn.LayerNorm(d); s.fc = nn.Linear(d, 4 * d); s.out = nn.Linear(4 * d, d)

    def forward(s, x, mask, B):
        D, dh, L = s.d, s.d // s.h, s.L
        q, k, v = s.attn(s.ln1(x)).split(D, dim=2)
        q = q.reshape(B, L, s.h, dh).transpose(1, 2)
        k = k.reshape(B, L, s.h, dh).transpose(1, 2)
        v = v.reshape(B, L, s.h, dh).transpose(1, 2)
        a = F.softmax((q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(dh)) + mask, dim=-1)
        y = (a @ v).transpose(1, 2).reshape(B, L, D)
        x = x + s.proj(y)
        return x + s.out(F.gelu(s.fc(s.ln2(x)), approximate="tanh"))


class GPT2Gather(nn.Module):
    """GPT-2 that returns log P(tgt) at the requested positions only."""

    def __init__(s, hf, B, L, P):
        super().__init__()
        c = hf.config; d = c.n_embd
        s.B, s.L, s.P = B, L, P
        s.wte = nn.Embedding(c.vocab_size, d); s.wpe = nn.Embedding(c.n_positions, d)
        s.blocks = nn.ModuleList(Block(d, c.n_head, L) for _ in range(c.n_layer)); s.lnf = nn.LayerNorm(d)
        sd = hf.state_dict()
        s.wte.weight.data = sd["transformer.wte.weight"]; s.wpe.weight.data = sd["transformer.wpe.weight"]
        for i, b in enumerate(s.blocks):
            p = f"transformer.h.{i}."
            b.ln1.weight.data = sd[p + "ln_1.weight"]; b.ln1.bias.data = sd[p + "ln_1.bias"]
            b.attn.weight.data = sd[p + "attn.c_attn.weight"].t().contiguous(); b.attn.bias.data = sd[p + "attn.c_attn.bias"]
            b.proj.weight.data = sd[p + "attn.c_proj.weight"].t().contiguous(); b.proj.bias.data = sd[p + "attn.c_proj.bias"]
            b.ln2.weight.data = sd[p + "ln_2.weight"]; b.ln2.bias.data = sd[p + "ln_2.bias"]
            b.fc.weight.data = sd[p + "mlp.c_fc.weight"].t().contiguous(); b.fc.bias.data = sd[p + "mlp.c_fc.bias"]
            b.out.weight.data = sd[p + "mlp.c_proj.weight"].t().contiguous(); b.out.bias.data = sd[p + "mlp.c_proj.bias"]
        s.lnf.weight.data = sd["transformer.ln_f.weight"]; s.lnf.bias.data = sd["transformer.ln_f.bias"]
        s.register_buffer("mask", torch.triu(torch.full((L, L), -1e4), 1)[None, None])
        s.register_buffer("posids", torch.arange(L)[None])
        s.register_buffer("rows", torch.arange(B)[:, None].expand(B, P).reshape(-1))

    def forward(s, ids, pos, tgt):
        x = s.wte(ids) + s.wpe(s.posids)
        for b in s.blocks:
            x = b(x, s.mask, s.B)
        x = s.lnf(x)                                          # (B, L, d)
        h = x[s.rows, pos.reshape(-1)]                        # (B*P, d)
        logits = h @ s.wte.weight.t()                         # (B*P, V)
        lp = F.log_softmax(logits, dim=-1)
        picked = lp.gather(1, tgt.reshape(-1, 1)).reshape(s.B, s.P)
        return picked


def score_pairs_hf(hf, tok, pairs):
    """Exactly fused_rescore.LMScorer: [bos] + ctx tokens, word with a leading space iff ctx."""
    out = []
    with torch.no_grad():
        for ctx, word in pairs:
            ids = [tok.bos_token_id] + (tok.encode(ctx) if ctx else [])
            cont = tok.encode((" " if ctx else "") + word)
            seq = torch.tensor([ids + cont])
            lp = F.log_softmax(hf(input_ids=seq).logits.float(), dim=-1)[0]
            out.append(sum(lp[j - 1, seq[0, j]].item() for j in range(len(ids), len(ids) + len(cont))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--quant", default="int8",
                    choices=["fp16", "int8", "int6", "int4", "int6pc", "int4pc", "int8pc"],
                    help="int8 = linear per-channel; intN = k-means palettized, grouped LUT (32); "
                         "intNpc = k-means palettized, one LUT per output channel (no LUT overhead)")
    ap.add_argument("--no-goldens", action="store_true")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--out", default="SwipeLM.mlpackage")
    a = ap.parse_args()
    import coremltools as ct
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    m = GPT2Gather(hf, a.B, a.L, a.P).eval()

    ids = torch.full((a.B, a.L), tok.eos_token_id, dtype=torch.long)
    pos = torch.zeros(a.B, a.P, dtype=torch.long); tgt = torch.zeros(a.B, a.P, dtype=torch.long)
    tr = torch.jit.trace(m, (ids, pos, tgt))
    ml = ct.convert(
        tr,
        inputs=[ct.TensorType(name="ids", shape=(a.B, a.L), dtype=np.int32),
                ct.TensorType(name="pos", shape=(a.B, a.P), dtype=np.int32),
                ct.TensorType(name="tgt", shape=(a.B, a.P), dtype=np.int32)],
        outputs=[ct.TensorType(name="logp", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL)
    if a.quant != "fp16":
        from coremltools.optimize.coreml import (OpLinearQuantizerConfig, OpPalettizerConfig,
                                                 OptimizationConfig, linear_quantize_weights, palettize_weights)
        if a.quant == "int8":
            ml = linear_quantize_weights(ml, OptimizationConfig(global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_channel")))
        elif a.quant.endswith("pc"):
            bits = int(a.quant[3:-2])
            ml = palettize_weights(ml, OptimizationConfig(global_config=OpPalettizerConfig(
                mode="kmeans", nbits=bits, granularity="per_tensor")))
        else:
            bits = int(a.quant[3:])
            ml = palettize_weights(ml, OptimizationConfig(global_config=OpPalettizerConfig(
                mode="kmeans", nbits=bits, granularity="per_grouped_channel", group_size=32)))
    ml.short_description = f"{a.model} {a.quant}: log P(tgt | ids[:pos]) at P positions, B={a.B} L={a.L}"
    path = OUT / a.out
    ml.save(str(path))
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e6

    # goldens: real capture words in decoded contexts, plus the marginal contexts
    import json as _json
    cap = sorted((RESEARCH / "iphone/data").glob("capture_*.json"))
    sents = []
    for f in cap[:40]:
        s = " ".join("".join(ch for ch in g["word"].lower() if ch.isalpha()) for g in _json.loads(f.read_text())["gestures"])
        if s and s not in sents: sents.append(s)
    pairs = []
    for s in sents[:12]:
        ws = s.split()
        for i, w in enumerate(ws):
            pairs.append((" ".join(ws[:i]), w))
    for w in ["the", "to", "too", "top", "ngl", "priya", "downstairs", "boba"]:
        for c in ["", "i think", "and then", "she said", "it was", "we can", "they will", "he did"]:
            pairs.append((c, w))
    pairs = list(dict.fromkeys(pairs))
    ref = score_pairs_hf(hf, tok, pairs)

    # run the Core ML model on the same pairs the way Swift will
    def coreml_scores(pairs):
        out = [None] * len(pairs)
        for start in range(0, len(pairs), a.B):
            chunk = pairs[start:start + a.B]
            ids_np = np.full((a.B, a.L), tok.eos_token_id, np.int32)
            pos_np = np.zeros((a.B, a.P), np.int32); tgt_np = np.zeros((a.B, a.P), np.int32)
            spans = []
            for i, (ctx, word) in enumerate(chunk):
                c = [tok.bos_token_id] + (tok.encode(ctx) if ctx else [])
                cont = tok.encode((" " if ctx else "") + word)
                c = c[-(a.L - len(cont)):] if len(c) + len(cont) > a.L else c
                seq = c + cont
                ids_np[i, :len(seq)] = seq
                n = min(len(cont), a.P)
                for j in range(n):
                    pos_np[i, j] = len(c) - 1 + j; tgt_np[i, j] = cont[j]
                spans.append(n)
            lp = ml.predict({"ids": ids_np, "pos": pos_np, "tgt": tgt_np})["logp"]
            for i, n in enumerate(spans):
                out[start + i] = float(lp[i, :n].sum())
        return out
    t0 = time.time(); got = coreml_scores(pairs); dt = (time.time() - t0) / math.ceil(len(pairs) / a.B) * 1000
    err = np.abs(np.array(got) - np.array(ref))
    print(f"{a.model} {a.quant}: {size:.0f} MB, {dt:.0f} ms per batch of {a.B} (mac), "
          f"|dlogp| mean {err.mean():.3f} p95 {np.percentile(err, 95):.3f} max {err.max():.3f} over {len(pairs)} pairs")
    if a.no_goldens:
        return
    (OUT / "lm_goldens.json").write_text(json.dumps({
        "model": a.model, "quant": a.quant, "B": a.B, "L": a.L, "P": a.P,
        "pairs": [{"ctx": c, "word": w, "logp": r, "coreml": g} for (c, w), r, g in zip(pairs, ref, got)]}))
    (OUT / "lm_meta.json").write_text(json.dumps({"model": a.model, "quant": a.quant, "B": a.B, "L": a.L, "P": a.P,
                                                  "eos": tok.eos_token_id, "bos": tok.bos_token_id}))
    print("wrote", path, "lm_goldens.json lm_meta.json")


if __name__ == "__main__":
    main()
