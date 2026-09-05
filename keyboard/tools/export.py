#!/usr/bin/env python3
"""Export the on-device decoder bundle for the iOS keyboard.

Writes into keyboard/Resources/:
  SwipeEncoder.mlpackage   the CTC encoder (runs/full by default), fp32, input (1,64,32)
  lexicon.bin              the train+wf320k trie, flattened, children contiguous
  goldens.json             raw gestures -> features -> log_probs -> beam candidates,
                           computed by the research code, for the Swift tests

Usage:  ../research/.venv/bin/python tools/export.py [--checkpoint runs/full/encoder.pt]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
KEYBOARD = HERE.parent
RESEARCH = KEYBOARD.parent / "research"
sys.path.insert(0, str(RESEARCH / "src"))
sys.path.insert(0, str(RESEARCH / "scripts"))

from swipe_typing import features  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_candidates  # noqa: E402
from swipe_typing.model.encoder import EncoderConfig, SwipeEncoder  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402

OUT = KEYBOARD / "Resources"


def load(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]); cfg["dilations"] = tuple(cfg["dilations"])
    m = SwipeEncoder(EncoderConfig(**cfg)); m.load_state_dict(ck["model"]); m.eval()
    a = ck.get("args", {}) or {}
    assert not a.get("no_key_units", False), "exporter assumes key-unit kinematics"
    assert a.get("resample_mode", "time") == "time"
    assert ck.get("alphabet", ALPHABET) == ALPHABET
    return m


def export_model(m: SwipeEncoder, out: Path) -> None:
    import coremltools as ct
    x = torch.zeros(1, features.N_POINTS, m.cfg.n_input)
    tr = torch.jit.trace(m, x)
    ml = ct.convert(
        tr,
        inputs=[ct.TensorType(name="features", shape=tuple(x.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name="log_probs", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    ml.short_description = "swipe CTC encoder: (1,64,26 affinity + 6 kinematics) -> (1,64,27) log-probs, blank=26"
    ml.save(str(out))
    print(f"model -> {out}")


def featurize(pts: np.ndarray, t: np.ndarray, kb: KeyboardLayout) -> np.ndarray:
    """Exactly SwipeDataset.__getitem__ with augment_cfg=None, key_units=True, mode=time."""
    scale = features.key_scale(kb.radii)
    resampled = features.resample(pts, t, n=features.N_POINTS, mode="time")
    aff = features.key_affinity(resampled, kb.centers, kb.radii)
    kin = features.kinematics(pts, t, 0.0, n=features.N_POINTS, mode="time", scale=scale)
    return np.concatenate([aff, kin], axis=1).astype(np.float32)


def export_lexicon(lex, out: Path) -> int:
    """Flatten the trie BFS so every node's children are contiguous and letter-sorted.

    Layout (little-endian):  magic 'SWTR', u32 version=1, u32 N, then
      letter[N] u8, flags[N] u8 (bit0 = is_word), child_start[N] i32,
      child_count[N] u8, logp[N] f32, parent[N] i32 (root: -1).
    Node 0 is the root; a prefix is identified by its node id.
    """
    nodes = [lex.root]
    for node in nodes:  # BFS; list grows while iterating
        for ch in sorted(node.children):
            nodes.append(node.children[ch])
    n = len(nodes)
    index = {id(nd): i for i, nd in enumerate(nodes)}
    letter = np.zeros(n, np.uint8); flags = np.zeros(n, np.uint8)
    start = np.full(n, -1, np.int32); count = np.zeros(n, np.uint8)
    logp = np.zeros(n, np.float32); parent = np.full(n, -1, np.int32)
    for i, nd in enumerate(nodes):
        flags[i] = 1 if nd.is_word else 0
        logp[i] = nd.logp if nd.is_word else 0.0
        kids = sorted(nd.children)
        if kids:
            first = index[id(nd.children[kids[0]])]
            start[i] = first; count[i] = len(kids)
            for j, ch in enumerate(kids):
                assert index[id(nd.children[ch])] == first + j
                letter[first + j] = ord(ch)
                parent[first + j] = i
    with open(out, "wb") as f:
        f.write(b"SWTR"); f.write(struct.pack("<II", 2, n))
        f.write(letter.tobytes()); f.write(flags.tobytes())
        f.write(start.astype("<i4").tobytes()); f.write(count.tobytes())
        f.write(logp.astype("<f4").tobytes())
        f.write(parent.astype("<i4").tobytes())
    print(f"lexicon -> {out}  ({n:,} nodes, {len(lex):,} words, {out.stat().st_size/1e6:.1f} MB)")
    return n


def export_goldens(m, lex, kb, out: Path, n_gestures: int) -> None:
    data = sorted((RESEARCH / "iphone/data").glob("capture_*.json"))
    gestures = []
    for f in data:
        p = json.loads(f.read_text())
        for g in p["gestures"]:
            w = "".join(c for c in g["word"].lower() if c.isalpha())
            if w and all(c in ALPHABET for c in w) and len(g["x"]) >= 3:
                gestures.append((w, g))
    rng = np.random.default_rng(0)
    pick = [gestures[i] for i in rng.choice(len(gestures), n_gestures, replace=False)]
    # also a synthetic near-tap and a 2-point gesture, for the degenerate paths
    pick.append(("a", {"x": [0.1, 0.101, 0.1], "y": [0.5, 0.5, 0.501], "t": [0, 40, 80]}))
    pick.append(("to", {"x": [0.45, 0.85], "y": [0.17, 0.17], "t": [0, 300]}))
    cfg = BeamConfig(beam_width=64, alpha=0.8, beta=1.2, top_k=8)
    items = []
    for w, g in pick:
        pts = np.stack([np.asarray(g["x"], np.float32), np.asarray(g["y"], np.float32)], 1)
        t = np.asarray(g["t"], np.int32)
        x = featurize(pts, t, kb)
        with torch.no_grad():
            lp = m(torch.from_numpy(x)[None])[0].numpy()
        cands = beam_candidates(lp, lex, ALPHABET, cfg)
        ranked = sorted(((wd, a + cfg.alpha * u + cfg.beta * n, a, u, n) for wd, a, u, n in cands),
                        key=lambda c: c[1], reverse=True)[:8]
        items.append({
            "word": w, "x": g["x"], "y": g["y"], "t": g["t"],
            "features": x.tolist(), "log_probs": lp.tolist(),
            "candidates": [{"word": c[0], "score": c[1], "acoustic": c[2],
                            "unigram": c[3], "length": c[4]} for c in ranked],
        })
    out.write_text(json.dumps({"alphabet": ALPHABET, "beam": {"width": 64, "prune": -13.0,
                               "alpha": 0.8, "beta": 1.2}, "items": items}))
    top1 = sum(1 for it in items[:-2] if it["candidates"] and it["candidates"][0]["word"] == it["word"])
    print(f"goldens -> {out}  ({len(items)} gestures; python top-1 on the real ones {top1}/{len(items)-2})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--goldens", type=int, default=24)
    ap.add_argument("--skip-lexicon", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    m = load(RESEARCH / args.checkpoint)
    kb = KeyboardLayout.qwerty()
    export_model(m, OUT / "SwipeEncoder.mlpackage")
    lex = build_lexicon("train+wf320k", RESEARCH / "data/canonical", ALPHABET, 1.0)
    if not args.skip_lexicon:
        export_lexicon(lex, OUT / "lexicon.bin")
    export_goldens(m, lex, kb, OUT / "goldens.json", args.goldens)
    # geometry constants the Swift side hard-codes; assert they still hold
    assert np.allclose(features.key_scale(kb.radii), [0.1, 1 / 3], atol=1e-6)
    assert np.allclose(kb.center("a"), [0.1, 0.5], atol=1e-6)
    print("geometry constants ok")


if __name__ == "__main__":
    main()
