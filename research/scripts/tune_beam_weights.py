#!/usr/bin/env python3
"""Re-tune the beam's alpha/beta for the current decoder configuration.

They were tuned once, at beam 16 / prune -9 / a 65k training-only lexicon. All
three have since changed (beam 64, prune -13, a 320k blended lexicon), and every
measurement after that has been sitting on hyperparameters fitted to a different
system. This re-fits them before anything else is measured.

Cheap because alpha and beta only rank the *finished* beams -- they never affect
which beams survive, so one search pass supports the whole grid.

Usage:
    python scripts/tune_beam_weights.py --limit 20000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_candidates  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    ap.add_argument("--betas", nargs="+", type=float,
                    default=[0.0, 0.6, 1.2, 1.8, 2.4, 3.0])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)

    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units, shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=512, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for x, _, _ in loader:
            chunks.append(model(x.to(device)).float().cpu().numpy())
    log_probs = np.concatenate(chunks)
    refs = corpus.words
    n = len(refs)

    cfg = BeamConfig(beam_width=args.beam_width)
    print(f"{args.split}: n={n:,}  searching once ...")
    raw = [beam_candidates(item, lexicon, alphabet, cfg) for item in log_probs]

    reachable = sum(any(w == refs[i] for w, _, _, _ in c)
                    for i, c in enumerate(raw)) / n
    print(f"  true word among surviving beams: {reachable:.4f} "
          f"(ceiling for any alpha/beta)\n")

    def evaluate(alpha: float, beta: float):
        top1 = topk = 0
        for i, cands in enumerate(raw):
            if not cands:
                continue
            ranked = sorted(cands, key=lambda c: c[1] + alpha * c[2] + beta * c[3],
                            reverse=True)[:args.top_k]
            if ranked[0][0] == refs[i]:
                top1 += 1
            if any(w == refs[i] for w, _, _, _ in ranked):
                topk += 1
        return top1 / n, topk / n

    print("  top-1 by (alpha, beta):")
    print(f"  {'a\\\\b':>6}" + "".join(f"{b:>9.1f}" for b in args.betas))
    best = (0.0, None, None)
    for a in args.alphas:
        row = f"  {a:>6.1f}"
        for b in args.betas:
            t1, _ = evaluate(a, b)
            if t1 > best[0]:
                best = (t1, a, b)
            row += f"{t1:>9.4f}"
        print(row)

    t1, a, b = best
    cur1, curk = evaluate(0.8, 1.2)
    _, bestk = evaluate(a, b)
    print(f"\n  current (alpha=0.8, beta=1.2): top-1 {cur1:.4f}  "
          f"top-{args.top_k} {curk:.4f}")
    print(f"  best    (alpha={a}, beta={b}): top-1 {t1:.4f}  "
          f"top-{args.top_k} {bestk:.4f}")
    print(f"  delta: top-1 {t1 - cur1:+.4f}   top-{args.top_k} {bestk - curk:+.4f}")


if __name__ == "__main__":
    main()
