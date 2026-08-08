#!/usr/bin/env python3
"""End-to-end: first pass, acoustic rescorer, and context LM together.

The two second-pass signals target different errors -- the rescorer re-scores
gesture evidence, the LM adds context -- so the question is whether they stack
or overlap. This sweeps both weights over one set of dumped n-best lists and
reports each in isolation and combined.

Usage:
    python scripts/dump_nbest.py --split futo/validation --limit 20000
    python scripts/eval_full_stack.py --rescorer runs/rescorer/rescorer.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model.rescore import RescoreConfig, Rescorer  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import pick_device  # noqa: E402
from eval_neural_rerank import NeuralLM  # noqa: E402
from train_rescorer import NBestSet  # noqa: E402


@torch.no_grad()
def rescorer_logits(model, dataset, device, batch: int = 256) -> np.ndarray:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2)
    out = []
    for feats, ids, pos, mask, scores, _, valid in loader:
        logits = model(feats.to(device), ids.to(device), pos.to(device),
                       mask.to(device), scores.to(device))
        out.append(logits.masked_fill(~valid.to(device), -1e4).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", default="data/nbest/futo_validation.npz")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--split", default="futo/validation")
    ap.add_argument("--rescorer", default="runs/rescorer/rescorer.pt")
    ap.add_argument("--lm", default="gpt2")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--rescorer-weights", nargs="+", type=float,
                    default=[0.0, 0.5, 1.0])
    ap.add_argument("--lm-weights", nargs="+", type=float,
                    default=[0.0, 0.5, 0.8, 1.2])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    kb = KeyboardLayout.qwerty()
    ds = NBestSet(Path(args.nbest), kb)
    n = len(ds)
    first_pass = ds.scores.copy()
    target = ds.target
    valid = ds.valid
    print(f"{args.split}: n={n:,}  first-pass top-1 {ds.baseline:.4f}  "
          f"ceiling {ds.ceiling:.4f}  headroom {ds.ceiling - ds.baseline:.4f}\n")

    ckpt = torch.load(args.rescorer, map_location="cpu", weights_only=False)
    cfg_d = dict(ckpt["cfg"])
    cfg_d["dilations"] = tuple(cfg_d["dilations"])
    model = Rescorer(RescoreConfig(**cfg_d))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    combined = rescorer_logits(model, ds, device)
    # The model returns scale*first_pass + residual; recover the residual so the
    # two second-pass signals can be weighted independently.
    scale = float(model.first_pass_scale.detach().cpu())
    residual = combined - scale * first_pass
    print(f"rescorer loaded ({model.num_parameters():,} params, "
          f"first_pass_scale {scale:.3f})")

    # Context LM over the same candidate lists.
    from swipe_typing.model import SwipeCorpus
    from eval_reranker import order_by_sentence

    corpus = SwipeCorpus.load(Path(args.cache) / args.split,
                              "abcdefghijklmnopqrstuvwxyz", limit=args.limit)
    groups = order_by_sentence(corpus)
    cands = np.load(args.nbest, allow_pickle=False)["candidates"]
    lm = NeuralLM(args.lm, device)
    print(f"scoring candidates with {args.lm} ...")
    lm_delta = np.zeros_like(first_pass)
    for idx in groups:
        raw = corpus.sentences[idx[0]].split() if idx else []
        for pos_i, i in enumerate(idx):
            words = [str(w) for w, ok in zip(cands[i], valid[i]) if ok]
            if not words:
                continue
            k = int(corpus.word_idx[i]) if corpus.word_idx[i] >= 0 else pos_i
            cond = lm.score(" ".join(raw[:k]), words)
            uncond = lm.unconditional(words)
            for j, d in enumerate(c - u for c, u in zip(cond, uncond)):
                lm_delta[i, j] = d

    def top1(rw: float, lw: float) -> float:
        logits = first_pass + rw * residual + lw * lm_delta
        logits = np.where(valid, logits, -1e9)
        return float((logits.argmax(1) == target)[target >= 0].sum()
                     + 0) / n

    print(f"\n  {'':>10}" + "".join(f"lm={w:<7.2f}" for w in args.lm_weights))
    for rw in args.rescorer_weights:
        row = f"  rs={rw:<7.2f}"
        for lw in args.lm_weights:
            row += f"{top1(rw, lw):<10.4f}"
        print(row)

    best = max(((top1(rw, lw), rw, lw)
                for rw in args.rescorer_weights for lw in args.lm_weights))
    print(f"\n  best {best[0]:.4f} at rescorer={best[1]} lm={best[2]}"
          f"   ({best[0] - ds.baseline:+.4f} over first pass, "
          f"{100 * (best[0] - ds.baseline) / (ds.ceiling - ds.baseline):.0f}% "
          f"of headroom)")


if __name__ == "__main__":
    main()
