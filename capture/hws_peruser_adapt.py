#!/usr/bin/env python3
"""Does per-user adaptation help ALL users? Today's recipe, run per-user
on How We Swipe (real users, real per-user volumes, foreign apparatus).

Per user (>=300 swipes in hws test): fine-tune the canonical encoder on
their chronologically-first 200 swipes + 1500 FUTO replay (the capture
study's exact recipe: LR 1e-4, 8 epochs, augmentation on), then beam
top-1 on their remaining swipes (cap 150), before vs after. The output
is the per-user delta distribution — mean, spread, and the worst case.

Usage:  .venv/bin/python capture/hws_peruser_adapt.py --users 20
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import ConcatDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT.parent / "scripts"))

from swipe_typing.layout import ALPHABET, KeyboardLayout            # noqa: E402
from swipe_typing.model import (SwipeCorpus, SwipeDataset,          # noqa: E402
                                ctc_loss, make_loader)
from swipe_typing.model.beam import BeamConfig, beam_search         # noqa: E402
from eval_decoder import (build_lexicon, load_model, pick_device,   # noqa: E402
                          run_encoder)

TRAIN_N, EVAL_N, EPOCHS, LR = 200, 150, 8, 1e-4


def subset(corpus: SwipeCorpus, idx: list[int]) -> SwipeCorpus:
    xs, ys, ts, offsets, words, aspects = [], [], [], [0], [], []
    for i in idx:
        a, b = corpus.offsets[i], corpus.offsets[i + 1]
        xs.append(corpus.x[a:b]); ys.append(corpus.y[a:b])
        ts.append(corpus.t[a:b])
        offsets.append(offsets[-1] + (b - a))
        words.append(corpus.words[i])
        aspects.append(corpus.aspects[i])
    return SwipeCorpus(np.concatenate(xs), np.concatenate(ys),
                       np.concatenate(ts),
                       np.asarray(offsets, dtype=np.int64),
                       words, np.asarray(aspects, dtype=np.float32))


def beam_top1(model, corpus, kb, lexicon, device, alphabet, key_units, mode):
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units, shape_only=model.cfg.shape_only)
    loader = make_loader(ds, batch_size=128, shuffle=False, num_workers=0)
    lp, refs = run_encoder(model, loader, device, alphabet)
    cfg = BeamConfig(beam_width=64, alpha=0.8, beta=1.2, top_k=1)
    hit = 0
    for i, item in enumerate(lp):
        hyps = beam_search(item, lexicon, alphabet, cfg)
        hit += bool(hyps) and hyps[0][0] == refs[i]
    return hit / len(refs), len(refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--replay", type=int, default=1500)
    args = ap.parse_args()

    device = pick_device("auto")
    kb = KeyboardLayout.qwerty()
    base_path = str(ROOT.parent / "runs/full/encoder.pt")
    model, alphabet, key_units, mode = load_model(base_path, device)
    lexicon = build_lexicon("train+wf320k", ROOT.parent / "data/canonical",
                            alphabet, 1.0)
    hws = SwipeCorpus.load(ROOT.parent / "data/canonical/how_we_swipe/test",
                           alphabet)
    replay = SwipeCorpus.load(ROOT.parent / "data/canonical/futo/train",
                              alphabet, limit=args.replay)
    rep_ds = SwipeDataset(replay, kb, resample_mode=mode, key_units=key_units)

    by_user: dict[str, list[int]] = {}
    for i, s in enumerate(hws.sessions):
        by_user.setdefault(s, []).append(i)
    eligible = [u for u, idx in by_user.items()
                if len(idx) >= TRAIN_N + 100]
    eligible.sort(key=lambda u: -len(by_user[u]))
    users = eligible[:args.users]
    print(f"{len(eligible)} eligible users; running {len(users)}", flush=True)

    deltas = []
    for k, u in enumerate(users):
        idx = by_user[u]
        tr = subset(hws, idx[:TRAIN_N])
        ev = subset(hws, idx[TRAIN_N:TRAIN_N + EVAL_N])
        base_acc, n = beam_top1(model, ev, kb, lexicon, device,
                                alphabet, key_units, mode)
        ft = copy.deepcopy(model).to(device).train()
        user_ds = SwipeDataset(tr, kb, resample_mode=mode, key_units=key_units)
        loader = make_loader(ConcatDataset([user_ds, rep_ds]), batch_size=32,
                             shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(ft.parameters(), lr=LR, weight_decay=1e-4)
        for _ in range(EPOCHS):
            for x, targets, lengths in loader:
                opt.zero_grad()
                loss = ctc_loss(ft(x.to(device)), targets.to(device),
                                lengths.to(device), ft.cfg.blank)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ft.parameters(), 1.0)
                opt.step()
        ft.eval()
        ft_acc, _ = beam_top1(ft, ev, kb, lexicon, device,
                              alphabet, key_units, mode)
        deltas.append(ft_acc - base_acc)
        print(f"[{k + 1}/{len(users)}] user {u}: base {base_acc:.3f} -> "
              f"ft {ft_acc:.3f}  (delta {ft_acc - base_acc:+.3f}, n={n})",
              flush=True)

    d = np.array(deltas)
    print(f"\nper-user adaptation deltas over {len(d)} users:")
    print(f"  mean {d.mean():+.3f}  median {np.median(d):+.3f}  "
          f"sd {d.std():.3f}")
    print(f"  improved {(d > 0).sum()}/{len(d)}  "
          f"hurt {(d < 0).sum()}/{len(d)}  worst {d.min():+.3f}  "
          f"best {d.max():+.3f}")


if __name__ == "__main__":
    main()
