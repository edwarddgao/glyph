#!/usr/bin/env python3
"""Fast quality oracle for a synthetic gesture corpus.

    python scripts/gen_quality.py --corpus data/canonical/minjerk_rand/train
    python scripts/gen_quality.py --calibrate      # validate against known arms

The honest test of a generator is "train a decoder on nothing but its
gestures, measure on real data" — 90 minutes per candidate, which is too
slow to iterate a generator against. This is the cheap stand-in: a small
decoder on a slice of the corpus for a few epochs, greedy on real val. The
--calibrate mode ranks the five corpora whose full-scale numbers are already
known (#56/#57 and the learned arms), so the proxy is *validated* rather
than assumed.

Alongside it, statistics that need no training at all:

  geometry  path/prototype ratio, endpoint error, per-letter visit distance
            — whether the gesture reaches its letters and its ending
  texture   dwell fraction, jerk proxy — where uniform-time samples bunch,
            the cue #56/#57 showed decoder training actually feeds on
  shape     mean |turning angle| and its high-frequency share — real
            gestures spend their extra path length on smooth overshoot,
            not on abrupt direction changes
  C2ST      a small classifier's real-vs-synthetic AUC; 0.5 is
            indistinguishable, 1.0 is trivially separable

Calibrated once against the six corpora above: Spearman rho 0.94 vs
full-scale beam, 0.77 vs greedy, at 3 minutes per candidate instead of 90.

Beware of reading realism metrics as quality: #57 measured the two coming
apart (the least "readable" corpus trained the best decoder). Only the
proxy column is a utility claim; the rest are diagnostics for *why* a
corpus scores as it does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing import features  # noqa: E402
from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import (SwipeCorpus, SwipeDataset, decode,  # noqa: E402
                                make_loader)
from swipe_typing.model.ar import (ARConfig, ARSwipeDecoder, ar_loss,  # noqa: E402
                                   greedy_decode)
from swipe_typing.model.encoder import fit_normalization  # noqa: E402
from swipe_typing.model.gesturegen import prototype  # noqa: E402

#: full-scale synthetic-only results already on the log (val greedy, val beam)
KNOWN = {
    "minjerk": (0.085, 0.351),        # spline profile, #56
    "minjerk_seg": (0.382, 0.663),    # rest-to-rest segments, #56
    "minjerk_rand": (0.734, 0.857),   # domain-randomized, #57
    "gesturegen": (0.263, 0.508),     # learned v1, free-running regression
    "gesturegen_mdn": (0.280, 0.343),  # learned v2, MDN steps
    "gesturegen_warp": (0.552, 0.825),  # learned v3, prototype + time warp
    "gesturediff": (0.737, 0.854),    # diffusion, ties the analytic generator
    "gmix_full": (0.799, 0.885),      # diffusion + min-jerk mixture, #65
    "futo": (0.867, 0.924),           # real data, the top of the scale (#46)
}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resampled(corpus: SwipeCorpus, idx) -> np.ndarray:
    return np.stack([
        features.resample(corpus.points(i), corpus.times(i),
                          n=features.N_POINTS, mode="time") for i in idx])


def shape_stats(pts: np.ndarray, words: list[str], kb: KeyboardLayout) -> dict:
    step = np.diff(pts, axis=1)
    slen = np.linalg.norm(step, axis=-1)
    proto_len = np.array([
        np.linalg.norm(np.diff(prototype(w, kb), axis=0), axis=1).sum()
        for w in words])
    # Turning angle between consecutive steps; the high-frequency share is
    # the part that alternates sign (zig-zag) rather than curving.
    u = step / np.maximum(slen[..., None], 1e-9)
    cos = np.clip((u[:, 1:] * u[:, :-1]).sum(-1), -1.0, 1.0)
    turn = np.arccos(cos)
    # Signed turn (2-D cross product); alternating sign is zig-zag, constant
    # sign is a smooth curve — the difference the eye reads as "natural".
    cross = (u[:, :-1, 0] * u[:, 1:, 1] - u[:, :-1, 1] * u[:, 1:, 0])
    s = np.sign(cross)
    sign_flips = float(np.mean(s[:, 1:] != s[:, :-1]))
    return {
        "path_ratio": float((slen.sum(1) / proto_len).mean()),
        "end_err": float(np.mean([np.linalg.norm(p[-1] - kb.center(w[-1]))
                                  for p, w in zip(pts, words)])),
        "visit": float(np.mean([
            np.mean([np.linalg.norm(p - kb.center(c), axis=1).min()
                     for c in w]) for p, w in zip(pts, words)])),
        "dwell": float((slen < 0.25 * slen.mean(1, keepdims=True)).mean()),
        "jerk": float(np.abs(np.diff(slen, n=2, axis=1)).mean()),
        "turn": float(turn.mean()),
        "zigzag": sign_flips,
    }


def diversity(corpus: SwipeCorpus, kb, max_words: int = 300) -> float:
    """Mean within-word spread, as a fraction of a key width.

    A collapsed latent gives one gesture per word; real users give a cloud.
    Reported next to the real corpus's own value, so "too uniform" and "too
    scattered" are both visible — the former is posterior collapse (a bug),
    the latter is a prior that has stopped respecting the word.
    """
    from collections import defaultdict
    by_word = defaultdict(list)
    for i in range(len(corpus)):
        w = corpus.words[i]
        if len(by_word[w]) < 8:
            by_word[w].append(i)
    spreads = []
    for w, idx in by_word.items():
        if len(idx) < 2 or len(spreads) >= max_words:
            continue
        pts = resampled(corpus, idx)
        spreads.append(float(np.linalg.norm(pts - pts.mean(0), axis=-1).mean()))
    kw = float(features.key_scale(kb.radii).mean())
    return float(np.mean(spreads) / kw) if spreads else float("nan")


def c2st(real: np.ndarray, synth: np.ndarray, device, epochs: int = 30) -> float:
    """Classifier two-sample test AUC on per-gesture shape+speed features."""

    def feat(p):
        step = np.linalg.norm(np.diff(p, axis=1), axis=-1)
        c = p.mean(1, keepdims=True)
        s = np.maximum(np.abs(p - c).max(axis=(1, 2), keepdims=True), 1e-6)
        return np.concatenate([((p - c) / s).reshape(len(p), -1),
                               step / np.maximum(step.mean(1, keepdims=True),
                                                 1e-9)], axis=1)

    X = np.concatenate([feat(real), feat(synth)]).astype(np.float32)
    y = np.concatenate([np.zeros(len(real)), np.ones(len(synth))]
                       ).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    cut = len(X) // 2
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(),
        torch.nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(Xtr_t).squeeze(-1), ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = net(torch.from_numpy(Xte).to(device)).squeeze(-1).cpu().numpy()
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(len(s)) + 1
    npos, nneg = float(yte.sum()), float((1 - yte).sum())
    auc = (ranks[yte == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return float(max(auc, 1 - auc))


def proxy_utility(corpus_path: Path, kb, device, args) -> dict:
    """Small decoder, few epochs, greedy on real val — the cheap oracle."""
    train = SwipeCorpus.load(corpus_path, kb.letters, limit=args.proxy_train)
    val = SwipeCorpus.load(Path(args.cache) / "futo/validation", kb.letters,
                           limit=args.proxy_val)
    from swipe_typing.augment import DEFAULT as DEFAULT_AUG
    tl = make_loader(SwipeDataset(train, kb, augment_cfg=DEFAULT_AUG),
                     batch_size=256, num_workers=args.workers)
    vl = make_loader(SwipeDataset(val, kb, augment_cfg=None), batch_size=256,
                     shuffle=False, num_workers=2)
    cfg = ARConfig(n_keys=len(kb.letters), d_model=args.proxy_dim,
                   dilations=(1, 2, 4, 8), dec_layers=1)
    model = ARSwipeDecoder(cfg).to(device)
    fit_normalization(model, tl)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total = len(tl) * args.proxy_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100)
        * 0.5 * (1 + math.cos(math.pi * min(s / max(total, 1), 1.0))))
    t0 = time.time()
    for ep in range(args.proxy_epochs):
        model.train()
        for x, targets, lengths in tl:
            loss = ar_loss(model, x.to(device), targets, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for x, targets, lengths in vl:
            preds.extend(greedy_decode(model, x.to(device), kb.letters))
            refs.extend(decode.target_strings(targets, lengths, kb.letters))
    m = decode.score(preds, refs)
    return {"proxy_greedy": round(m["wacc"], 4), "proxy_cer": round(m["cer"], 4),
            "proxy_secs": round(time.time() - t0)}


def evaluate(path: Path, kb, device, args) -> dict:
    corpus = SwipeCorpus.load(path, kb.letters, limit=args.stat_n * 3)
    idx = [i for i in range(len(corpus))
           if 3 <= len(corpus.words[i]) <= 8][:args.stat_n]
    words = [corpus.words[i] for i in idx]
    pts = resampled(corpus, idx)
    row = shape_stats(pts, words, kb)

    val = SwipeCorpus.load(Path(args.cache) / "futo/validation", kb.letters,
                           limit=args.stat_n * 3)
    vidx = [i for i in range(len(val))
            if 3 <= len(val.words[i]) <= 8][:args.stat_n]
    row["c2st_auc"] = round(c2st(resampled(val, vidx), pts, device), 4)
    row["diversity"] = round(diversity(corpus, kb), 4)
    row.update(proxy_utility(path, kb, device, args))
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in row.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="run the corpora with known full-scale numbers and "
                         "report rank correlation")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--stat-n", type=int, default=1500)
    ap.add_argument("--proxy-train", type=int, default=120_000)
    ap.add_argument("--proxy-val", type=int, default=5_000)
    ap.add_argument("--proxy-epochs", type=int, default=3)
    ap.add_argument("--proxy-dim", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/gen_quality.json")
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    device = pick_device(args.device)
    targets = ([Path(args.cache) / k / "train" for k in KNOWN]
               if args.calibrate else [Path(args.corpus)])

    rows = {}
    hdr = ("corpus            path  end    visit  dwell  turn  zig   c2st  "
           "div   proxy   (secs)")
    print(hdr)
    print("-" * len(hdr))
    for p in targets:
        name = p.parent.name
        r = evaluate(p, kb, device, args)
        rows[name] = r
        print(f"{name:<16} {r['path_ratio']:.2f}  {r['end_err']:.3f}  "
              f"{r['visit']:.3f}  {r['dwell']:.3f}  {r['turn']:.2f}  "
              f"{r['zigzag']:.2f}  {r['c2st_auc']:.3f}  "
              f"{r['diversity']:.2f}  {r['proxy_greedy']:.3f}   "
              f"({r['proxy_secs']}s)", flush=True)

    if args.calibrate:
        names = [n for n in rows if n in KNOWN]
        proxy = np.array([rows[n]["proxy_greedy"] for n in names])
        for j, label in enumerate(["full greedy", "full beam"]):
            true = np.array([KNOWN[n][j] for n in names])
            rp, rt = proxy.argsort().argsort(), true.argsort().argsort()
            rho = float(np.corrcoef(rp, rt)[0, 1])
            print(f"\nproxy vs {label}: Spearman rho = {rho:.3f}")
            for n, p_, t_ in zip(names, proxy, true):
                print(f"   {n:<16} proxy {p_:.3f}   full {t_:.3f}")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
