#!/usr/bin/env python3
"""Fit the minimum-jerk generator and measure how real its gestures look.

    python scripts/probe_minjerk.py --decode-limit 3000 --device cpu

Three read-outs, each against real futo/val gestures for the same words:
  1. the fitted CLC duration law and its RMSE (the paper's Table-7-adjacent
     sanity check: WordGesture-GAN 1180ms vs CLC 1150ms on their corpus);
  2. kinematic realism -- speed percentiles, dwell fraction, mean |jerk| --
     where min-jerk's known signature is being too smooth by about half;
  3. readability: greedy decode of synthetic vs real gestures under the
     frozen AR decoder. High synthetic accuracy is necessary for training
     utility but suspicious if it far exceeds real -- it means the gestures
     are too clean to teach robustness.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing import features, minjerk  # noqa: E402
from swipe_typing.layout import KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeCorpus, SwipeDataset, decode, make_loader  # noqa: E402
from swipe_typing.model.ar import greedy_decode  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import pick_device  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402


def kin_stats(corpus: SwipeCorpus, label: str) -> dict:
    speeds, jerks, dwell = [], [], []
    for i in range(len(corpus)):
        pts = features.aspect_correct(corpus.points(i), float(corpus.aspects[i]))
        t = corpus.times(i)
        xy = features.resample(pts, t, n=features.N_POINTS, mode="time")
        dur = max(float(t[-1] - t[0]), 1.0) / 1000.0
        dt = dur / (features.N_POINTS - 1)
        _, vel, acc = features._smooth_derivatives(xy.astype(np.float64), dt)
        speed = np.linalg.norm(vel, axis=1)
        jerk = np.linalg.norm(np.gradient(acc, dt, axis=0), axis=1)
        speeds.append(speed)
        jerks.append(jerk.mean())
        dwell.append(speed)
    speed_all = np.concatenate(speeds)
    return {
        "label": label,
        "speed_p50": float(np.percentile(speed_all, 50)),
        "speed_p90": float(np.percentile(speed_all, 90)),
        "jerk_mean": float(np.mean(jerks)),
        "speed_pool": speed_all,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--checkpoint", default="runs/ar_full/ar_decoder.pt")
    ap.add_argument("--fit-limit", type=int, default=100_000)
    ap.add_argument("--val-limit", type=int, default=20_000)
    ap.add_argument("--decode-limit", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/minjerk_model.json")
    ap.add_argument("--profile", default=None,
                    choices=["spline", "segments", "random"])
    ap.add_argument("--dwell-prob", type=float, default=None)
    ap.add_argument("--tremor", type=float, default=None)
    ap.add_argument("--seg-jitter", type=float, default=None)
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)

    print("fitting on futo/train ...")
    t0 = time.time()
    train = SwipeCorpus.load(root / "futo/train", kb.letters,
                             limit=args.fit_limit)
    model = minjerk.fit(train, kb, max_swipes=args.fit_limit)
    for k in ("profile", "dwell_prob", "tremor", "seg_jitter"):
        v = getattr(args, k)
        if v is not None:
            setattr(model, k, v)
    model.save(args.out)
    print(f"  m={model.m:.1f}  n={model.n:.2f}  log_sigma={model.log_sigma:.3f}"
          f"  offset_sigma=({model.offset_sigma_x:.3f}, "
          f"{model.offset_sigma_y:.3f}) key-halfwidths"
          f"  aspect={model.aspect:.2f}  ({time.time() - t0:.0f}s)")

    val = SwipeCorpus.load(root / "futo/validation", kb.letters,
                           limit=args.val_limit)

    # 1. Duration law on held-out words.
    pred, real_dur = [], []
    for i in range(len(val)):
        w = val.words[i]
        if len(w) < 2:
            continue
        aspect = float(val.aspects[i])
        lengths = minjerk._segment_lengths(
            w, kb, aspect if np.isfinite(aspect) and aspect > 0 else 1.0)
        pred.append(model.m * float((lengths**model.n).sum()))
        t = val.times(i)
        real_dur.append(float(t[-1] - t[0]))
    pred, real_dur = np.asarray(pred), np.asarray(real_dur)
    rmse = float(np.sqrt(((pred - real_dur) ** 2).mean()))
    print(f"\nduration on val: real mean {real_dur.mean():.0f}ms  "
          f"pred mean {pred.mean():.0f}ms  RMSE {rmse:.0f}ms")

    # 2. Kinematics, real vs synthetic for the same words.
    sub = list(range(0, len(val), max(len(val) // 4000, 1)))
    words = [val.words[i] for i in sub]
    synth = minjerk.generate_many(model, words, kb, seed=1)
    synth_corpus = SwipeCorpus.from_swipes(synth, kb.letters)
    real_sub = SwipeCorpus.from_swipes(
        [_as_swipe(val, i) for i in sub], kb.letters)

    r = kin_stats(real_sub, "real")
    s = kin_stats(synth_corpus, "minjerk")
    slow = float(np.percentile(r["speed_pool"], 20))
    for st in (r, s):
        dwell_frac = float((st["speed_pool"] < slow).mean())
        print(f"  {st['label']:8s} speed p50/p90 {st['speed_p50']:.3f}/"
              f"{st['speed_p90']:.3f}  mean|jerk| {st['jerk_mean']:.1f}  "
              f"dwell<p20 {dwell_frac:.2f}")

    # 3. Readability under the frozen decoder.
    device = pick_device(args.device)
    ar, alphabet, mode = load_ar(args.checkpoint, device)
    for label, corpus in [("real", real_sub), ("minjerk", synth_corpus)]:
        n = min(args.decode_limit, len(corpus))
        ds = SwipeDataset(
            SwipeCorpus.from_swipes(
                [_as_swipe(corpus, i) for i in range(n)], kb.letters),
            kb, augment_cfg=None, resample_mode=mode,
            shape_only=ar.cfg.shape_only)
        loader = make_loader(ds, batch_size=256, shuffle=False, num_workers=2)
        preds, refs = [], []
        with torch.no_grad():
            for x, targets, lengths in loader:
                preds.extend(greedy_decode(ar, x.to(device), alphabet))
                refs.extend(decode.target_strings(targets, lengths, alphabet))
        m = decode.score(preds, refs)
        print(f"  greedy on {label:8s} n={len(refs):,}  "
              f"wacc {m['wacc']:.3f}  cer {m['cer']:.3f}")


def _as_swipe(corpus: SwipeCorpus, i: int):
    from swipe_typing.schema import Swipe
    pts = corpus.points(i)
    return Swipe(word=corpus.words[i], x=pts[:, 0], y=pts[:, 1],
                 t=corpus.times(i), aspect=float(corpus.aspects[i]),
                 session=corpus.sessions[i], source="probe")


if __name__ == "__main__":
    main()
