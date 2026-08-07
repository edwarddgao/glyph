#!/usr/bin/env python3
"""Why is cross-corpus accuracy 12 points lower?

Before assuming the answer is "more data", find out *which* swipes fail. How We
Swipe ships per-participant metadata FUTO has no equivalent for -- device size,
which finger, which hand, self-rated English -- so the gap can be attributed
rather than guessed at.

Three hypotheses this separates:

  geometry     accuracy tracks keyboard aspect / screen size, meaning the
               canonical-space mapping still has a residual error
  population   accuracy tracks who is typing (finger, hand, English level),
               meaning FUTO's donors are simply not representative
  long tail    a minority of users are catastrophically bad and drag the mean,
               versus everyone being uniformly a bit worse

Each implies a different fix, and only the third is really a data-volume story.

Usage:
    python scripts/diagnose_transfer.py --checkpoint runs/full/encoder.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (  # noqa: E402
    EncoderConfig,
    SwipeCorpus,
    SwipeDataset,
    SwipeEncoder,
    decode,
    make_loader,
)
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import build_lexicon, load_model, pick_device  # noqa: E402


def read_metadata(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["uid"]: row for row in csv.DictReader(fh, delimiter="\t")}


def bucket_report(title: str, groups: dict[str, list[bool]], min_n: int = 100):
    rows = [(k, sum(v) / len(v), len(v)) for k, v in groups.items() if len(v) >= min_n]
    if not rows:
        return
    rows.sort(key=lambda r: -r[1])
    print(f"\n  {title}")
    for key, acc, n in rows:
        bar = "#" * int(acc * 40)
        print(f"    {str(key):<18}{n:>7,}  {acc:6.3f}  {bar}")
    spread = rows[0][1] - rows[-1][1]
    print(f"    {'spread':<18}{'':>7}  {spread:6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--metadata", default="data/how_we_swipe/metadata.tsv")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--beam-width", type=int, default=32)
    ap.add_argument("--lexicon", default="train+wf320k")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default="how_we_swipe/test")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, mode = load_model(args.checkpoint, device)
    kb = KeyboardLayout.qwerty()
    root = Path(args.cache)
    lexicon = build_lexicon(args.lexicon, root, alphabet, 1.0)
    meta = read_metadata(Path(args.metadata))
    print(f"lexicon {len(lexicon):,} words; metadata for {len(meta):,} participants")

    corpus = SwipeCorpus.load(root / args.split, alphabet, limit=args.limit)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units)
    loader = make_loader(ds, batch_size=256, shuffle=False, num_workers=2)

    chunks = []
    with torch.no_grad():
        for x, _, _ in loader:
            chunks.append(model(x.to(device)).float().cpu().numpy())
    log_probs = np.concatenate(chunks)

    cfg = BeamConfig(beam_width=args.beam_width, top_k=1)
    correct = []
    for i, item in enumerate(log_probs):
        hyps = beam_search(item, lexicon, alphabet, cfg)
        correct.append(bool(hyps) and hyps[0][0] == corpus.words[i])

    n = len(correct)
    print(f"\n== {args.split}  n={n:,}  overall {sum(correct) / n:.3f} ==")

    # --- population ---------------------------------------------------------
    for field in ("english_level", "swipe_finger", "swipe_hand",
                  "dominant_hand", "familiarity"):
        groups = defaultdict(list)
        for i, ok in enumerate(correct):
            row = meta.get(corpus.sessions[i])
            if row and row.get(field):
                groups[row[field]].append(ok)
        bucket_report(f"by {field}", groups)

    # --- geometry -----------------------------------------------------------
    groups = defaultdict(list)
    for i, ok in enumerate(correct):
        a = float(corpus.aspects[i])
        groups[f"{np.floor(a * 4) / 4:.2f}"].append(ok)
    bucket_report("by keyboard aspect (letter grid w/h)", groups)

    groups = defaultdict(list)
    for i, ok in enumerate(correct):
        row = meta.get(corpus.sessions[i])
        if row and row.get("screen_width", "").isdigit():
            w = int(row["screen_width"])
            groups[f"{w // 60 * 60}-{w // 60 * 60 + 59}px"].append(ok)
    bucket_report("by screen width", groups)

    # --- long tail ----------------------------------------------------------
    per_user = defaultdict(list)
    for i, ok in enumerate(correct):
        per_user[corpus.sessions[i]].append(ok)
    accs = sorted(sum(v) / len(v) for v in per_user.values() if len(v) >= 20)
    if accs:
        print(f"\n  per-user accuracy ({len(accs)} users with >=20 swipes)")
        for q in (5, 10, 25, 50, 75, 90, 95):
            print(f"    p{q:<17}{'':>7}  {accs[int(len(accs) * q / 100)]:6.3f}")
        print(f"    {'mean':<18}{'':>7}  {statistics.mean(accs):6.3f}")
        worst = sum(1 for a in accs if a < 0.5)
        print(f"\n    users below 0.50 accuracy: {worst} of {len(accs)} "
              f"({worst / len(accs):.1%})")
        # How much of the total error do the worst users account for?
        ranked = sorted(per_user.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        errs = sum(len(v) - sum(v) for v in per_user.values())
        cum = 0
        for frac in (0.1, 0.25):
            k = max(int(len(ranked) * frac), 1)
            cum = sum(len(v) - sum(v) for _, v in ranked[:k])
            print(f"    worst {frac:.0%} of users hold {cum / max(errs, 1):.1%} "
                  f"of all errors")


if __name__ == "__main__":
    main()
