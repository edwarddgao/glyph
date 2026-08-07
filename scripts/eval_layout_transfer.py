#!/usr/bin/env python3
"""Score a QWERTY-trained encoder on layouts it has never seen.

This is the test the architecture was built for. The encoder never receives
absolute coordinates -- the keyboard reaches it only as per-frame affinity to
whatever key positions are supplied at runtime, in units of each key's own
half-extents. If that design works, handing it a different layout plus real
gestures performed on that layout should just work, with no retraining.

FUTO's swipe-5 config makes this testable with real human data:

    clearflow   11.8k English gestures   5 rows x 8 cols
    dvorak       2.8k                    4 rows
    kasroz       1.1k                    5 rows x 8 cols
    qwertz       0.8k                    3 rows (a permutation of qwerty)

clearflow and kasroz matter most: their grid geometry is nothing like the 3x10
training layout, so passing there cannot be explained by the model having
memorized a QWERTY-shaped keyboard.

Usage:
    python scripts/eval_layout_transfer.py --checkpoint runs/full/encoder.pt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
from swipe_typing.sources import futo  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device: torch.device):
    """Load a checkpoint along with the feature settings it was trained under.

    The feature convention has to travel with the weights -- evaluating a
    key-units model with grid-unit features silently produces garbage rather
    than an error.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["dilations"] = tuple(cfg_dict["dilations"])
    cfg = EncoderConfig(**cfg_dict)
    model = SwipeEncoder(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    train_args = ckpt.get("args", {}) or {}
    return (model, ckpt.get("alphabet", ALPHABET),
            not train_args.get("no_key_units", False),
            train_args.get("resample_mode", "time"))


def group_by_layout(path, language: str, alphabet: str):
    """Bucket swipe-5 records by the layout they were performed on."""
    letters = set(alphabet)
    buckets = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if language and rec.get("language") != language:
                continue
            sw = futo.parse_record(rec, split="test", source="futo/swipe-5")
            if sw is None or not letters.issuperset(sw.word):
                continue
            buckets[rec.get("layout") or "?"].append(sw)
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/full/encoder.pt")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min-swipes", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, key_units, resample_mode = load_model(args.checkpoint, device)
    print(f"checkpoint {args.checkpoint}  ({model.num_parameters():,} params)")
    print(f"device {device}  alphabet {len(alphabet)} letters")
    print(f"features: key_units={key_units} resample={resample_mode}\n")

    print("loading swipe-5 ...")
    buckets = group_by_layout(
        futo.download("swipe-5", "train"), args.language, alphabet
    )

    rows = []
    for name, swipes in sorted(buckets.items(),
                               key=lambda kv: -len(kv[1])):
        if len(swipes) < args.min_swipes:
            continue
        try:
            kb = KeyboardLayout.from_futo_json(futo.download_layout(name))
            kb = kb.reindex(alphabet)
        except (KeyError, OSError) as exc:
            print(f"  [skip] {name}: {exc}")
            continue

        # Columns = widest row. Counting distinct x positions instead would
        # report 19 for qwerty, since each row is inset by half a key.
        rows_y = [round(float(c), 2) for c in kb.centers[:, 1]]
        n_rows = len(set(rows_y))
        n_cols = max(rows_y.count(v) for v in set(rows_y))

        corpus = SwipeCorpus.from_swipes(swipes, alphabet, origin=name)
        ds = SwipeDataset(corpus, kb, augment_cfg=None,
                          resample_mode=resample_mode, key_units=key_units)
        loader = make_loader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)
        m = decode.evaluate(model, loader, device, alphabet, collect=4)
        rows.append((name, n_rows, n_cols, len(corpus), m))

    print(f"\n{'layout':<12}{'grid':>8}{'n':>8}{'CER':>8}{'word acc':>10}")
    print("-" * 46)
    for name, nr, nc, n, m in rows:
        print(f"{name:<12}{f'{nr}x{nc}':>8}{n:>8,}{m['cer']:>8.3f}"
              f"{m['wacc']:>10.3f}")

    print("\nsamples:")
    for name, _, _, _, m in rows:
        print(f"  [{name}]")
        for ref, pred in m.get("samples", []):
            flag = " " if ref == pred else "x"
            print(f"    {flag} {ref:<16} -> {pred}")


if __name__ == "__main__":
    main()
