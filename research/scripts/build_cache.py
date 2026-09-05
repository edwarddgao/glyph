#!/usr/bin/env python3
"""Normalize every source into one canonical Parquet cache.

    python scripts/build_cache.py --sources futo how_we_swipe --out data/canonical

FUTO's own train/dev/test split is preserved (it is partitioned by donor
session). How We Swipe is written as a single ``test`` split by default -- it is
most useful as held-out cross-corpus evaluation, not as extra training data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from swipe_typing import cache
from swipe_typing.sources import futo, how_we_swipe


def build_futo(out: Path, config: str, splits: list[str], limit: int | None,
               keep_flagged: bool) -> None:
    for split in splits:
        if split not in futo.FILES.get(config, {}):
            print(f"[skip] {config} has no {split} split")
            continue
        print(f"\n== futo {config}/{split} ==")
        path = futo.download(config, split)
        stream = futo.iter_swipes(path, split=split, config=config,
                                  keep_flagged=keep_flagged)
        if limit:
            stream = (s for i, s in enumerate(stream) if i < limit)
        dest = out / "futo" / split
        files = cache.write(tqdm(stream, unit=" swipes"), dest, prefix=config)
        print(f"  -> {len(files)} shard(s) in {dest}")


def build_hws(out: Path, logdir: Path, split: str, limit: int | None,
              keep_flagged: bool) -> None:
    if not logdir.is_dir():
        print(f"[skip] {logdir} not found -- run scripts/fetch_how_we_swipe.py")
        return
    print(f"\n== how_we_swipe/{split} ==")
    stream = how_we_swipe.iter_swipes(logdir, split=split, limit=limit,
                                      keep_flagged=keep_flagged)
    dest = out / "how_we_swipe" / split
    files = cache.write(tqdm(stream, unit=" swipes"), dest, prefix="hws")
    print(f"  -> {len(files)} shard(s) in {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["futo", "how_we_swipe"],
                    choices=["futo", "how_we_swipe"])
    ap.add_argument("--out", default="data/canonical")
    ap.add_argument("--futo-config", default="swipe-1")
    ap.add_argument("--futo-splits", nargs="+",
                    default=["train", "validation", "test"])
    ap.add_argument("--hws-logs", default="data/how_we_swipe/swipelogs")
    ap.add_argument("--hws-split", default="test",
                    help="split label for How We Swipe (default: held-out eval)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap swipes per split (for smoke tests)")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="keep attempts the source marked as errors")
    args = ap.parse_args()

    out = Path(args.out)
    if "futo" in args.sources:
        build_futo(out, args.futo_config, args.futo_splits, args.limit,
                   args.keep_flagged)
    if "how_we_swipe" in args.sources:
        build_hws(out, Path(args.hws_logs), args.hws_split, args.limit,
                  args.keep_flagged)

    print("\n== cache summary ==")
    for d in sorted(out.glob("*/*")):
        if d.is_dir():
            s = cache.stats(d)
            print(f"  {d.relative_to(out)}: {s['swipes']:,} swipes, "
                  f"{s['unique_words']:,} words, {s['sessions']:,} sessions")


if __name__ == "__main__":
    sys.exit(main())
