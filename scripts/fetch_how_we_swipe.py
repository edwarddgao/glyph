#!/usr/bin/env python3
"""Download the How We Swipe release from OSF (https://osf.io/sj67f/).

Pulls ``swipelogs.zip`` (~70MB, expands to ~920MB of text logs) plus the
aggregate TSVs, into ``data/how_we_swipe/``.

Usage:
    python scripts/fetch_how_we_swipe.py [--out data/how_we_swipe] [--logs-only]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

OSF_NODE = "sj67f"
OSF_FILES = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/"
OSF_DOWNLOAD = "https://osf.io/download/{fid}/"

WANTED = ("swipelogs.zip", "metadata.tsv", "stats-words.tsv", "stats-sentences.tsv",
          "wordfreq.txt")


def listing() -> dict[str, tuple[str, int]]:
    with urllib.request.urlopen(OSF_FILES, timeout=60) as fh:
        payload = json.load(fh)
    out = {}
    for item in payload.get("data", []):
        attrs = item["attributes"]
        if attrs.get("kind") == "file":
            out[attrs["name"]] = (item["id"].split("/")[-1], attrs.get("size") or 0)
    return out


def fetch(fid: str, dest: Path, size: int) -> None:
    if dest.exists() and (not size or dest.stat().st_size == size):
        print(f"  have {dest.name}")
        return
    print(f"  get  {dest.name} ({size / 1e6:.1f} MB)")
    urllib.request.urlretrieve(OSF_DOWNLOAD.format(fid=fid), dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/how_we_swipe")
    ap.add_argument("--logs-only", action="store_true",
                    help="skip the aggregate TSV/word-frequency files")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = listing()
    wanted = ("swipelogs.zip",) if args.logs_only else WANTED
    print(f"OSF node {OSF_NODE}: {len(files)} files")
    for name in wanted:
        if name not in files:
            print(f"  [warn] {name} not in OSF listing")
            continue
        fid, size = files[name]
        fetch(fid, out / name, size)

    zpath = out / "swipelogs.zip"
    logdir = out / "swipelogs"
    if zpath.exists() and not logdir.exists():
        print(f"  unzip -> {logdir}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out)

    if logdir.exists():
        n_logs = len(list(logdir.glob("*.log")))
        print(f"\nready: {n_logs} participant logs in {logdir}")


if __name__ == "__main__":
    main()
