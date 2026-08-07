"""Read/write the canonical corpus as Parquet shards.

Normalizing once and caching means training never re-parses 70MB of text logs or
2GB of JSONL, and every source ends up in a single schema (``schema.ARROW_SCHEMA``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import ARROW_SCHEMA, Swipe

SHARD_ROWS = 50_000


def write(swipes: Iterable[Swipe], out_dir: str | Path, prefix: str = "part",
          shard_rows: int = SHARD_ROWS, compression: str = "zstd") -> list[Path]:
    """Write swipes to ``out_dir`` as one or more Parquet shards."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    buf: list[dict] = []
    shard = 0

    def flush() -> None:
        nonlocal buf, shard
        if not buf:
            return
        table = pa.Table.from_pylist(buf, schema=ARROW_SCHEMA)
        path = out_dir / f"{prefix}-{shard:05d}.parquet"
        pq.write_table(table, path, compression=compression)
        written.append(path)
        shard += 1
        buf = []

    for sw in swipes:
        buf.append(sw.as_row())
        if len(buf) >= shard_rows:
            flush()
    flush()
    return written


def read(path: str | Path, columns: list[str] | None = None) -> Iterator[Swipe]:
    """Stream swipes back from a shard file or a directory of shards."""
    path = Path(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=columns):
            for row in batch.to_pylist():
                yield Swipe.from_row(row)


def stats(path: str | Path) -> dict:
    """Row counts and unique-word/session counts for a cache directory."""
    path = Path(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    n = 0
    words: set[str] = set()
    sessions: set[str] = set()
    for f in files:
        table = pq.read_table(f, columns=["word", "session"])
        n += table.num_rows
        words.update(table.column("word").to_pylist())
        sessions.update(table.column("session").to_pylist())
    return {
        "shards": len(files),
        "swipes": n,
        "unique_words": len(words),
        "sessions": len(sessions),
    }
