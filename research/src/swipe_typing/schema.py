"""The one record type every source normalizes into."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

#: Arrow schema for the on-disk canonical cache.
ARROW_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("split", pa.string()),
        pa.field("session", pa.string()),
        pa.field("word", pa.string()),
        pa.field("x", pa.list_(pa.float32())),
        pa.field("y", pa.list_(pa.float32())),
        pa.field("t", pa.list_(pa.int32())),
        pa.field("aspect", pa.float32()),
        pa.field("sentence", pa.string()),
        pa.field("word_idx", pa.int32()),
        pa.field("flagged", pa.bool_()),
        # Present for multi-layout corpora (FUTO swipe-5); empty elsewhere.
        pa.field("layout", pa.string()),
        pa.field("language", pa.string()),
    ]
)


@dataclass(slots=True)
class Swipe:
    """A single gesture, in canonical keyboard coordinates.

    Attributes:
        word: Target word, lowercased, letters only.
        x: (N,) float32 in canonical units; 0..1 spans the 10 key columns.
        y: (N,) float32 in canonical units; 0..1 spans the 3 letter rows.
        t: (N,) int32 milliseconds, rebased so ``t[0] == 0``.
        aspect: Physical width/height of the letter grid on the source device.
            Needed to compute angles and curvature undistorted.
        session: Stable per-donor id. Splits must be grouped by this.
        flagged: Source marked this attempt as an error/failure. Excluded by
            default when building the cache.
    """

    word: str
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    aspect: float
    session: str
    source: str
    split: str = "train"
    sentence: str = ""
    word_idx: int = -1
    flagged: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
        self.t = np.asarray(self.t, dtype=np.int32)
        if not (len(self.x) == len(self.y) == len(self.t)):
            raise ValueError(
                f"ragged swipe for {self.word!r}: "
                f"{len(self.x)}/{len(self.y)}/{len(self.t)}"
            )

    def __len__(self) -> int:
        return len(self.x)

    @property
    def points(self) -> np.ndarray:
        """(N, 2) float32 array of canonical coordinates."""
        return np.stack([self.x, self.y], axis=1)

    @property
    def duration_ms(self) -> int:
        return int(self.t[-1] - self.t[0]) if len(self.t) else 0

    def as_row(self) -> dict:
        return {
            "source": self.source,
            "split": self.split,
            "session": self.session,
            "word": self.word,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "t": self.t.tolist(),
            "aspect": float(self.aspect),
            "sentence": self.sentence,
            "word_idx": int(self.word_idx),
            "flagged": bool(self.flagged),
            "layout": str(self.meta.get("layout", "") or ""),
            "language": str(self.meta.get("language", "") or ""),
        }

    @classmethod
    def from_row(cls, row: dict) -> "Swipe":
        return cls(
            word=row["word"],
            x=np.asarray(row["x"], dtype=np.float32),
            y=np.asarray(row["y"], dtype=np.float32),
            t=np.asarray(row["t"], dtype=np.int32),
            aspect=row["aspect"],
            session=row["session"],
            source=row["source"],
            split=row.get("split", "train"),
            sentence=row.get("sentence", "") or "",
            word_idx=row.get("word_idx", -1),
            flagged=row.get("flagged", False),
            meta={k: row[k] for k in ("layout", "language")
                  if row.get(k)},
        )


def is_plausible(sw: Swipe, min_points: int = 4, max_points: int = 2000) -> bool:
    """Cheap sanity filter shared by every loader."""
    if not sw.word or not sw.word.isalpha():
        return False
    n = len(sw)
    if n < min_points or n > max_points:
        return False
    if not (np.isfinite(sw.x).all() and np.isfinite(sw.y).all()):
        return False
    # Allow modest overshoot past the grid edge, but reject wild coordinates.
    if sw.x.min() < -0.5 or sw.x.max() > 1.5:
        return False
    if sw.y.min() < -0.5 or sw.y.max() > 1.5:
        return False
    if sw.duration_ms <= 0 or sw.duration_ms > 30_000:
        return False
    return True
