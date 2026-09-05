"""Canonical keyboard geometry.

Every source is mapped into one coordinate space:

    x in [0, 1] spans the 10 key columns of the top row
    y in [0, 1] spans the 3 letter rows (top row center at 1/6)

This is not an arbitrary choice -- it is exactly the space FUTO's own
``swipe-5/layouts/*.json`` files are expressed in. Their ``qwerty.json`` places
``a`` at (0.1005, 0.5) with half-extents (0.05, 0.1667), which ``key_center``
reproduces to within 5e-4. ``scripts/calibrate_layout.py`` independently recovers
the same grid from touch data in both corpora.

Aspect ratio is deliberately *not* baked into these coordinates: squashing every
keyboard to a unit square distorts angles and curvature by a different factor per
device. Each swipe carries an ``aspect`` (physical width/height of the letter
grid) so geometry-dependent features can be computed undistorted. See
``features.aspect_correct``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROWS: tuple[str, ...] = ("qwertyuiop", "asdfghjkl", "zxcvbnm")

#: Horizontal offset of each row, in units of key width (1/10 of the grid).
ROW_INSET: tuple[float, ...] = (0.0, 0.05, 0.15)

N_COLS = 10
N_ROWS = len(ROWS)

ALPHABET = "".join(sorted("".join(ROWS)))


def key_center(char: str) -> tuple[float, float]:
    """(x, y) center of ``char`` on the canonical QWERTY grid."""
    char = char.lower()
    for r, row in enumerate(ROWS):
        c = row.find(char)
        if c >= 0:
            return ((c + 0.5) / N_COLS + ROW_INSET[r], (r + 0.5) / N_ROWS)
    raise KeyError(f"{char!r} is not a letter key")


@dataclass(slots=True)
class KeyboardLayout:
    """Key centers and half-extents in canonical coordinates.

    ``centers`` is (K, 2) and ``radii`` is (K, 2), both indexed by ``letters``.
    Augmentation transforms ``centers`` jointly with the trajectory, which is
    what makes a decoder layout-agnostic rather than QWERTY-memorizing.
    """

    name: str
    letters: str
    centers: np.ndarray
    radii: np.ndarray

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float32).reshape(-1, 2)
        self.radii = np.asarray(self.radii, dtype=np.float32).reshape(-1, 2)
        if len(self.letters) != len(self.centers):
            raise ValueError(
                f"{self.name}: {len(self.letters)} letters vs "
                f"{len(self.centers)} key centers"
            )

    def __len__(self) -> int:
        return len(self.letters)

    def index(self, char: str) -> int:
        i = self.letters.find(char)
        if i < 0:
            raise KeyError(f"{char!r} not in layout {self.name!r}")
        return i

    def center(self, char: str) -> np.ndarray:
        return self.centers[self.index(char)]

    def covers(self, word: str) -> bool:
        return all(ch in self.letters for ch in word)

    def reindex(self, alphabet: str) -> "KeyboardLayout":
        """Restrict and reorder keys to match ``alphabet``.

        A trained model's output index *i* means ``alphabet[i]``, so any layout
        handed to it at inference must present its keys in that same order.
        Layouts also carry keys outside the model's alphabet (azerty and dvorak
        ship 27), which are dropped here.
        """
        missing = [ch for ch in alphabet if ch not in self.letters]
        if missing:
            raise KeyError(
                f"layout {self.name!r} lacks {''.join(missing)!r}"
            )
        idx = [self.index(ch) for ch in alphabet]
        return KeyboardLayout(self.name, alphabet, self.centers[idx],
                              self.radii[idx])

    @classmethod
    def qwerty(cls) -> "KeyboardLayout":
        """The canonical 3x10 English QWERTY grid."""
        letters = ALPHABET
        centers = np.array([key_center(ch) for ch in letters], dtype=np.float32)
        radii = np.tile(
            np.array([0.5 / N_COLS, 0.5 / N_ROWS], dtype=np.float32),
            (len(letters), 1),
        )
        return cls("qwerty", letters, centers, radii)

    @classmethod
    def from_futo_json(cls, path: str | Path) -> "KeyboardLayout":
        """Load one of FUTO's ``swipe-5/layouts/*.json`` files.

        These are already in canonical coordinates, so no rescaling is needed.
        """
        spec = json.loads(Path(path).read_text())
        keys = {k["letter"]: k for k in spec["keys"]}
        letters = "".join(ch for ch in spec["letters"] if ch in keys)
        centers = np.array([[keys[c]["cx"], keys[c]["cy"]] for c in letters], np.float32)
        radii = np.array([[keys[c]["rx"], keys[c]["ry"]] for c in letters], np.float32)
        return cls(spec.get("name", Path(path).stem), letters, centers, radii)


def layout_tensor(alphabet: str = ALPHABET) -> np.ndarray:
    """(K, 2) float32 key centers on canonical QWERTY, ordered by ``alphabet``."""
    return np.array([key_center(ch) for ch in alphabet], dtype=np.float32)


def ideal_trace(word: str, points_per_key: int = 8,
                kb: KeyboardLayout | None = None) -> np.ndarray:
    """Straight-line path through the key centers of ``word``.

    A synthetic baseline, and a way to sanity-check that a trajectory actually
    lands on the keys its label claims. Returns (N, 2) float32.
    """
    kb = kb or KeyboardLayout.qwerty()
    keys = [kb.center(ch) for ch in word.lower() if ch in kb.letters]
    if not keys:
        raise ValueError(f"no layout keys in {word!r}")
    if len(keys) == 1:
        return np.array(keys, dtype=np.float32)
    segs = []
    for a, b in zip(keys[:-1], keys[1:]):
        ts = np.linspace(0.0, 1.0, points_per_key, endpoint=False)[:, None]
        segs.append(np.asarray(a) + ts * (np.asarray(b) - np.asarray(a)))
    segs.append(np.asarray([keys[-1]]))
    return np.concatenate(segs).astype(np.float32)
