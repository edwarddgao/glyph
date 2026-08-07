"""Torch dataset over the canonical Parquet cache.

Swipes are held as flat concatenated arrays with an offset index rather than a
list of per-swipe arrays -- 940k small numpy objects costs far more in overhead
than the data itself.

Augmentation has to happen *before* featurization (an affine changes the geometry
the features are derived from), so features cannot be precomputed. They are built
per item in DataLoader workers instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import cache, features
from ..augment import DEFAULT as DEFAULT_AUG
from ..augment import AugmentConfig, apply_affine, sample_affine
from ..layout import KeyboardLayout

MAX_WORD_LEN = 24


class SwipeCorpus:
    """Flat in-memory store of canonical swipes."""

    def __init__(self, x, y, t, offsets, words, aspects):
        self.x, self.y, self.t = x, y, t
        self.offsets = offsets
        self.words = words
        self.aspects = aspects

    def __len__(self) -> int:
        return len(self.words)

    def points(self, i: int) -> np.ndarray:
        a, b = self.offsets[i], self.offsets[i + 1]
        return np.stack([self.x[a:b], self.y[a:b]], axis=1)

    def times(self, i: int) -> np.ndarray:
        return self.t[self.offsets[i]:self.offsets[i + 1]]

    @classmethod
    def load(cls, path: str | Path, alphabet: str, limit: int | None = None,
             max_word_len: int = MAX_WORD_LEN) -> "SwipeCorpus":
        return cls.from_swipes(cache.read(path), alphabet, limit=limit,
                               max_word_len=max_word_len, origin=str(path))

    @classmethod
    def from_swipes(cls, swipes, alphabet: str, limit: int | None = None,
                    max_word_len: int = MAX_WORD_LEN,
                    origin: str = "<swipes>") -> "SwipeCorpus":
        letters = set(alphabet)
        xs, ys, ts, words, aspects = [], [], [], [], []
        offsets = [0]
        for sw in swipes:
            w = sw.word
            if not w or len(w) > max_word_len or not letters.issuperset(w):
                continue
            xs.append(sw.x)
            ys.append(sw.y)
            ts.append(sw.t)
            words.append(w)
            aspects.append(sw.aspect)
            offsets.append(offsets[-1] + len(sw.x))
            if limit and len(words) >= limit:
                break
        if not words:
            raise ValueError(f"no usable swipes in {origin}")
        return cls(
            np.concatenate(xs).astype(np.float32),
            np.concatenate(ys).astype(np.float32),
            np.concatenate(ts).astype(np.int32),
            np.asarray(offsets, dtype=np.int64),
            words,
            np.asarray(aspects, dtype=np.float32),
        )


class SwipeDataset(Dataset):
    """Yields ``(features, target_indices)`` for CTC training.

    Args:
        corpus: loaded :class:`SwipeCorpus`.
        layout: keyboard the affinities are computed against. During training
            it is co-augmented with each gesture; at eval it is used as-is,
            which is what lets a QWERTY-trained model be scored on azerty.
        augment_cfg: ``None`` disables augmentation (use for validation).
    """

    def __init__(self, corpus: SwipeCorpus, layout: KeyboardLayout,
                 augment_cfg: AugmentConfig | None = DEFAULT_AUG,
                 n_points: int = features.N_POINTS, resample_mode: str = "time",
                 key_units: bool = True, seed: int = 0):
        self.corpus = corpus
        self.layout = layout
        self.cfg = augment_cfg
        self.n_points = n_points
        self.resample_mode = resample_mode
        self.key_units = key_units
        self.seed = seed
        self.char_to_idx = {ch: i for i, ch in enumerate(layout.letters)}

    def __len__(self) -> int:
        return len(self.corpus)

    def _rng(self, idx: int) -> np.random.Generator:
        # Seeded per (epoch-agnostic) index *and* global seed so workers do not
        # replay identical augmentations across processes.
        return np.random.default_rng((self.seed * 1_000_003 + idx) & 0xFFFFFFFF)

    def __getitem__(self, idx: int):
        pts = self.corpus.points(idx)
        t = self.corpus.times(idx)
        aspect = float(self.corpus.aspects[idx])
        word = self.corpus.words[idx]
        centers, radii = self.layout.centers, self.layout.radii

        if self.cfg is not None:
            rng = self._rng(idx + self.seed)
            mat = sample_affine(self.cfg, rng)
            pts = apply_affine(pts, mat)
            centers = apply_affine(centers, mat)
            radii = np.abs(radii @ np.abs(mat[:2, :2]).T).astype(np.float32)
            if self.cfg.jitter:
                pts = pts + rng.normal(0.0, self.cfg.jitter, pts.shape).astype(
                    np.float32
                )

        resampled = features.resample(pts, t, n=self.n_points,
                                      mode=self.resample_mode)
        aff = features.key_affinity(resampled, centers, radii)
        # Measure motion in keys/second so a 5-row layout and a 3-row one put
        # the same gesture on the same scale.
        scale = features.key_scale(radii) if self.key_units else None
        kin = features.kinematics(pts, t, aspect, n=self.n_points,
                                  mode=self.resample_mode, scale=scale)
        x = np.concatenate([aff, kin], axis=1)

        target = np.fromiter(
            (self.char_to_idx[c] for c in word), dtype=np.int64, count=len(word)
        )
        return torch.from_numpy(x), torch.from_numpy(target)


def collate(batch):
    """Pack a batch for ``F.ctc_loss``: features stacked, targets concatenated."""
    xs, targets = zip(*batch)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(xs), torch.cat(targets), lengths


def make_loader(dataset: SwipeDataset, batch_size: int = 256, shuffle: bool = True,
                num_workers: int = 4, **kwargs):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        **kwargs,
    )
