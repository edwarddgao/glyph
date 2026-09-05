"""Second-pass acoustic rescorer over first-pass n-best lists.

The measured motivation: a pretrained GPT-2 recovers only ~27% of the n-best
headroom on futo/validation, and scaling the LM barely moves it. The rest is
acoustic -- ``philipp``/``philip``, ``wayne``/``warner`` -- candidates the first
pass orders wrongly on gesture evidence alone.

A second pass can do better because it is not under the first pass's
constraints. CTC treats frames as conditionally independent given the label, and
beam search has to stay streamable and cover a 300k-word lexicon. A rescorer
sees one whole gesture against ~8 candidates, so it can attend from each
candidate letter to the whole trajectory and model exactly the dependencies the
first pass gives up.

It learns a **residual**: the final logit is ``scale * first_pass + model(...)``,
so it only has to fix the ordering the first pass gets wrong rather than
relearn decoding from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layout import ALPHABET, KeyboardLayout
from .encoder import ResidualBlock

MAX_CAND_LEN = 24
PAD = 26  # one past the alphabet


@dataclass
class RescoreConfig:
    n_input: int = 32          # gesture feature channels (26 affinity + 6 kinematic)
    n_letters: int = 26
    d_model: int = 96
    n_heads: int = 4
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    max_len: int = MAX_CAND_LEN


def encode_candidates(words: list[str], layout: KeyboardLayout | None = None,
                      max_len: int = MAX_CAND_LEN):
    """Words -> (letter ids, key centers, padding mask).

    Each candidate letter carries its key position, so the model compares the
    trajectory against where the candidate's keys actually are rather than
    against a bare symbol sequence.
    """
    layout = layout or KeyboardLayout.qwerty()
    n = len(words)
    ids = np.full((n, max_len), PAD, dtype=np.int64)
    pos = np.zeros((n, max_len, 2), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=bool)   # True where padded
    for i, w in enumerate(words):
        w = w[:max_len]
        for j, ch in enumerate(w):
            k = layout.letters.find(ch)
            if k < 0:
                continue
            ids[i, j] = k
            pos[i, j] = layout.centers[k]
        mask[i, len(w):] = True
        if not w:
            mask[i, :] = True
            mask[i, 0] = False        # never fully-masked: attention needs a row
    return ids, pos, mask


class Rescorer(nn.Module):
    """Scores (gesture, candidate) pairs; returns a residual logit per candidate."""

    def __init__(self, cfg: RescoreConfig | None = None):
        super().__init__()
        self.cfg = cfg or RescoreConfig()
        c = self.cfg
        self.register_buffer("input_mean", torch.zeros(c.n_input))
        self.register_buffer("input_std", torch.ones(c.n_input))

        self.gesture_proj = nn.Conv1d(c.n_input, c.d_model, 1)
        self.gesture_blocks = nn.ModuleList(
            ResidualBlock(c.d_model, 5, d, c.dropout) for d in c.dilations
        )
        self.gesture_norm = nn.GroupNorm(1, c.d_model)

        self.letter_emb = nn.Embedding(c.n_letters + 1, c.d_model, padding_idx=PAD)
        self.pos_proj = nn.Linear(2, c.d_model)
        self.order_emb = nn.Embedding(c.max_len, c.d_model)

        self.attn = nn.MultiheadAttention(c.d_model, c.n_heads,
                                          dropout=c.dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(c.d_model)
        self.head = nn.Sequential(
            nn.Linear(c.d_model * 2, c.d_model), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.d_model, 1),
        )
        # Weight on the first-pass score. Starts at 1 so the model begins as a
        # no-op refinement of the existing ranking.
        self.first_pass_scale = nn.Parameter(torch.ones(1))

    def set_normalization(self, mean, std) -> None:
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def encode_gesture(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, d)."""
        x = (x - self.input_mean) / self.input_std
        h = self.gesture_proj(x.transpose(1, 2))
        for block in self.gesture_blocks:
            h = block(h)
        return F.gelu(self.gesture_norm(h)).transpose(1, 2)

    def forward(self, gesture: torch.Tensor, cand_ids: torch.Tensor,
                cand_pos: torch.Tensor, cand_mask: torch.Tensor,
                first_pass: torch.Tensor) -> torch.Tensor:
        """Score a batch of n-best lists.

        Args:
            gesture: (B, T, C) features.
            cand_ids / cand_pos / cand_mask: (B, K, L[, 2]).
            first_pass: (B, K) beam scores.

        Returns:
            (B, K) logits.
        """
        b, k, ln = cand_ids.shape
        mem = self.encode_gesture(gesture)                      # (B, T, d)
        t, d = mem.shape[1], mem.shape[2]
        # One gesture encoding shared across its K candidates.
        mem = mem.unsqueeze(1).expand(b, k, t, d).reshape(b * k, t, d)

        order = torch.arange(ln, device=cand_ids.device)
        q = (self.letter_emb(cand_ids.reshape(b * k, ln))
             + self.pos_proj(cand_pos.reshape(b * k, ln, 2))
             + self.order_emb(order)[None])
        mask = cand_mask.reshape(b * k, ln)

        attended, _ = self.attn(q, mem, mem, need_weights=False)
        h = self.attn_norm(q + attended)

        keep = (~mask).float().unsqueeze(-1)
        denom = keep.sum(1).clamp_min(1.0)
        pooled = (h * keep).sum(1) / denom                      # mean over letters
        peak = (h.masked_fill(mask.unsqueeze(-1), -1e4)).max(1).values
        score = self.head(torch.cat([pooled, peak], dim=-1)).reshape(b, k)
        return self.first_pass_scale * first_pass + score

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def listwise_loss(logits: torch.Tensor, target: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over each n-best list.

    Only lists whose true word is present carry signal; the rest are dropped
    rather than treated as all-negative, which would teach the model to push
    every candidate down.
    """
    logits = logits.masked_fill(~valid, -1e4)
    keep = target >= 0
    if not keep.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[keep], target[keep].long())
