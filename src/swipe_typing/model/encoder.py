"""Layout-agnostic swipe encoder.

A dilated temporal convolutional network over 64 frames. Per frame it consumes

    [ key affinity (K) | kinematics (6) ]

and emits logits over ``K + 1`` symbols (the alphabet plus a CTC blank).

The design constraint is that the model must never learn *one* keyboard. It sees
no absolute coordinates: position enters only as affinity to the keys of
whatever layout is supplied at runtime, expressed in each key's own half-extents.
Swap in azerty and the affinities stay on the same scale, so the encoder
transfers without retraining. ``scripts/train_encoder.py --eval-layout`` tests
exactly that.

CTC is the right objective here because a gesture crosses its target keys in
order -- alignment is monotonic -- and blanks absorb the intermediate keys the
path passes through on the way. It also collapses repeats for free, which
matches the physics: "hello" dwells on 'l' once.

Size defaults to ~600K parameters, in the range FUTO reports for an on-device
encoder.

The trunk is configurable (#83). ``trunk="tcn"`` is the dilated-conv stack
every checkpoint before the encoder sweep used, with unchanged state-dict
keys; ``"hybrid"`` appends self-attention layers after the convs; and
``"conformer"`` replaces them with conformer blocks (macaron FFN, MHSA,
depthwise conv). At matched parameters the conformer was the only variant
to lift the beam on both corpora and the only one to move the cross-corpus
n-best ceiling; the hybrid and a dual time/arclength input stream were nulls
or losses. Frames per gesture (``n_frames``) is a parameter too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    n_keys: int = 26
    n_kinematic: int = 6
    d_model: int = 96
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2)
    dropout: float = 0.1
    #: Ablation: the 8-channel translation/scale-invariant input of
    #: ``features.shape_features`` replaces the affinity block. The output
    #: alphabet is unchanged; only the input width differs.
    shape_only: bool = False
    #: Frames per gesture. The featurizer resamples every swipe to this many
    #: points; the AR head's memory positions are sized to it.
    n_frames: int = 64
    #: Trunk family. ``"tcn"`` is the dilated-conv stack (every checkpoint
    #: before the encoder sweep); ``"hybrid"`` appends ``n_attn``
    #: self-attention layers after the conv blocks; ``"conformer"`` replaces
    #: the conv blocks with ``n_attn`` conformer blocks (macaron FFN, MHSA,
    #: depthwise conv). State-dict keys for ``"tcn"`` are unchanged.
    trunk: str = "tcn"
    n_attn: int = 0
    attn_heads: int = 4
    #: Attention FFN width; 0 means ``2 * d_model``.
    attn_ffn: int = 0
    #: Depthwise kernel of the conformer conv module.
    conv_kernel: int = 7
    #: Two front-end streams: the time-uniform features (the default) and the
    #: same features on the arclength-uniform resampling, concatenated per
    #: frame. Doubles ``n_input``.
    dual_stream: bool = False

    @property
    def n_input(self) -> int:
        if self.shape_only:
            return 8
        n = self.n_keys + self.n_kinematic
        return 2 * n if self.dual_stream else n

    @property
    def n_output(self) -> int:
        """Alphabet plus one CTC blank."""
        return self.n_keys + 1

    @property
    def blank(self) -> int:
        return self.n_keys


class ResidualBlock(nn.Module):
    """Pre-norm dilated conv block. Padding keeps the sequence length fixed."""

    def __init__(self, d: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.norm1 = nn.GroupNorm(1, d)
        self.conv1 = nn.Conv1d(d, d, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, d)
        self.conv2 = nn.Conv1d(d, d, kernel_size, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.drop(h)
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + self.drop(h)


class AttnBlock(nn.Module):
    """Pre-norm transformer encoder layer over frames; (B, d, T) in and out."""

    def __init__(self, d: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.transpose(1, 2)).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Macaron FFN / MHSA / depthwise-conv module / macaron FFN, pre-norm.

    Gulati et al. 2020 in miniature: the conv module keeps the TCN's local
    inductive bias, attention adds the global view. (B, d, T) in and out.
    """

    def __init__(self, d: int, heads: int, ffn: int, kernel: int,
                 dropout: float):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d))
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.conv_norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=(kernel - 1) // 2, groups=d)
        self.dw_norm = nn.GroupNorm(1, d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, d))
        self.out_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                                   # (B, T, d)
        h = h + 0.5 * self.drop(self.ff1(h))
        a = self.attn_norm(h)
        h = h + self.drop(self.attn(a, a, a, need_weights=False)[0])
        c = self.conv_norm(h).transpose(1, 2)                   # (B, d, T)
        c = F.glu(self.pw1(c), dim=1)
        c = self.pw2(F.silu(self.dw_norm(self.dw(c))))
        h = h + self.drop(c.transpose(1, 2))
        h = h + 0.5 * self.drop(self.ff2(h))
        return self.out_norm(h).transpose(1, 2)


def build_trunk(cfg: EncoderConfig) -> tuple[nn.ModuleList, nn.ModuleList,
                                             nn.Parameter | None]:
    """``(blocks, attn, attn_pos)`` for a config.

    Kept as three attributes rather than one module so the ``"tcn"`` trunk's
    state dict is byte-identical to every checkpoint trained before the
    attention variants existed: ``blocks.*`` only, no ``attn*`` keys.
    """
    c = cfg
    ffn = c.attn_ffn or 2 * c.d_model
    if c.trunk == "tcn":
        blocks = nn.ModuleList(
            ResidualBlock(c.d_model, c.kernel_size, d, c.dropout)
            for d in c.dilations
        )
        attn = nn.ModuleList()
    elif c.trunk == "hybrid":
        blocks = nn.ModuleList(
            ResidualBlock(c.d_model, c.kernel_size, d, c.dropout)
            for d in c.dilations
        )
        attn = nn.ModuleList(
            AttnBlock(c.d_model, c.attn_heads, ffn, c.dropout)
            for _ in range(c.n_attn)
        )
    elif c.trunk == "conformer":
        blocks = nn.ModuleList(
            ConformerBlock(c.d_model, c.attn_heads, ffn, c.conv_kernel,
                           c.dropout)
            for _ in range(c.n_attn)
        )
        attn = nn.ModuleList()
    else:
        raise ValueError(f"unknown trunk {c.trunk!r}")
    uses_attn = c.trunk == "conformer" or len(attn) > 0
    attn_pos = (nn.Parameter(torch.zeros(1, c.d_model, c.n_frames))
                if uses_attn else None)
    return blocks, attn, attn_pos


def run_trunk(cfg: EncoderConfig, blocks: nn.ModuleList, attn: nn.ModuleList,
              attn_pos: nn.Parameter | None, h: torch.Tensor) -> torch.Tensor:
    """(B, d, T) -> (B, d, T). Position is added once, before the first
    attention-bearing block; conv blocks need none."""
    if cfg.trunk == "conformer":
        h = h + attn_pos
    for block in blocks:
        h = block(h)
    if len(attn) > 0:
        h = h + attn_pos
        for block in attn:
            h = block(h)
    return h


class SwipeEncoder(nn.Module):
    """(B, T, n_input) -> (B, T, n_output) log-probabilities."""

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        c = self.cfg
        # Input channels arrive on wildly different scales -- affinities live in
        # [0, 1] while curvature has std ~160. Normalization is carried as
        # buffers rather than applied in the dataset so that a saved checkpoint
        # is self-contained and inference cannot silently disagree with training.
        self.register_buffer("input_mean", torch.zeros(c.n_input))
        self.register_buffer("input_std", torch.ones(c.n_input))
        self.input_proj = nn.Conv1d(c.n_input, c.d_model, 1)
        self.blocks, self.attn, self.attn_pos = build_trunk(c)
        self.out_norm = nn.GroupNorm(1, c.d_model)
        self.head = nn.Conv1d(c.d_model, c.n_output, 1)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(
            torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        h = self.input_proj(x.transpose(1, 2))
        h = run_trunk(self.cfg, self.blocks, self.attn, self.attn_pos, h)
        h = self.head(F.gelu(self.out_norm(h)))
        return F.log_softmax(h.transpose(1, 2), dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def fit_normalization(model: SwipeEncoder, loader, max_batches: int = 40) -> None:
    """Estimate per-channel mean/std from a sample of the training loader."""
    total = torch.zeros(model.cfg.n_input, dtype=torch.float64)
    total_sq = torch.zeros(model.cfg.n_input, dtype=torch.float64)
    count = 0
    for i, (x, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        flat = x.reshape(-1, x.shape[-1]).double()
        total += flat.sum(0)
        total_sq += flat.square().sum(0)
        count += flat.shape[0]
    mean = total / max(count, 1)
    var = (total_sq / max(count, 1) - mean.square()).clamp_min(0)
    model.set_normalization(mean.float(), var.sqrt().float())


def ctc_loss(log_probs: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor, blank: int) -> torch.Tensor:
    """CTC over a fixed-length input.

    ``log_probs`` is (B, T, C); ``torch.nn.functional.ctc_loss`` wants (T, B, C).
    Every swipe is resampled to the same T, so input lengths are constant.
    """
    b, t, _ = log_probs.shape
    input_lengths = torch.full((b,), t, dtype=torch.long, device=log_probs.device)
    return F.ctc_loss(
        log_probs.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )
