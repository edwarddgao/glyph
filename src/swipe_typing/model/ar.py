"""Autoregressive letter decoder over the swipe encoder trunk.

Motivation (probe, #44): under translation/scale-invariant input the letter
posterior is *coupled*-multimodal — an ``is`` gesture is exactly an ``od``
gesture, so "i goes with s, o goes with d". CTC's conditionally-independent
frames can only emit marginals, and the measured signature of that mismatch is
an incoherent cross-term (``os``) tying with the truth while the coherent twin
(``od``) loses by the implicit-LM margin. An autoregressive decoder scores
``P(s | i, gesture)`` directly, so it can express the coupling CTC cannot.

Architecture: the same dilated-TCN trunk as :class:`SwipeEncoder` produces a
64-frame memory; a small causal transformer decoder cross-attends into it and
emits letters left to right, closed by EOS. The lexicon constraint reuses the
same trie, flattened to arrays so the beam can be advanced with tensor lookups
instead of per-beam pointer chasing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderConfig, build_trunk, run_trunk
from .lexicon import Lexicon

PAD = -100  # cross_entropy's default ignore_index


@dataclass
class ARConfig(EncoderConfig):
    dec_layers: int = 2
    dec_heads: int = 4
    dec_ffn: int = 256
    max_word_len: int = 24

    @property
    def bos(self) -> int:
        return self.n_keys

    @property
    def eos(self) -> int:
        return self.n_keys + 1

    @property
    def vocab(self) -> int:
        """Alphabet plus BOS and EOS."""
        return self.n_keys + 2


class ARSwipeDecoder(nn.Module):
    """(B, T, n_input) gesture + letter prefix -> next-letter logits."""

    def __init__(self, cfg: ARConfig | None = None):
        super().__init__()
        self.cfg = cfg or ARConfig()
        c = self.cfg
        # Same input contract as SwipeEncoder: normalization lives in buffers
        # so a checkpoint is self-contained.
        self.register_buffer("input_mean", torch.zeros(c.n_input))
        self.register_buffer("input_std", torch.ones(c.n_input))
        self.input_proj = nn.Conv1d(c.n_input, c.d_model, 1)
        self.blocks, self.attn, self.attn_pos = build_trunk(c)
        self.mem_norm = nn.GroupNorm(1, c.d_model)
        self.mem_pos = nn.Parameter(torch.zeros(1, c.n_frames, c.d_model))

        self.tok_emb = nn.Embedding(c.vocab, c.d_model)
        self.tok_pos = nn.Parameter(torch.zeros(1, c.max_word_len + 1, c.d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=c.d_model, nhead=c.dec_heads, dim_feedforward=c.dec_ffn,
            dropout=c.dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=c.dec_layers)
        self.head = nn.Linear(c.d_model, c.vocab)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(
            torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        h = self.input_proj(x.transpose(1, 2))
        h = run_trunk(self.cfg, self.blocks, self.attn, self.attn_pos, h)
        return F.gelu(self.mem_norm(h)).transpose(1, 2) + self.mem_pos

    def decode_step(self, memory: torch.Tensor,
                    tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, d) memory + (B, L) prefix tokens -> (B, L, vocab) logits."""
        L = tokens.shape[1]
        h = self.tok_emb(tokens) + self.tok_pos[:, :L]
        causal = nn.Transformer.generate_square_subsequent_mask(
            L, device=tokens.device, dtype=h.dtype
        )
        out = self.decoder(h, memory, tgt_mask=causal)
        return self.head(out)

    def forward(self, x: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode_step(self.encode(x), tgt_in)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def shift_targets(targets: torch.Tensor, lengths: torch.Tensor,
                  cfg: ARConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """CTC-style flat targets -> (BOS+word, word+EOS) padded matrices."""
    parts = torch.split(targets, lengths.tolist())
    L = int(lengths.max()) + 1
    b = len(parts)
    tgt_in = torch.full((b, L), cfg.eos, dtype=torch.long)
    tgt_out = torch.full((b, L), PAD, dtype=torch.long)
    for i, p in enumerate(parts):
        n = len(p)
        tgt_in[i, 0] = cfg.bos
        tgt_in[i, 1:n + 1] = p
        tgt_out[i, :n] = p
        tgt_out[i, n] = cfg.eos
    return tgt_in, tgt_out


def ar_loss(model: ARSwipeDecoder, x: torch.Tensor, targets: torch.Tensor,
            lengths: torch.Tensor) -> torch.Tensor:
    tgt_in, tgt_out = shift_targets(targets, lengths, model.cfg)
    logits = model(x, tgt_in.to(x.device))
    return F.cross_entropy(
        logits.reshape(-1, model.cfg.vocab), tgt_out.to(x.device).reshape(-1)
    )


@torch.no_grad()
def greedy_decode(model: ARSwipeDecoder, x: torch.Tensor,
                  alphabet: str) -> list[str]:
    """Stepwise argmax decode, unconstrained -- the AR analogue of CTC greedy."""
    cfg = model.cfg
    memory = model.encode(x)
    b = x.shape[0]
    tokens = torch.full((b, 1), cfg.bos, dtype=torch.long, device=x.device)
    done = torch.zeros(b, dtype=torch.bool, device=x.device)
    for _ in range(cfg.max_word_len):
        logits = model.decode_step(memory, tokens)[:, -1]
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, cfg.eos), nxt)
        done |= nxt == cfg.eos
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        if bool(done.all()):
            break
    out = []
    for row in tokens[:, 1:].tolist():
        word = []
        for t in row:
            if t >= cfg.n_keys:
                break
            word.append(alphabet[t])
        out.append("".join(word))
    return out


@torch.no_grad()
def score_words(model: ARSwipeDecoder, x: torch.Tensor,
                words: list[str], alphabet: str) -> torch.Tensor:
    """Teacher-forced log P(word | gesture) for each (gesture, word) pair."""
    cfg = model.cfg
    char = {c: i for i, c in enumerate(alphabet)}
    targets = torch.cat(
        [torch.tensor([char[c] for c in w], dtype=torch.long) for w in words]
    )
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    tgt_in, tgt_out = shift_targets(targets, lengths, cfg)
    logits = model(x, tgt_in.to(x.device))
    lp = F.log_softmax(logits, dim=-1)
    tgt = tgt_out.to(x.device)
    mask = tgt != PAD
    picked = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask.float()).sum(-1)


class FlatTrie:
    """The lexicon trie as arrays, so a beam advances by tensor indexing.

    ``children[node, letter]`` is the child node id or -1; ``logp`` is the
    unigram log-probability where ``is_word``. Built once per lexicon
    (~1s for 300k words) and shared across every swipe.
    """

    def __init__(self, lexicon: Lexicon, alphabet: str):
        char = {c: i for i, c in enumerate(alphabet)}
        nodes = [lexicon.root]
        index = {id(lexicon.root): 0}
        for node in nodes:  # grows during iteration: BFS
            for child in node.children.values():
                index[id(child)] = len(nodes)
                nodes.append(child)
        n = len(nodes)
        self.children = np.full((n, len(alphabet)), -1, dtype=np.int64)
        self.is_word = np.zeros(n, dtype=bool)
        self.logp = np.zeros(n, dtype=np.float32)
        for i, node in enumerate(nodes):
            self.is_word[i] = node.is_word
            self.logp[i] = node.logp
            for ch, child in node.children.items():
                self.children[i, char[ch]] = index[id(child)]


@torch.no_grad()
def ar_beam(model: ARSwipeDecoder, x: torch.Tensor, trie: FlatTrie,
            alphabet: str, beam_width: int = 64
            ) -> list[list[tuple[str, float, float, int]]]:
    """Trie-constrained AR beam search over a batch of gestures.

    Returns per swipe the finished candidates as ``(word, ar_logp,
    unigram_logp, length)`` -- the same components ``beam_candidates`` exposes,
    so alpha/beta can be swept without re-running the search.
    """
    cfg = model.cfg
    device = x.device
    memory = model.encode(x)                                  # (B, T, d)
    B, T, d = memory.shape
    K = beam_width
    n_keys = cfg.n_keys

    children = torch.from_numpy(trie.children).to(device)     # (N, 26)
    is_word = torch.from_numpy(trie.is_word).to(device)
    NEG = torch.finfo(torch.float32).min

    # State. Dead beams park on node 0 with score -inf; they never win topk.
    tokens = torch.full((B, K, 1), cfg.bos, dtype=torch.long, device=device)
    node = torch.zeros(B, K, dtype=torch.long, device=device)
    score = torch.full((B, K), NEG, device=device)
    score[:, 0] = 0.0

    finished: list[list[tuple[str, float, float, int]]] = [[] for _ in range(B)]
    mem_exp = memory[:, None].expand(B, K, T, d).reshape(B * K, T, d)

    # A word of max_word_len letters takes that many extension steps plus one
    # final iteration to close on EOS, hence the +1.
    for _step in range(cfg.max_word_len + 1):
        logits = model.decode_step(mem_exp, tokens.reshape(B * K, -1))[:, -1]
        lp = F.log_softmax(logits.float(), dim=-1).reshape(B, K, -1)

        kids = children[node]                                  # (B, K, 26)
        letter_lp = score[..., None] + lp[..., :n_keys]
        letter_lp = torch.where(kids >= 0, letter_lp,
                                torch.full_like(letter_lp, NEG))

        # Close beams sitting on a word-terminal node. One device->host copy
        # per step; per-element indexing would sync once per beam.
        closing = is_word[node] & (score > NEG / 2)
        if bool(closing.any()):
            eos_lp = score + lp[..., cfg.eos]
            toks_c = tokens[closing].cpu().tolist()
            eos_c = eos_lp[closing].cpu().tolist()
            node_c = node[closing].cpu().tolist()
            rows = closing.nonzero()[:, 0].cpu().tolist()
            for b_i, toks, s, nd in zip(rows, toks_c, eos_c, node_c):
                word = "".join(alphabet[t] for t in toks[1:])
                finished[b_i].append((word, s, float(trie.logp[nd]), len(word)))

        if _step == cfg.max_word_len:
            break
        flat = letter_lp.reshape(B, K * n_keys)
        top, arg = flat.topk(K, dim=-1)
        src, letter = arg // n_keys, arg % n_keys
        score = top
        # Dead beams may gather a -1 child; clamp so the next lookup is valid
        # (their score is already NEG, so they can never win or finish).
        node = kids.reshape(B, K * n_keys).gather(1, arg).clamp_min(0)
        tokens = torch.cat(
            [tokens.gather(1, src[..., None].expand(-1, -1, tokens.shape[-1])),
             letter[..., None]], dim=-1,
        )
        if bool((score <= NEG / 2).all()):
            break

    # Dedupe by word, keep the best AR score.
    out = []
    for cands in finished:
        best: dict[str, tuple[str, float, float, int]] = {}
        for c in cands:
            if c[0] not in best or c[1] > best[c[0]][1]:
                best[c[0]] = c
        out.append(sorted(best.values(), key=lambda c: c[1], reverse=True))
    return out
