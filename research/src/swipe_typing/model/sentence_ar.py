"""One decoder, two conditioning streams: the sentence-level AR decoder.

The word-level AR decoder (:mod:`.ar`) scores ``P(word | gesture)`` and leaves
sentence context to an external LM, glued on with prior algebra (#49). This
module trains the maximal alternative the design ladder called "the horizon":
a single autoregressive decoder over the *sentence's* letters, where

  - self-attention over the token history (letters + end-of-word marks of all
    preceding words) is the language-model stream, and
  - cross-attention into the current word's gesture memory is the acoustic
    stream.

Same weights serve both roles; there is no external LM, no unigram, no
interpolation weight. The architecture is deliberately identical to
:class:`~.ar.ARSwipeDecoder` — same TCN trunk, same 2-layer transformer head,
same vocabulary (EOS reinterpreted as end-of-word) — so any accuracy delta
against ``runs/ar_full`` is attributable to the conditioning scope, not
capacity. Each token cross-attends *only* to its own word's 64-frame memory
(block mask); prior words influence the present through the token stream
alone, which keeps per-swipe memories independent and the search incremental.

Known trades, priced by this experiment rather than assumed: the LM stream is
learned from the swipe corpus's ~95k prompt sentences (5 MB of text, vs
GPT-2's 40 GB), and the prior it internalizes is maximally baked in (#36-38).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .ar import ARConfig, PAD
from .data import SwipeCorpus, SwipeDataset
from .encoder import ResidualBlock


@dataclass
class SentenceARConfig(ARConfig):
    max_words: int = 24
    max_tokens: int = 160  # futo train max is 142 (letters + one EOW per word)

    @property
    def eow(self) -> int:
        """End-of-word: same token id the word-level decoder used for EOS."""
        return self.eos


class SentenceARDecoder(nn.Module):
    """(B, W, T, n_input) gestures + sentence letter prefix -> next-letter logits."""

    def __init__(self, cfg: SentenceARConfig | None = None):
        super().__init__()
        self.cfg = cfg or SentenceARConfig()
        c = self.cfg
        self.register_buffer("input_mean", torch.zeros(c.n_input))
        self.register_buffer("input_std", torch.ones(c.n_input))
        self.input_proj = nn.Conv1d(c.n_input, c.d_model, 1)
        self.blocks = nn.ModuleList(
            ResidualBlock(c.d_model, c.kernel_size, d, c.dropout)
            for d in c.dilations
        )
        self.mem_norm = nn.GroupNorm(1, c.d_model)
        self.mem_pos = nn.Parameter(torch.zeros(1, 64, c.d_model))

        self.tok_emb = nn.Embedding(c.vocab, c.d_model)
        self.tok_pos = nn.Parameter(torch.zeros(1, c.max_tokens, c.d_model))
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
        """(N, T, n_input) independent swipes -> (N, T, d) memories."""
        x = (x - self.input_mean) / self.input_std
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return F.gelu(self.mem_norm(h)).transpose(1, 2) + self.mem_pos

    def decode_step(self, memory: torch.Tensor, tokens: torch.Tensor,
                    tok_word: torch.Tensor) -> torch.Tensor:
        """Sentence-prefix tokens over per-word memories -> next-token logits.

        Args:
            memory: (B, W, T, d) per-word gesture memories, zero-padded in W.
            tokens: (B, L) token prefix (BOS, letters, EOW marks).
            tok_word: (B, L) index of the word each position is *predicting*
                (BOS predicts word 0's first letter; a word's letters predict
                within that word, closing on its EOW; each EOW predicts the
                next word's first letter). Padding rows must point at a valid
                word so no attention row is fully masked.
        Returns:
            (B, L, vocab) logits.
        """
        B, W, T, d = memory.shape
        L = tokens.shape[1]
        h = self.tok_emb(tokens) + self.tok_pos[:, :L]
        causal = nn.Transformer.generate_square_subsequent_mask(
            L, device=tokens.device, dtype=h.dtype
        )
        # Block cross-attention outside each token's own word: True = masked.
        word_of_s = (torch.arange(W * T, device=tokens.device) // T)
        blocked = tok_word[:, :, None] != word_of_s[None, None, :]  # (B, L, S)
        blocked = blocked.repeat_interleave(self.cfg.dec_heads, dim=0)
        out = self.decoder(h, memory.reshape(B, W * T, d),
                           tgt_mask=causal, memory_mask=blocked)
        return self.head(out)

    def forward(self, memory: torch.Tensor, tgt_in: torch.Tensor,
                tok_word: torch.Tensor) -> torch.Tensor:
        return self.decode_step(memory, tgt_in, tok_word)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def order_by_sentence(corpus: SwipeCorpus) -> list[list[int]]:
    """Swipe indices grouped by (session, sentence), ordered by word position.

    Same convention as the eval scripts: a "sentence" is whatever swipes of it
    survived filtering, in word_idx order.
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(len(corpus)):
        groups[(corpus.sessions[i], corpus.sentences[i])].append(i)
    out = []
    for idx in groups.values():
        idx.sort(key=lambda i: (corpus.word_idx[i]
                                if corpus.word_idx[i] >= 0 else 0))
        out.append(idx)
    return out


def sentence_tokens(word_targets: list[torch.Tensor], cfg: SentenceARConfig
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-word letter targets -> (tgt_in, tgt_out, tok_word) for one sentence.

    tgt_in:  BOS w1 EOW w2 EOW ... wN        (the final EOW is output-only)
    tgt_out: w1 EOW w2 EOW ... wN EOW
    tok_word[i] = which word position i predicts (see decode_step).
    """
    tin, tout, tword = [cfg.bos], [], []
    for wi, tgt in enumerate(word_targets):
        letters = tgt.tolist()
        tout.extend(letters + [cfg.eow])
        tword.extend([wi] * (len(letters) + 1))
        tin.extend(letters + [cfg.eow])
    tin = tin[:-1]  # drop trailing EOW: nothing left to predict after it
    return (torch.tensor(tin, dtype=torch.long),
            torch.tensor(tout, dtype=torch.long),
            torch.tensor(tword, dtype=torch.long))


class SentenceDataset(Dataset):
    """Yields one sentence: per-swipe features plus the joined token stream.

    Featurization and augmentation are delegated per swipe to
    :class:`SwipeDataset` (each gesture co-augments with its own layout copy,
    exactly as in word-level training), so the input distribution matches
    ``runs/ar_full`` and only the conditioning scope differs.
    """

    def __init__(self, corpus: SwipeCorpus, layout, cfg: SentenceARConfig,
                 augment_cfg=None, resample_mode: str = "time",
                 shape_only: bool = False):
        self.inner = SwipeDataset(corpus, layout, augment_cfg=augment_cfg,
                                  resample_mode=resample_mode,
                                  shape_only=shape_only)
        self.cfg = cfg
        self.groups = [g[:cfg.max_words] for g in order_by_sentence(corpus)]

    @property
    def seed(self) -> int:
        return self.inner.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.inner.seed = value

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        feats, tgts = zip(*(self.inner[i] for i in self.groups[idx]))
        tin, tout, tword = sentence_tokens(list(tgts), self.cfg)
        L = self.cfg.max_tokens
        return (torch.stack(feats), tin[:L], tout[:L], tword[:L],
                torch.tensor(self.groups[idx], dtype=torch.long))


def collate_sentences(batch):
    """Pad a batch of sentences; swipes stay flat with a scatter index.

    Returns:
        x: (N, T, n_input) all real swipes, concatenated.
        sent_id, word_id: (N,) where each swipe's memory goes in (B, W, ...).
        tgt_in, tok_word: (B, L) padded (pads point at word 0 — see
            decode_step's masking note).
        tgt_out: (B, L) padded with PAD for the loss to ignore.
        n_words: (B,), swipe_idx: (N,) corpus indices for bookkeeping.
    """
    feats, tins, touts, twords, idxs = zip(*batch)
    B = len(feats)
    n_words = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    x = torch.cat(list(feats))
    sent_id = torch.repeat_interleave(torch.arange(B), n_words)
    word_id = torch.cat([torch.arange(n) for n in n_words])
    L = max(t.shape[0] for t in tins)
    tgt_in = torch.zeros(B, L, dtype=torch.long)
    tgt_out = torch.full((B, L), PAD, dtype=torch.long)
    tok_word = torch.zeros(B, L, dtype=torch.long)
    for i, (a, b, c) in enumerate(zip(tins, touts, twords)):
        tgt_in[i, :len(a)] = a
        tgt_out[i, :len(b)] = b
        tok_word[i, :len(c)] = c
    return x, sent_id, word_id, tgt_in, tgt_out, tok_word, n_words, \
        torch.cat(list(idxs))


def scatter_memory(model: SentenceARDecoder, x: torch.Tensor,
                   sent_id: torch.Tensor, word_id: torch.Tensor,
                   n_sent: int) -> torch.Tensor:
    """Encode the flat swipe batch and place memories into (B, W, T, d)."""
    mem = model.encode(x)
    N, T, d = mem.shape
    W = int(word_id.max()) + 1
    out = torch.zeros(n_sent, W, T, d, device=mem.device, dtype=mem.dtype)
    out[sent_id, word_id] = mem
    return out


def sentence_loss(model: SentenceARDecoder, x, sent_id, word_id,
                  tgt_in, tgt_out, tok_word, n_words) -> torch.Tensor:
    memory = scatter_memory(model, x, sent_id, word_id, len(n_words))
    logits = model.decode_step(memory, tgt_in, tok_word)
    return F.cross_entropy(
        logits.reshape(-1, model.cfg.vocab), tgt_out.reshape(-1)
    )


@torch.no_grad()
def greedy_sentences(model: SentenceARDecoder, x, sent_id, word_id, n_words,
                     alphabet: str, max_word_len: int = 24) -> list[list[str]]:
    """Greedy decode whole sentences, feeding back decoded letters and EOWs.

    The sentence-level analogue of the word decoder's unconstrained greedy:
    context across words is the model's own output, lexicon-free.
    """
    cfg = model.cfg
    device = x.device
    B = len(n_words)
    memory = scatter_memory(model, x, sent_id, word_id, B)
    W = memory.shape[1]
    tokens = torch.full((B, 1), cfg.bos, dtype=torch.long, device=device)
    tword = torch.zeros(B, 1, dtype=torch.long, device=device)
    cur = torch.zeros(B, dtype=torch.long, device=device)   # word being decoded
    in_word = torch.zeros(B, dtype=torch.long, device=device)
    n_w = n_words.to(device)
    for _ in range(cfg.max_tokens - 1):
        logits = model.decode_step(memory, tokens, tword)[:, -1]
        nxt = logits.argmax(-1)
        # Never emit BOS; force EOW once a word hits max length.
        nxt = torch.where(nxt == cfg.bos, torch.full_like(nxt, cfg.eow), nxt)
        nxt = torch.where(in_word >= max_word_len,
                          torch.full_like(nxt, cfg.eow), nxt)
        done = cur >= n_w
        nxt = torch.where(done, torch.full_like(nxt, cfg.eow), nxt)
        is_eow = nxt == cfg.eow
        cur = cur + is_eow.long()
        in_word = torch.where(is_eow, torch.zeros_like(in_word), in_word + 1)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        tword = torch.cat([tword, cur.clamp_max(W - 1)[:, None]], dim=1)
        if bool((cur >= n_w).all()):
            break
    out = []
    for b in range(B):
        words, cur_w = [], []
        for t in tokens[b, 1:].tolist():
            if t == cfg.eow:
                words.append("".join(cur_w))
                cur_w = []
                if len(words) >= int(n_words[b]):
                    break
            elif t < cfg.n_keys:
                cur_w.append(alphabet[t])
        while len(words) < int(n_words[b]):
            words.append("".join(cur_w))
            cur_w = []
        out.append(words)
    return out
