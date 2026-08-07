"""Trie-constrained CTC prefix beam search.

Standard CTC prefix beam search (Hannun et al. 2014), with two changes:

1.  A prefix may only be extended by a character the lexicon trie allows, so the
    decoder can never emit a non-word. On this encoder that alone addresses the
    85% of greedy errors which are not words.

2.  Only prefixes sitting on a word-terminal trie node may finish, and their
    final score may be tilted by a unigram prior (``alpha``) plus a length bonus
    (``beta``) to offset the fact that longer words accumulate more negative
    log-probability.

Why prefix beam search rather than rescoring an n-best list: CTC assigns
probability to *label sequences*, and many alignments collapse to the same
string. Summing those alignments is the whole point, and it is what the
blank/non-blank split below is doing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lexicon import Lexicon, TrieNode

NEG_INF = -float("inf")


def _logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass
class BeamConfig:
    beam_width: int = 32
    #: Skip characters whose per-frame log-prob is below this. Most frames are
    #: dominated by blank, so this prunes the inner loop hard with no measurable
    #: accuracy cost.
    prune_logp: float = -9.0
    #: Weight on the unigram word prior. Tuned on 2,000 futo/validation swipes;
    #: the whole grid alpha in [0, 1.6] x beta in [0, 2.0] spans under one point
    #: of word accuracy, so neither value is load-bearing.
    alpha: float = 0.8
    #: Per-character bonus, counteracting the bias toward short words.
    beta: float = 1.2
    top_k: int = 1


class _Beam:
    """One prefix: probability of ending in blank vs in a real label."""

    __slots__ = ("p_blank", "p_nonblank", "node")

    def __init__(self, p_blank: float, p_nonblank: float, node: TrieNode):
        self.p_blank = p_blank
        self.p_nonblank = p_nonblank
        self.node = node

    @property
    def total(self) -> float:
        return _logaddexp(self.p_blank, self.p_nonblank)


def beam_search(log_probs: np.ndarray, lexicon: Lexicon, alphabet: str,
                cfg: BeamConfig | None = None) -> list[tuple[str, float]]:
    """Decode one utterance.

    Args:
        log_probs: ``(T, C)`` log-probabilities, blank at index ``len(alphabet)``.

    Returns:
        Up to ``cfg.top_k`` ``(word, score)`` pairs, best first. Empty if the
        lexicon admits nothing -- callers should fall back to greedy.
    """
    cfg = cfg or BeamConfig()
    log_probs = np.asarray(log_probs, dtype=np.float64)
    n_labels = len(alphabet)
    blank = n_labels
    char_index = {ch: i for i, ch in enumerate(alphabet)}

    beams: dict[str, _Beam] = {"": _Beam(0.0, NEG_INF, lexicon.root)}

    for frame in log_probs:
        # Characters worth considering at this frame.
        candidates = np.flatnonzero(frame[:n_labels] > cfg.prune_logp)
        lp_blank = float(frame[blank])
        nxt: dict[str, _Beam] = {}

        for prefix, beam in beams.items():
            total = beam.total

            # (a) emit blank -- prefix unchanged, now ends in blank
            entry = nxt.get(prefix)
            if entry is None:
                entry = nxt[prefix] = _Beam(NEG_INF, NEG_INF, beam.node)
            entry.p_blank = _logaddexp(entry.p_blank, total + lp_blank)

            # (b) repeat the last label without a blank between -- collapses,
            #     so the prefix is unchanged
            if prefix:
                last = float(frame[char_index[prefix[-1]]])
                entry.p_nonblank = _logaddexp(entry.p_nonblank,
                                              beam.p_nonblank + last)

            # (c) extend, but only where the lexicon can still go
            children = beam.node.children
            if not children:
                continue
            for ci in candidates:
                ch = alphabet[ci]
                child = children.get(ch)
                if child is None:
                    continue
                score = float(frame[ci])
                extended = prefix + ch
                nb = nxt.get(extended)
                if nb is None:
                    nb = nxt[extended] = _Beam(NEG_INF, NEG_INF, child)
                if prefix and ch == prefix[-1]:
                    # A repeat only becomes two labels across a blank.
                    nb.p_nonblank = _logaddexp(nb.p_nonblank,
                                               beam.p_blank + score)
                else:
                    nb.p_nonblank = _logaddexp(nb.p_nonblank, total + score)

        if not nxt:
            break
        beams = dict(
            sorted(nxt.items(), key=lambda kv: kv[1].total, reverse=True)
            [:cfg.beam_width]
        )

    scored: list[tuple[str, float]] = []
    for prefix, beam in beams.items():
        if not prefix or not beam.node.is_word:
            continue
        score = beam.total + cfg.alpha * beam.node.logp + cfg.beta * len(prefix)
        scored.append((prefix, score))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:cfg.top_k]


def decode_batch(log_probs: np.ndarray, lexicon: Lexicon, alphabet: str,
                 cfg: BeamConfig | None = None,
                 fallback: list[str] | None = None) -> list[str]:
    """Decode a ``(B, T, C)`` batch, falling back per item when needed."""
    out = []
    for i, item in enumerate(log_probs):
        hyps = beam_search(item, lexicon, alphabet, cfg)
        if hyps:
            out.append(hyps[0][0])
        else:
            out.append(fallback[i] if fallback is not None else "")
    return out


def decode_batch_topk(log_probs: np.ndarray, lexicon: Lexicon, alphabet: str,
                      cfg: BeamConfig | None = None) -> list[list[str]]:
    """Top-k candidate words per item -- what a keyboard shows as suggestions."""
    return [[w for w, _ in beam_search(item, lexicon, alphabet, cfg)]
            for item in log_probs]
