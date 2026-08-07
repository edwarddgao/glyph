"""Greedy CTC decoding and encoder-quality metrics.

Deliberately lexicon-free. A trie-constrained beam search would score much
higher, but it would also mask what the encoder is actually doing -- a strong
language model can carry a weak encoder. These metrics measure the encoder
alone; the lexicon belongs in the decoding layer built on top.
"""

from __future__ import annotations

import numpy as np
import torch


def greedy_decode(log_probs: torch.Tensor, blank: int, alphabet: str) -> list[str]:
    """Collapse the argmax path: drop repeats, then drop blanks."""
    ids = log_probs.argmax(dim=-1).detach().cpu().numpy()
    out = []
    for row in ids:
        prev = -1
        chars = []
        for k in row:
            if k != prev and k != blank:
                chars.append(alphabet[k])
            prev = k
        out.append("".join(chars))
    return out


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align_ops(pred: str, ref: str) -> list[tuple[str, str, str]]:
    """Levenshtein backtrace as ``(op, pred_char, ref_char)`` triples.

    ``op`` is one of ``match``, ``sub``, ``ins`` (extra char in ``pred``), or
    ``del`` (missing from ``pred``). Used to find *which* letters the encoder
    confuses, rather than just how many.
    """
    n, m = len(pred), len(ref)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (pred[i - 1] != ref[j - 1]))

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (pred[i - 1] != ref[j - 1]):
            ops.append(("match" if pred[i - 1] == ref[j - 1] else "sub",
                        pred[i - 1], ref[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("ins", pred[i - 1], ""))
            i -= 1
        else:
            ops.append(("del", "", ref[j - 1]))
            j -= 1
    return ops[::-1]


def edits1(word: str, alphabet: str) -> set[str]:
    """Every string within edit distance 1 of ``word``."""
    out = set()
    for i in range(len(word) + 1):
        head, tail = word[:i], word[i:]
        if tail:
            out.add(head + tail[1:])                       # delete
            for c in alphabet:
                out.add(head + c + tail[1:])               # substitute
        for c in alphabet:
            out.add(head + c + tail)                       # insert
    out.discard(word)
    return out


def score(predictions: list[str], targets: list[str]) -> dict[str, float]:
    """Character error rate and exact-match word accuracy."""
    if not targets:
        return {"cer": float("nan"), "wacc": float("nan"), "n": 0}
    dist = sum(edit_distance(p, t) for p, t in zip(predictions, targets))
    chars = sum(len(t) for t in targets)
    exact = sum(p == t for p, t in zip(predictions, targets))
    return {
        "cer": dist / max(chars, 1),
        "wacc": exact / len(targets),
        "n": len(targets),
    }


@torch.no_grad()
def evaluate(model, loader, device, alphabet: str, max_batches: int | None = None,
             collect: int = 0) -> dict:
    """Run greedy decoding over a loader and return metrics."""
    model.eval()
    blank = model.cfg.blank
    preds: list[str] = []
    refs: list[str] = []
    samples: list[tuple[str, str]] = []

    for i, (x, targets, lengths) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        log_probs = model(x.to(device))
        batch_preds = greedy_decode(log_probs, blank, alphabet)

        offset = 0
        batch_refs = []
        for n in lengths.tolist():
            idx = targets[offset:offset + n].tolist()
            batch_refs.append("".join(alphabet[k] for k in idx))
            offset += n

        preds.extend(batch_preds)
        refs.extend(batch_refs)
        if len(samples) < collect:
            samples.extend(zip(batch_refs, batch_preds))

    out = score(preds, refs)
    if collect:
        out["samples"] = samples[:collect]
    return out


def target_strings(targets: torch.Tensor, lengths: torch.Tensor,
                   alphabet: str) -> list[str]:
    """Unpack a concatenated CTC target tensor back into words."""
    out, offset = [], 0
    for n in lengths.tolist():
        idx = targets[offset:offset + n].tolist()
        out.append("".join(alphabet[k] for k in idx))
        offset += n
    return out


def confusion_by_length(predictions: list[str], targets: list[str]) -> dict[int, dict]:
    """Break accuracy down by word length -- short words are the hard case."""
    buckets: dict[int, list[tuple[str, str]]] = {}
    for p, t in zip(predictions, targets):
        buckets.setdefault(len(t), []).append((p, t))
    return {
        n: score([p for p, _ in v], [t for _, t in v])
        for n, v in sorted(buckets.items())
    }


def as_numpy(log_probs: torch.Tensor) -> np.ndarray:
    return log_probs.detach().cpu().numpy()
