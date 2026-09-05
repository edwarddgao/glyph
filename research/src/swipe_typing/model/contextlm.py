"""Word bigram language model with stupid backoff.

Purpose is narrow and set by the error analysis: after lexicon-constrained beam
search, 85-92% of what is left are *real-word confusions* -- ``has``->``had``,
``im``->``in``, ``val``->``call``. The gesture genuinely resembles both words, so
no lexicon separates them. What separates them is the previous word.

Deliberately a bigram, not a neural LM. The confusions are local agreement
errors, which is exactly what a bigram fixes, and it costs microseconds per
candidate rather than a forward pass.

**Train it on external text.** FUTO's train and validation splits both transcribe
Mozilla Common Voice, so an LM fitted on train-split sentences would have
memorized many validation sentences outright and the reranking gain would be
contamination rather than signal. ``from_norvig`` reads the Google Web corpus
n-gram tables, which are independent of both corpora.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

#: Weight applied each time the model backs off from bigram to unigram.
#: "Stupid backoff" (Brants et al. 2007) -- unnormalized, and at web scale it
#: matches Kneser-Ney closely while being far simpler to build.
BACKOFF = 0.4

#: Sentence-start pseudo-token, so the first word of a sentence still gets a
#: context rather than falling straight back to the unigram.
BOS = "<s>"


class ContextLM:
    """P(word | previous word) with stupid backoff to the unigram."""

    def __init__(self, unigrams: Counter | None = None,
                 bigrams: Counter | None = None, backoff: float = BACKOFF):
        self.unigrams = Counter(unigrams or {})
        self.bigrams = Counter(bigrams or {})
        self.backoff = backoff
        self.total = sum(self.unigrams.values())
        # Denominator for P(w2 | w1). Uses the unigram count of the context word
        # where available so the bigram table need not be exhaustive.
        self._ctx_total: dict[str, int] = {}
        for (w1, _), n in self.bigrams.items():
            self._ctx_total[w1] = self._ctx_total.get(w1, 0) + n

    def __len__(self) -> int:
        return len(self.unigrams)

    @property
    def n_bigrams(self) -> int:
        return len(self.bigrams)

    def logp_unigram(self, word: str) -> float:
        n = self.unigrams.get(word, 0)
        if not n or not self.total:
            # Unseen: floor below the rarest observed word rather than -inf, so
            # a candidate the LM has never seen is penalized but not excluded.
            return math.log(0.5 / max(self.total, 1))
        return math.log(n / self.total)

    def logp(self, word: str, context: str | None = None) -> float:
        """log P(word | context), backing off when the pair is unseen."""
        if context:
            n = self.bigrams.get((context, word), 0)
            if n:
                denom = self._ctx_total.get(context) or self.unigrams.get(context)
                if denom:
                    return math.log(n / denom)
        return math.log(self.backoff) + self.logp_unigram(word)

    def has_bigram(self, word: str, context: str | None) -> bool:
        return bool(context) and (context, word) in self.bigrams

    def bigram_delta(self, word: str, context: str | None) -> float:
        """Contextual evidence only: ``log P(w | ctx) - log P(w)``.

        Zero when the bigram was never observed, so a sparse table contributes
        nothing rather than silently re-applying the unigram prior. This is the
        term worth adding to a beam score that already has a frequency prior in
        it -- see :func:`rerank`.
        """
        if not self.has_bigram(word, context):
            return 0.0
        return self.logp(word, context) - self.logp_unigram(word)

    def coverage(self, pairs) -> float:
        """Fraction of ``(context, word)`` pairs the bigram table actually has.

        The ceiling on what context reranking can do: unobserved pairs
        contribute nothing under gated reranking.
        """
        pairs = list(pairs)
        if not pairs:
            return 0.0
        return sum((a, b) in self.bigrams for a, b in pairs) / len(pairs)

    def score_sequence(self, words: list[str]) -> float:
        total, prev = 0.0, BOS
        for w in words:
            total += self.logp(w, prev)
            prev = w
        return total

    @classmethod
    def from_norvig(cls, unigram_path: str | Path, bigram_path: str | Path,
                    alphabet: str | None = None,
                    max_bigrams: int | None = None) -> "ContextLM":
        """Read Norvig's ``count_1w.txt`` / ``count_2w.txt`` (tab separated).

        Both are derived from the Google Web Trillion Word Corpus, so they are
        external to every corpus in this repo.
        """
        letters = set(alphabet) if alphabet else None

        def ok(word: str) -> bool:
            return bool(word) and (letters is None or letters.issuperset(word))

        unigrams: Counter[str] = Counter()
        for line in Path(unigram_path).read_text(
                encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            word, count = parts[0].strip().lower(), parts[1].strip()
            if ok(word) and count.isdigit():
                unigrams[word] += int(count)

        bigrams: Counter[tuple[str, str]] = Counter()
        for line in Path(bigram_path).read_text(
                encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pair, count = parts[0].strip().lower().split(), parts[1].strip()
            if len(pair) != 2 or not count.isdigit():
                continue
            if ok(pair[0]) and ok(pair[1]):
                bigrams[(pair[0], pair[1])] += int(count)
                if max_bigrams and len(bigrams) >= max_bigrams:
                    break
        return cls(unigrams, bigrams)

    @classmethod
    def from_sentences(cls, sentences, alphabet: str | None = None) -> "ContextLM":
        """Fit from raw text. Only safe on text unrelated to the eval corpora."""
        letters = set(alphabet) if alphabet else None
        unigrams: Counter[str] = Counter()
        bigrams: Counter[tuple[str, str]] = Counter()
        for sentence in sentences:
            words = [w for w in
                     "".join(c if c.isalpha() or c.isspace() else " "
                             for c in sentence.lower()).split()
                     if letters is None or letters.issuperset(w)]
            prev = BOS
            for w in words:
                unigrams[w] += 1
                bigrams[(prev, w)] += 1
                prev = w
        return cls(unigrams, bigrams)


#: Default weight on the bigram term. Tuned on futo/validation; the curve is
#: flat between 0.5 and 1.0 and only turns negative past ~2.0.
DEFAULT_WEIGHT = 0.5


def rerank(candidates: list[tuple[str, float]], lm: ContextLM,
           context: str | None, weight: float = DEFAULT_WEIGHT,
           gated: bool = True) -> list[tuple[str, float]]:
    """Re-order beam candidates using context.

    ``candidates`` are ``(word, score)`` from :func:`beam.beam_search`.

    The default adds only :meth:`ContextLM.bigram_delta` -- the *contextual*
    evidence, zero where no bigram was observed. Adding raw ``log P(w | ctx)``
    instead makes accuracy strictly worse at every weight, and the reason is
    worth stating because it is easy to walk into:

    the beam score already contains a unigram prior, and a sparse bigram model
    backs off to unigram most of the time, so raw scoring mostly re-applies a
    prior that is already there. Residual errors are exactly the cases where
    word frequency *misleads*, so doubling down on frequency hurts. Measured on
    futo/validation: raw scoring prefers the wrong word on 56.8% of recoverable
    cases, while on observed bigrams alone the LM is right 74.2% of the time.

    Set ``gated=False`` to reproduce the naive version.
    """
    if not candidates:
        return []
    if gated:
        rescored = [(w, s + weight * lm.bigram_delta(w, context))
                    for w, s in candidates]
    else:
        rescored = [(w, s + weight * lm.logp(w, context)) for w, s in candidates]
    rescored.sort(key=lambda kv: kv[1], reverse=True)
    return rescored
