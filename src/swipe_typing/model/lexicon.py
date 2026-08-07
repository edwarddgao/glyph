"""Word lexicon as a trie, with an optional unigram prior.

The trie does two jobs during decoding: it forbids extending a prefix down a
path no word follows (which is what kills the 85% of greedy errors that are not
words at all), and it marks which prefixes are complete words and may therefore
end a beam.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path


class TrieNode:
    __slots__ = ("children", "is_word", "logp")

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_word: bool = False
        self.logp: float = 0.0


class Lexicon:
    """Prefix trie over a vocabulary, with log unigram probabilities."""

    def __init__(self, counts: dict[str, int] | None = None,
                 smoothing: float = 0.5):
        self.root = TrieNode()
        self.size = 0
        self._counts = Counter(counts or {})
        if self._counts:
            self._build(smoothing)

    def _build(self, smoothing: float) -> None:
        total = sum(self._counts.values()) + smoothing * len(self._counts)
        for word, n in self._counts.items():
            node = self.root
            for ch in word:
                node = node.children.setdefault(ch, TrieNode())
            if not node.is_word:
                self.size += 1
            node.is_word = True
            node.logp = math.log((n + smoothing) / total)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, word: str) -> bool:
        node = self.node_for(word)
        return node is not None and node.is_word

    def node_for(self, prefix: str) -> TrieNode | None:
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def logp(self, word: str) -> float:
        node = self.node_for(word)
        return node.logp if node is not None and node.is_word else -math.inf

    def count(self, word: str) -> int:
        return self._counts.get(word, 0)

    @classmethod
    def from_words(cls, words, smoothing: float = 0.5) -> "Lexicon":
        """Build from an iterable of words; repeats become the unigram counts."""
        return cls(Counter(words), smoothing=smoothing)

    @classmethod
    def from_file(cls, path: str | Path, smoothing: float = 0.5) -> "Lexicon":
        """Read a word list, one entry per line.

        Accepts a bare word, or a word and a count in either column order --
        How We Swipe's ``wordfreq.txt`` puts the count first, most other lists
        put it second. Non-alphabetic entries are skipped.
        """
        counts: Counter[str] = Counter()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            word, n = None, 1
            for tok in parts[:2]:
                if tok.isdigit():
                    n = int(tok)
                elif word is None and tok.isalpha():
                    word = tok.lower()
            if word:
                counts[word] += n
        return cls(counts, smoothing=smoothing)

    def restricted(self, alphabet: str) -> "Lexicon":
        """Drop any word using letters outside ``alphabet``."""
        letters = set(alphabet)
        return Lexicon(
            {w: c for w, c in self._counts.items() if letters.issuperset(w)}
        )

    def counts(self) -> Counter:
        return Counter(self._counts)


#: Pseudo-count assigned to a word with unit frequency, so that general English
#: frequencies and observed in-domain counts live on a comparable integer scale.
FREQ_SCALE = 1e9


def english_counts(top_n: int = 200_000, alphabet: str | None = None,
                   scale: float = FREQ_SCALE) -> Counter:
    """Unigram counts for the ``top_n`` most frequent English words.

    Requires the optional ``wordfreq`` package. Frequencies are turned into
    pseudo-counts so they can be blended with observed corpus counts.
    """
    try:
        from wordfreq import top_n_list, word_frequency
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "english_counts needs the 'wordfreq' package: pip install wordfreq"
        ) from exc

    letters = set(alphabet) if alphabet else None
    counts: Counter[str] = Counter()
    for word in top_n_list("en", top_n):
        if not word.isalpha():
            continue
        if letters is not None and not letters.issuperset(word):
            continue
        counts[word] = max(int(word_frequency(word, "en") * scale), 1)
    return counts


def blended(general: Counter, in_domain: Counter | None = None,
            in_domain_weight: float = 0.0, smoothing: float = 0.5) -> Lexicon:
    """Union a general English vocabulary with an in-domain one.

    ``in_domain_weight`` scales observed corpus counts before adding them to the
    general frequencies. 0 keeps general English frequencies as the prior while
    still admitting in-domain-only words (which get the smoothing floor); larger
    values tilt the prior toward what the training corpus actually asked for.
    """
    counts = Counter(general)
    if in_domain:
        for word, n in in_domain.items():
            counts[word] = counts.get(word, 0) + int(n * in_domain_weight)
            counts[word] = max(counts[word], 1)
    return Lexicon(counts, smoothing=smoothing)
