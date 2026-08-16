import numpy as np

from swipe_typing.layout import KeyboardLayout, ideal_trace
from swipe_typing.schema import Swipe
from swipe_typing.trace import (
    collapse,
    is_subsequence,
    key_trace,
    nearest_keys,
    template_trace,
)

KB = KeyboardLayout.qwerty()


def _swipe_from_path(pts: np.ndarray, word: str) -> Swipe:
    t = np.arange(len(pts), dtype=np.int32) * 20
    return Swipe(word=word, x=pts[:, 0], y=pts[:, 1], t=t,
                 aspect=2.38, session="s", source="test")


def test_nearest_keys_hits_key_centers():
    idx = nearest_keys(KB.centers, KB)
    assert "".join(KB.letters[i] for i in idx) == KB.letters


def test_template_trace_contains_word():
    for word in ("the", "hello", "keyboard", "swipe", "acquaintance"):
        tr = template_trace(word, KB)
        assert is_subsequence(collapse(word), tr), (word, tr)
        # Starts and ends exactly on the word's first/last letters.
        assert tr[0] == word[0] and tr[-1] == word[-1]


def test_key_trace_roundtrips_ideal_path():
    for word in ("dog", "world", "question"):
        pts = ideal_trace(word, kb=KB)
        tr = key_trace(_swipe_from_path(pts, word), KB)
        assert is_subsequence(word, tr), (word, tr)


def test_key_trace_collapses_repeats():
    pts = ideal_trace("no", kb=KB)
    tr = key_trace(_swipe_from_path(pts, "no"), KB)
    assert all(a != b for a, b in zip(tr, tr[1:]))


def test_dwell_trace_repeats_on_pause():
    # Sit on 'a' for a while, then move to 'l': dwell mode shows the pause.
    a, l = KB.center("a"), KB.center("l")
    pts = np.vstack([np.tile(a, (10, 1)), np.linspace(a, l, 10)]).astype(np.float32)
    sw = _swipe_from_path(pts, "al")
    tr = key_trace(sw, KB, mode="dwell")
    assert tr.startswith("aaa")
    assert tr[-1] == "l"


def test_is_subsequence():
    assert is_subsequence("cat", "carrot")
    assert not is_subsequence("cat", "track")
    assert is_subsequence("", "anything")


def test_collapse():
    assert collapse("hello") == "helo"
    assert collapse("aabbcc") == "abc"
    assert collapse("swipe") == "swipe"
