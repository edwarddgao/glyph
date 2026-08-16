import numpy as np
import pytest

from swipe_typing.geomllm import GestureDP, TokenLetterTable
from swipe_typing.layout import KeyboardLayout, ideal_trace

KB = KeyboardLayout.qwerty()


def _dp(word: str) -> GestureDP:
    pts = ideal_trace(word, kb=KB)
    t = np.arange(len(pts), dtype=np.int32) * 20
    return GestureDP(pts, t, KB)


def test_true_word_beats_lookalikes():
    dp = _dp("hello")
    truth = dp.word_cost("hello")
    for other in ("world", "help", "yellow", "hero"):
        assert truth < dp.word_cost(other), other
    # The collapsed form is geometrically near-indistinguishable — LM's job.
    assert abs(truth - dp.word_cost("helo")) < 0.01


def test_corner_cut_letter_is_cheap_not_impossible():
    # "that" passes exactly through h; "tat" must pay transit past it.
    dp = _dp("that")
    assert dp.word_cost("that") < dp.word_cost("tat")
    assert np.isfinite(dp.word_cost("tat"))


def test_incremental_matches_batch():
    dp = _dp("swipe")
    row = dp.init_row("s")
    for prev, c in zip("swipe", "wipe"):
        row = dp.extend(row, prev, c)
    assert abs(dp.final(row, "e") - dp.word_cost("swipe")) < 1e-9


def test_costs_grow_with_extra_letters():
    dp = _dp("dog")
    assert dp.word_cost("dog") < dp.word_cost("dodge")


def test_tail_bound_charges_camping():
    # A one-letter hypothesis on a long gesture must not look finished:
    # the unexplained remainder carries a nonzero lower bound.
    dp = _dp("keyboard")
    row = dp.init_row("k")
    j = int(np.argmin(row + dp.tail_bound))
    assert dp.tail_bound[j] > 1.0
    assert dp.tail_bound[-1] == 0.0


def test_token_letter_table():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        "gpt2")
    tbl = TokenLetterTable(tok)
    # " the" is a word-start token, "he" a continuation, and both map to
    # their letters.
    the = tok.encode(" the")[0]
    assert the in set(tbl.start_ids) and tbl.letters[the] == "the"
    he = tok.encode("he")[0]
    assert he in set(tbl.cont_ids) and tbl.letters[he] == "he"
    # Every letter has a bare single-char token in both groups.
    assert (tbl.start_single >= 0).all() and (tbl.cont_single >= 0).all()
    for c in range(26):
        pos = tbl.cont_single[c]
        assert tbl.letters[tbl.cont_ids[pos]] == chr(97 + c)
