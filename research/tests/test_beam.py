import math

import numpy as np
import pytest

from swipe_typing.layout import ALPHABET
from swipe_typing.model import decode
from swipe_typing.model.beam import BeamConfig, beam_search, decode_batch
from swipe_typing.model.lexicon import Lexicon

BLANK = len(ALPHABET)


def frames(path, confidence=0.99, n_classes=BLANK + 1):
    """Build (T, C) log-probs whose argmax follows ``path``.

    ``path`` items are letters, or None for a blank frame.
    """
    lp = np.full((len(path), n_classes), (1 - confidence) / (n_classes - 1))
    for t, sym in enumerate(path):
        idx = BLANK if sym is None else ALPHABET.index(sym)
        lp[t, idx] = confidence
    return np.log(lp / lp.sum(axis=1, keepdims=True))


# --- lexicon ----------------------------------------------------------------

def test_lexicon_membership_and_prefixes():
    lex = Lexicon.from_words(["cat", "car", "cart"])
    assert "cat" in lex
    assert "car" in lex
    assert "ca" not in lex                 # a prefix is not a word
    assert lex.node_for("ca") is not None  # but it is a live path
    assert lex.node_for("cz") is None
    assert len(lex) == 3


def test_lexicon_unigram_prior_ranks_by_frequency():
    lex = Lexicon.from_words(["the"] * 100 + ["thy"])
    assert lex.logp("the") > lex.logp("thy")
    assert lex.logp("nope") == -math.inf


def test_lexicon_probabilities_are_normalized():
    lex = Lexicon.from_words(["a", "b", "c"])
    total = sum(math.exp(lex.logp(w)) for w in "abc")
    assert total == pytest.approx(1.0, abs=1e-6)


def test_lexicon_word_is_also_a_prefix():
    lex = Lexicon.from_words(["car", "cart"])
    assert "car" in lex and "cart" in lex
    assert lex.node_for("car").is_word


def test_lexicon_from_file(tmp_path):
    p = tmp_path / "words.txt"
    p.write_text("the 500\ncat 3\n\nnot-a-word!\ndog\n")
    lex = Lexicon.from_file(p)
    assert "the" in lex and "cat" in lex and "dog" in lex
    assert lex.count("the") == 500
    assert lex.logp("the") > lex.logp("cat")


def test_lexicon_restricted_drops_out_of_alphabet_words():
    lex = Lexicon.from_words(["cat", "café"]).restricted(ALPHABET)
    assert "cat" in lex
    assert "café" not in lex


def test_empty_lexicon():
    lex = Lexicon()
    assert len(lex) == 0
    assert "cat" not in lex


# --- vocabulary blending ----------------------------------------------------

def test_blended_unions_vocabularies():
    from collections import Counter

    from swipe_typing.model.lexicon import blended

    lex = blended(Counter({"the": 1000, "cat": 10}),
                  Counter({"androscoggin": 3}), in_domain_weight=1.0)
    assert "the" in lex and "cat" in lex
    assert "androscoggin" in lex          # in-domain-only word survives
    assert lex.logp("the") > lex.logp("androscoggin")


def test_blended_weight_tilts_prior_toward_in_domain():
    from collections import Counter

    from swipe_typing.model.lexicon import blended

    general = Counter({"had": 1000, "has": 1000})
    in_domain = Counter({"has": 5000})
    flat = blended(general, in_domain, in_domain_weight=0.0)
    tilted = blended(general, in_domain, in_domain_weight=1.0)
    assert flat.logp("has") == pytest.approx(flat.logp("had"), abs=1e-9)
    assert tilted.logp("has") > tilted.logp("had")


def test_blended_without_in_domain():
    from collections import Counter

    from swipe_typing.model.lexicon import blended

    lex = blended(Counter({"cat": 5}))
    assert "cat" in lex and len(lex) == 1


def test_english_counts_shape_and_filtering():
    pytest.importorskip("wordfreq")
    from swipe_typing.model.lexicon import english_counts

    counts = english_counts(5_000, alphabet=ALPHABET)
    assert len(counts) > 1_000
    assert all(set(ALPHABET).issuperset(w) for w in counts)
    assert counts["the"] > counts["cat"]      # frequency order preserved
    assert all(c >= 1 for c in counts.values())


def test_english_counts_respects_top_n():
    pytest.importorskip("wordfreq")
    from swipe_typing.model.lexicon import english_counts

    assert len(english_counts(1_000, alphabet=ALPHABET)) <= len(
        english_counts(10_000, alphabet=ALPHABET)
    )


# --- beam search ------------------------------------------------------------

def test_decodes_a_clean_path():
    lex = Lexicon.from_words(["cat", "car"])
    lp = frames(["c", None, "a", None, "t"])
    assert beam_search(lp, lex, ALPHABET)[0][0] == "cat"


def test_never_emits_a_non_word():
    """The property that matters: 85% of greedy errors are non-words."""
    lex = Lexicon.from_words(["cat"])
    # Signal clearly spells "cot", which is not in the lexicon.
    lp = frames(["c", None, "o", None, "t"])
    assert decode.greedy_decode(
        __import__("torch").from_numpy(lp)[None], BLANK, ALPHABET
    ) == ["cot"]
    hyps = beam_search(lp, lex, ALPHABET)
    assert [w for w, _ in hyps] == ["cat"]


def test_repeated_letters_need_a_blank_between():
    lex = Lexicon.from_words(["hello"])
    lp = frames(["h", "e", "l", None, "l", "o"])
    assert beam_search(lp, lex, ALPHABET)[0][0] == "hello"


def test_repeat_without_blank_collapses():
    """Two adjacent identical frames are one label -- that is CTC's rule."""
    lex = Lexicon.from_words(["helo", "hello"])
    lp = frames(["h", "e", "l", "l", "o"])
    assert beam_search(lp, lex, ALPHABET)[0][0] == "helo"


# A lexicon-constrained decoder is *meant* to always offer its best in-vocabulary
# word -- a keyboard has to suggest something. It only comes back empty when
# pruning makes every word unreachable, so these tests use near-certain frames
# to push the off-path characters below prune_logp.
CERTAIN = 1 - 1e-7


def test_always_offers_its_best_lexicon_word():
    """With reachable alternatives, output is constrained but never empty."""
    lex = Lexicon.from_words(["cat"])
    assert beam_search(frames(["z", None, "z"]), lex, ALPHABET)[0][0] == "cat"


def test_returns_empty_when_lexicon_admits_nothing():
    lex = Lexicon.from_words(["zzz"])
    lp = frames(["c", "a", "t"], confidence=CERTAIN)
    assert beam_search(lp, lex, ALPHABET) == []


def test_top_k_returns_ranked_candidates():
    lex = Lexicon.from_words(["cat", "car", "cap"])
    lp = frames(["c", None, "a", None, "t"], confidence=0.5)
    hyps = beam_search(lp, lex, ALPHABET, BeamConfig(top_k=3))
    assert len(hyps) >= 2
    assert hyps[0][0] == "cat"
    assert all(hyps[i][1] >= hyps[i + 1][1] for i in range(len(hyps) - 1))


def test_scores_are_finite():
    lex = Lexicon.from_words(["cat", "car"])
    for w, s in beam_search(frames(["c", "a", "t"]), lex, ALPHABET,
                            BeamConfig(top_k=5)):
        assert math.isfinite(s)


def test_unigram_prior_breaks_ties():
    """With an ambiguous signal, the commoner word should win."""
    lex = Lexicon.from_words(["had"] * 500 + ["hac"])
    # deliberately ambiguous between the two final letters
    lp = np.full((5, BLANK + 1), -20.0)
    for t, ch in enumerate("ha"):
        lp[t, ALPHABET.index(ch)] = 0.0
    lp[2, BLANK] = 0.0
    lp[3, ALPHABET.index("d")] = -0.7
    lp[3, ALPHABET.index("c")] = -0.7
    lp[4, BLANK] = 0.0
    strong = beam_search(lp, lex, ALPHABET, BeamConfig(alpha=1.0))
    assert strong[0][0] == "had"


def test_beta_length_bonus_favours_longer_words():
    lex = Lexicon.from_words(["car", "cart"])
    lp = frames(["c", None, "a", None, "r", None, "t"], confidence=0.55)
    short = beam_search(lp, lex, ALPHABET, BeamConfig(alpha=0.0, beta=0.0))
    long_ = beam_search(lp, lex, ALPHABET, BeamConfig(alpha=0.0, beta=3.0))
    assert long_[0][0] == "cart"
    assert short[0][0] in {"car", "cart"}


def test_beam_width_one_still_works():
    lex = Lexicon.from_words(["cat"])
    hyps = beam_search(frames(["c", None, "a", None, "t"]), lex, ALPHABET,
                       BeamConfig(beam_width=1))
    assert hyps and hyps[0][0] == "cat"


def test_pruning_does_not_change_a_confident_result():
    lex = Lexicon.from_words(["cat", "car", "cart"])
    lp = frames(["c", None, "a", None, "t"])
    a = beam_search(lp, lex, ALPHABET, BeamConfig(prune_logp=-9.0))
    b = beam_search(lp, lex, ALPHABET, BeamConfig(prune_logp=-1e9))
    assert a[0][0] == b[0][0]


def test_decode_batch_uses_fallback_when_no_hypothesis():
    lex = Lexicon.from_words(["cat"])
    batch = np.stack([frames(["c", None, "a", None, "t"], confidence=CERTAIN),
                      frames(["z", None, "z", None, "z"], confidence=CERTAIN)])
    out = decode_batch(batch, lex, ALPHABET, fallback=["xxx", "zzz"])
    assert out == ["cat", "zzz"]


def test_decode_batch_without_fallback_yields_empty_string():
    lex = Lexicon.from_words(["cat"])
    batch = np.stack([frames(["z", None, "z"], confidence=CERTAIN)])
    assert decode_batch(batch, lex, ALPHABET) == [""]


def test_single_letter_word():
    lex = Lexicon.from_words(["i", "a"])
    assert beam_search(frames(["i"]), lex, ALPHABET)[0][0] == "i"


def test_all_blank_input_returns_nothing():
    lex = Lexicon.from_words(["cat"])
    lp = frames([None] * 6, confidence=CERTAIN)
    assert beam_search(lp, lex, ALPHABET) == []


# --- context language model -------------------------------------------------

def test_lm_bigram_beats_unigram_where_observed():
    from swipe_typing.model.contextlm import ContextLM

    lm = ContextLM.from_sentences(["i am here", "i am late", "an apple"] * 10)
    assert lm.logp("am", "i") > lm.logp("apple", "i")
    assert lm.has_bigram("am", "i")
    assert not lm.has_bigram("apple", "i")


def test_lm_backs_off_when_bigram_unseen():
    from swipe_typing.model.contextlm import BACKOFF, ContextLM

    lm = ContextLM.from_sentences(["the cat sat"] * 5)
    got = lm.logp("sat", "zebra")
    assert got == pytest.approx(math.log(BACKOFF) + lm.logp_unigram("sat"))


def test_lm_unseen_word_is_floored_not_infinite():
    from swipe_typing.model.contextlm import ContextLM

    lm = ContextLM.from_sentences(["the cat sat"])
    assert math.isfinite(lm.logp_unigram("qwertyuiop"))
    assert lm.logp_unigram("qwertyuiop") < lm.logp_unigram("cat")


def test_bigram_delta_is_zero_when_unobserved():
    """The property that makes gated reranking work: a sparse table must
    contribute nothing rather than re-applying the unigram prior."""
    from swipe_typing.model.contextlm import ContextLM

    lm = ContextLM.from_sentences(["i am here"] * 5)
    assert lm.bigram_delta("here", "zebra") == 0.0
    assert lm.bigram_delta("am", "i") != 0.0


def test_bigram_delta_matches_definition():
    from swipe_typing.model.contextlm import ContextLM

    lm = ContextLM.from_sentences(["i am here", "you are here"] * 5)
    d = lm.bigram_delta("am", "i")
    assert d == pytest.approx(lm.logp("am", "i") - lm.logp_unigram("am"))


def test_rerank_promotes_the_contextual_candidate():
    from swipe_typing.model.contextlm import ContextLM, rerank

    lm = ContextLM.from_sentences(["it has rained", "it has snowed"] * 20)
    # acoustic score slightly favours "had"; context should flip it
    cands = [("had", -2.0), ("has", -2.3)]
    assert rerank(cands, lm, "it", weight=2.0)[0][0] == "has"


def test_rerank_zero_weight_preserves_order():
    from swipe_typing.model.contextlm import ContextLM, rerank

    lm = ContextLM.from_sentences(["it has rained"] * 5)
    cands = [("had", -2.0), ("has", -2.3)]
    assert [w for w, _ in rerank(cands, lm, "it", weight=0.0)] == ["had", "has"]


def test_rerank_gated_ignores_unobserved_context():
    from swipe_typing.model.contextlm import ContextLM, rerank

    lm = ContextLM.from_sentences(["it has rained"] * 5)
    cands = [("had", -2.0), ("has", -2.3)]
    # no bigram for either word after "zebra" -> gated must not reorder
    assert [w for w, _ in rerank(cands, lm, "zebra", weight=5.0)] == ["had", "has"]


def test_rerank_empty():
    from swipe_typing.model.contextlm import ContextLM, rerank

    assert rerank([], ContextLM.from_sentences(["a b"]), "a") == []


def test_lm_coverage():
    from swipe_typing.model.contextlm import ContextLM

    lm = ContextLM.from_sentences(["i am here"] * 3)
    assert lm.coverage([("i", "am")]) == 1.0
    assert lm.coverage([("zebra", "quux")]) == 0.0
    assert lm.coverage([]) == 0.0


def test_never_emits_a_zero_probability_candidate():
    """Extending by a repeated letter draws on p_blank, which is -inf for a
    prefix that has never ended in a blank. Such beams are impossible, not
    merely unlikely, and a -inf score NaNs any downstream model that consumes
    it as a feature."""
    lex = Lexicon.from_words(["ay", "ayy", "ayyy", "by"])
    lp = frames(["b", "y"], confidence=0.6)
    for w, s in beam_search(lp, lex, ALPHABET, BeamConfig(top_k=8)):
        assert math.isfinite(s), f"{w} scored {s}"


def test_all_scores_finite_across_random_inputs():
    rng = np.random.default_rng(0)
    lex = Lexicon.from_words(["aa", "aaa", "ab", "abb", "abbb", "ba", "b"])
    for _ in range(40):
        lp = np.log(rng.dirichlet(np.ones(BLANK + 1), size=12))
        for _, s in beam_search(lp, lex, ALPHABET, BeamConfig(top_k=8)):
            assert math.isfinite(s)
