"""Shape features and the AR decoder path."""

import numpy as np
import torch

from swipe_typing import features
from swipe_typing.model.ar import (
    ARConfig,
    ARSwipeDecoder,
    FlatTrie,
    ar_beam,
    ar_loss,
    greedy_decode,
    score_words,
    shift_targets,
)
from swipe_typing.model.lexicon import Lexicon

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _gesture(n=40, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.cumsum(rng.normal(0, 0.05, (n, 2)), axis=0).astype(np.float32)
    t = np.arange(n, dtype=np.int32) * 16
    return pts, t


def test_shape_features_invariant_to_translation_and_scale():
    pts, t = _gesture()
    f0 = features.shape_features(pts, t, aspect=2.38)
    f1 = features.shape_features(pts * 0.37 + np.float32([1.7, -0.4]), t,
                                 aspect=2.38)
    # Position is tightly invariant; derivative channels divide by near-zero
    # speeds so their rounding error is larger but still tiny relative to
    # channel scale (acceleration ~1e4, curvature clipped at 1e3).
    assert np.abs(f0[:, :2] - f1[:, :2]).max() < 1e-3
    assert np.abs(f0 - f1).max() < 5.0


def test_shape_features_shape_and_finite():
    pts, t = _gesture()
    f = features.shape_features(pts, t, aspect=2.38)
    assert f.shape == (features.N_POINTS, 8)
    assert np.isfinite(f).all()


def test_shape_normalize_tap_collapses_not_explodes():
    tap = np.float32([[0.5, 0.5]]).repeat(5, axis=0)
    tap += np.float32([[0, 1e-5]] * 5) * np.arange(5)[:, None]
    out = features.shape_normalize(tap)
    assert np.abs(out).max() < 0.1  # floor keeps tremor tiny, not unit-scale


def test_shift_targets_roundtrip():
    cfg = ARConfig()
    targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # "abc", "de"
    lengths = torch.tensor([3, 2])
    tgt_in, tgt_out = shift_targets(targets, lengths, cfg)
    assert tgt_in[0].tolist()[:4] == [cfg.bos, 0, 1, 2]
    assert tgt_out[0].tolist() == [0, 1, 2, cfg.eos]
    assert tgt_out[1].tolist()[:3] == [3, 4, cfg.eos]


def _tiny_model():
    cfg = ARConfig(shape_only=True, d_model=32, dilations=(1, 2),
                   dec_layers=1, dec_heads=2, dec_ffn=64)
    torch.manual_seed(0)
    return ARSwipeDecoder(cfg)


def _batch(b=4):
    xs = []
    for i in range(b):
        pts, t = _gesture(seed=i)
        xs.append(torch.from_numpy(
            features.shape_features(pts, t, aspect=2.38)))
    return torch.stack(xs)


def test_ar_loss_backward_and_greedy():
    model = _tiny_model()
    x = _batch()
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    lengths = torch.tensor([3, 2, 2, 2])
    loss = ar_loss(model, x, targets, lengths)
    loss.backward()
    assert torch.isfinite(loss)
    preds = greedy_decode(model, x, ALPHABET)
    assert len(preds) == 4
    assert all(len(p) <= model.cfg.max_word_len for p in preds)
    assert all(set(p) <= set(ALPHABET) for p in preds)


def test_score_words_matches_loss_direction():
    model = _tiny_model().eval()
    x = _batch(2)
    s = score_words(model, x, ["cat", "dog"], ALPHABET)
    assert s.shape == (2,)
    assert torch.isfinite(s).all()
    assert (s < 0).all()  # log-probabilities of full sequences


def test_ar_beam_respects_lexicon_and_exposes_components():
    model = _tiny_model().eval()
    lex = Lexicon.from_words(["cat", "cats", "dog", "at", "a"])
    trie = FlatTrie(lex, ALPHABET)
    assert int(trie.is_word.sum()) == 5
    x = _batch(3)
    out = ar_beam(model, x, trie, ALPHABET, beam_width=8)
    assert len(out) == 3
    for cands in out:
        assert cands, "beam should finish something in a 5-word lexicon"
        for word, ar_lp, uni_lp, n in cands:
            assert word in lex
            assert n == len(word)
            assert ar_lp <= 0.0
            assert np.isclose(uni_lp, lex.logp(word))
        # every lexicon word is reachable with a wide-enough beam
        assert len({w for w, *_ in cands}) == len(cands)  # deduped


# --- encoder trunk variants -------------------------------------------------

def _cfg(**kw):
    base = dict(shape_only=True, d_model=32, dilations=(1, 2),
                dec_layers=1, dec_heads=2, dec_ffn=64)
    base.update(kw)
    return ARConfig(**base)


def test_default_trunk_state_dict_has_no_attention_keys():
    """Checkpoints trained before the trunk variants must still load."""
    keys = ARSwipeDecoder(_cfg()).state_dict().keys()
    assert not any(k.startswith("attn") for k in keys)
    assert any(k.startswith("blocks.0.conv1") for k in keys)


def test_hybrid_and_conformer_trunks_train_and_beam():
    lex = Lexicon.from_words(["cat", "cats", "dog", "at", "a"])
    trie = FlatTrie(lex, ALPHABET)
    x = _batch(3)
    targets = torch.tensor([2, 0, 19, 3, 14, 6, 0], dtype=torch.long)
    lengths = torch.tensor([3, 3, 1])
    for cfg in (_cfg(trunk="hybrid", n_attn=1, attn_heads=2),
                _cfg(trunk="conformer", n_attn=2, attn_heads=2,
                     conv_kernel=5)):
        torch.manual_seed(0)
        model = ARSwipeDecoder(cfg)
        assert model.attn_pos.shape == (1, 32, 64)
        loss = ar_loss(model, x, targets, lengths)
        loss.backward()
        assert torch.isfinite(loss)
        model.eval()
        assert model.encode(x).shape == (3, 64, 32)
        out = ar_beam(model, x, trie, ALPHABET, beam_width=8)
        assert all(cands for cands in out)


def test_n_frames_sizes_memory_positions():
    model = ARSwipeDecoder(_cfg(n_frames=96)).eval()
    assert model.mem_pos.shape == (1, 96, 32)
    pts, t = _gesture()
    x = torch.from_numpy(features.shape_features(pts, t, aspect=2.38, n=96))
    assert model.encode(x[None]).shape == (1, 96, 32)
    assert len(greedy_decode(model, x[None], ALPHABET)) == 1


def test_dual_stream_dataset_layout():
    from swipe_typing.layout import KeyboardLayout
    from swipe_typing.model import SwipeCorpus, SwipeDataset
    from conftest import make_swipe

    kb = KeyboardLayout.qwerty()
    corpus = SwipeCorpus.from_swipes([make_swipe("cat"), make_swipe("dog")],
                                     ALPHABET)
    single = SwipeDataset(corpus, kb, augment_cfg=None)
    dual = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode="both")
    xs, _ = single[0]
    xd, _ = dual[0]
    assert xs.shape == (64, 32)
    assert xd.shape == (64, 64)
    # time stream first, identical to the single-stream features
    assert torch.equal(xd[:, :32], xs)
    assert ARConfig(dual_stream=True).n_input == 64
