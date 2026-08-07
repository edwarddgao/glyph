import numpy as np
import pytest

torch = pytest.importorskip("torch")

from swipe_typing import features                      # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import (                        # noqa: E402
    EncoderConfig,
    SwipeCorpus,
    SwipeDataset,
    SwipeEncoder,
    collate,
    ctc_loss,
    decode,
)
from swipe_typing.model.encoder import fit_normalization  # noqa: E402
from conftest import make_swipe                          # noqa: E402


# --- key affinity -----------------------------------------------------------

def test_affinity_peaks_at_the_touched_key():
    kb = KeyboardLayout.qwerty()
    pts = np.array([kb.center("g")], dtype=np.float32)
    aff = features.key_affinity(pts, kb.centers, kb.radii)
    assert aff.shape == (1, 26)
    assert ALPHABET[int(aff[0].argmax())] == "g"
    assert aff[0].max() == pytest.approx(1.0, abs=1e-5)


def test_affinity_is_scale_invariant_across_layouts():
    """The property that makes transfer possible: a gesture at a key's center
    yields the same affinity regardless of how big that key is."""
    kb = KeyboardLayout.qwerty()
    stretched = KeyboardLayout("wide", kb.letters, kb.centers, kb.radii * 3.0)
    a = features.key_affinity(np.array([kb.center("g")]), kb.centers, kb.radii)
    b = features.key_affinity(np.array([kb.center("g")]), stretched.centers,
                              stretched.radii)
    assert a.max() == pytest.approx(b.max(), abs=1e-5)


def test_affinity_bounded():
    kb = KeyboardLayout.qwerty()
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.2, 1.2, size=(50, 2)).astype(np.float32)
    aff = features.key_affinity(pts, kb.centers, kb.radii)
    assert aff.min() >= 0.0
    assert aff.max() <= 1.0 + 1e-6
    assert np.isfinite(aff).all()


def test_affinity_handles_zero_radius():
    kb = KeyboardLayout.qwerty()
    aff = features.key_affinity(np.array([[0.5, 0.5]]), kb.centers,
                                np.zeros_like(kb.radii))
    assert np.isfinite(aff).all()


# --- kinematics -------------------------------------------------------------

def test_kinematics_excludes_position():
    sw = make_swipe("cat")
    k = features.kinematics(sw.points, sw.t, sw.aspect)
    assert k.shape == (features.N_POINTS, features.N_KINEMATIC)
    assert features.N_KINEMATIC == 6


def test_kinematics_translation_invariant():
    """No absolute position means shifting the gesture cannot change it.

    Tolerance is 1e-3 rather than exact: shifting the operands changes their
    magnitude, so the Savitzky-Golay convolution rounds differently in float32.
    Measured residual is ~2e-4 on the acceleration channels.
    """
    sw = make_swipe("cat")
    a = features.kinematics(sw.points, sw.t, sw.aspect)
    b = features.kinematics(sw.points + np.array([0.13, -0.07], np.float32),
                            sw.t, sw.aspect)
    assert np.allclose(a, b, atol=1e-3)


def test_key_scale_is_full_key_size():
    kb = KeyboardLayout.qwerty()
    scale = features.key_scale(kb.radii)
    assert scale == pytest.approx([2 / 20, 2 / 6], abs=1e-4)


def test_key_scale_floors_at_zero():
    assert (features.key_scale(np.zeros((26, 2))) > 0).all()


def test_key_units_equalize_speed_across_row_counts():
    """The reason key units exist.

    The same gesture, expressed relative to keys, must measure the same speed on
    a 3-row and a 5-row layout. In grid-height units it does not: a 5-row
    keyboard packs its rows into the same unit square, so the identical motion
    reads as slower.
    """
    three = KeyboardLayout.qwerty()
    # 5 rows x 8 cols, same unit square -- clearflow's geometry.
    five_r = np.tile(np.array([[1 / 16, 1 / 10]], np.float32), (26, 1))
    five = KeyboardLayout("five", ALPHABET, three.centers, five_r)

    n = 40
    # A gesture spanning two key-widths on each layout.
    def gest(scale_xy):
        x = np.linspace(0.0, 2 * scale_xy[0], n, dtype=np.float32)
        return np.stack([x, np.full(n, 0.5, np.float32)], axis=1)

    t = (np.arange(n) * 16).astype(np.int32)
    s3 = features.key_scale(three.radii)
    s5 = features.key_scale(five.radii)

    k3 = features.kinematics(gest(s3), t, 2.38, scale=s3)
    k5 = features.kinematics(gest(s5), t, 1.15, scale=s5)
    i = features.KINEMATIC_NAMES.index("speed")
    assert k3[:, i].mean() == pytest.approx(k5[:, i].mean(), rel=1e-3)

    # Without key units the same pair diverges.
    g3 = features.kinematics(gest(s3), t, 2.38)
    g5 = features.kinematics(gest(s5), t, 1.15)
    assert g3[:, i].mean() != pytest.approx(g5[:, i].mean(), rel=0.1)


def test_to_key_units_matches_manual_division():
    pts = np.array([[0.2, 0.5]], dtype=np.float32)
    scale = np.array([0.1, 0.3333], dtype=np.float32)
    got = features.to_key_units(pts, scale).ravel()
    assert got == pytest.approx([2.0, 1.5], abs=1e-3)


def test_kinematics_finite_on_degenerate_input():
    n = 20
    pts = np.full((n, 2), 0.5, dtype=np.float32)
    k = features.kinematics(pts, np.arange(n, dtype=np.int32) * 16, 2.0)
    assert np.isfinite(k).all()


# --- encoder ----------------------------------------------------------------

def test_encoder_shapes():
    cfg = EncoderConfig(d_model=32, dilations=(1, 2))
    m = SwipeEncoder(cfg)
    x = torch.randn(4, 64, cfg.n_input)
    out = m(x)
    assert out.shape == (4, 64, 27)
    assert cfg.blank == 26


def test_encoder_outputs_log_probabilities():
    m = SwipeEncoder(EncoderConfig(d_model=32, dilations=(1,)))
    out = m(torch.randn(2, 64, m.cfg.n_input))
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 64), atol=1e-4)
    assert (out <= 0).all()


def test_encoder_preserves_sequence_length():
    for dil in [(1,), (1, 2, 4, 8), (1, 2, 4, 8, 1, 2, 4, 8)]:
        m = SwipeEncoder(EncoderConfig(d_model=16, dilations=dil))
        assert m(torch.randn(2, 64, m.cfg.n_input)).shape[1] == 64


def test_normalization_buffers_are_applied():
    m = SwipeEncoder(EncoderConfig(d_model=16, dilations=(1,)))
    m.set_normalization(torch.full((m.cfg.n_input,), 5.0),
                        torch.full((m.cfg.n_input,), 2.0))
    assert torch.allclose(m.input_mean, torch.full((m.cfg.n_input,), 5.0))
    assert torch.allclose(m.input_std, torch.full((m.cfg.n_input,), 2.0))
    # A degenerate std must be floored, not allowed to divide by ~0.
    m.set_normalization(torch.zeros(m.cfg.n_input), torch.zeros(m.cfg.n_input))
    assert (m.input_std >= 1e-3).all()


def test_normalization_survives_state_dict_round_trip():
    """Normalization must ride with the checkpoint or inference silently
    disagrees with training."""
    a = SwipeEncoder(EncoderConfig(d_model=16, dilations=(1,)))
    a.set_normalization(torch.arange(a.cfg.n_input).float(),
                        torch.full((a.cfg.n_input,), 3.0))
    b = SwipeEncoder(EncoderConfig(d_model=16, dilations=(1,)))
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a.input_mean, b.input_mean)
    assert torch.allclose(a.input_std, b.input_std)


def test_ctc_loss_is_finite_and_differentiable():
    m = SwipeEncoder(EncoderConfig(d_model=16, dilations=(1,)))
    out = m(torch.randn(3, 64, m.cfg.n_input))
    targets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    lengths = torch.tensor([2, 2, 2], dtype=torch.long)
    loss = ctc_loss(out, targets, lengths, m.cfg.blank)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())


# --- decode -----------------------------------------------------------------

def test_greedy_decode_collapses_repeats_then_blanks():
    alphabet = "abc"
    blank = 3
    # a a blank a b  ->  "aab" collapsed to "a" + "a" (split by blank) + "b"
    path = torch.tensor([[0, 0, 3, 0, 1]])
    log_probs = torch.full((1, 5, 4), -10.0)
    for i, k in enumerate(path[0]):
        log_probs[0, i, k] = 0.0
    assert decode.greedy_decode(log_probs, blank, alphabet) == ["aab"]


def test_greedy_decode_all_blank_is_empty():
    log_probs = torch.full((1, 6, 4), -10.0)
    log_probs[0, :, 3] = 0.0
    assert decode.greedy_decode(log_probs, 3, "abc") == [""]


@pytest.mark.parametrize("a,b,d", [
    ("cat", "cat", 0), ("cat", "car", 1), ("cat", "", 3),
    ("", "cat", 3), ("hello", "helo", 1),
])
def test_edit_distance(a, b, d):
    assert decode.edit_distance(a, b) == d


def test_score():
    m = decode.score(["cat", "dog", "co"], ["cat", "dog", "cow"])
    assert m["wacc"] == pytest.approx(2 / 3)
    assert m["cer"] == pytest.approx(1 / 9)
    assert m["n"] == 3


def test_score_empty():
    assert decode.score([], [])["n"] == 0


def test_target_strings_roundtrip():
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    lengths = torch.tensor([3, 1])
    assert decode.target_strings(targets, lengths, ALPHABET) == ["abc", "d"]


def test_align_ops_all_match():
    ops = decode.align_ops("cat", "cat")
    assert [o[0] for o in ops] == ["match"] * 3


def test_align_ops_substitution():
    ops = decode.align_ops("cot", "cat")
    assert ("sub", "o", "a") in ops
    assert sum(o[0] == "sub" for o in ops) == 1


def test_align_ops_insertion_and_deletion():
    # extra char in prediction
    assert sum(o[0] == "ins" for o in decode.align_ops("caat", "cat")) == 1
    # missing from prediction
    assert sum(o[0] == "del" for o in decode.align_ops("ct", "cat")) == 1


def test_align_ops_consistent_with_edit_distance():
    for a, b in [("cat", "cot"), ("hello", "helo"), ("", "abc"),
                 ("abc", ""), ("kitten", "sitting")]:
        ops = decode.align_ops(a, b)
        n_edits = sum(o[0] != "match" for o in ops)
        assert n_edits == decode.edit_distance(a, b)


def test_edits1_contains_expected_variants():
    e = decode.edits1("cat", ALPHABET)
    assert "at" in e          # deletion
    assert "cot" in e         # substitution
    assert "cats" in e        # insertion
    assert "cat" not in e     # never itself


def test_edits1_size_and_distance():
    word = "ab"
    e = decode.edits1(word, "abc")
    assert all(decode.edit_distance(word, v) == 1 for v in e)


def test_confusion_by_length():
    out = decode.confusion_by_length(["at", "cat"], ["at", "cot"])
    assert out[2]["wacc"] == 1.0
    assert out[3]["wacc"] == 0.0


# --- dataset ----------------------------------------------------------------

def _corpus(words=("cat", "dog", "hello")):
    return SwipeCorpus.from_swipes(
        [make_swipe(w) for w in words], ALPHABET
    )


def test_dataset_item_shape():
    kb = KeyboardLayout.qwerty()
    ds = SwipeDataset(_corpus(), kb, augment_cfg=None)
    x, y = ds[0]
    assert x.shape == (features.N_POINTS, 26 + features.N_KINEMATIC)
    assert y.tolist() == [ALPHABET.index(c) for c in "cat"]


def test_dataset_no_augment_is_deterministic():
    kb = KeyboardLayout.qwerty()
    ds = SwipeDataset(_corpus(), kb, augment_cfg=None)
    assert torch.equal(ds[0][0], ds[0][0])


def test_dataset_augment_changes_input():
    from swipe_typing.augment import DEFAULT

    kb = KeyboardLayout.qwerty()
    plain = SwipeDataset(_corpus(), kb, augment_cfg=None)[0][0]
    aug = SwipeDataset(_corpus(), kb, augment_cfg=DEFAULT, seed=3)[0][0]
    assert not torch.allclose(plain, aug, atol=1e-4)


def test_corpus_filters_uncovered_words():
    # "act" covers cat; dog needs d/o/g, so it must be dropped.
    corpus = SwipeCorpus.from_swipes(
        [make_swipe("cat"), make_swipe("dog")], "act"
    )
    assert corpus.words == ["cat"]


def test_corpus_respects_max_word_len():
    corpus = SwipeCorpus.from_swipes(
        [make_swipe("cat"), make_swipe("hello")], ALPHABET, max_word_len=3
    )
    assert corpus.words == ["cat"]


def test_corpus_rejects_empty():
    with pytest.raises(ValueError):
        SwipeCorpus.from_swipes([], ALPHABET)


def test_collate_packs_ctc_targets():
    kb = KeyboardLayout.qwerty()
    ds = SwipeDataset(_corpus(), kb, augment_cfg=None)
    x, targets, lengths = collate([ds[0], ds[2]])
    assert x.shape[0] == 2
    assert lengths.tolist() == [3, 5]
    assert targets.shape[0] == 8
    assert decode.target_strings(targets, lengths, ALPHABET) == ["cat", "hello"]


def test_fit_normalization_runs():
    from torch.utils.data import DataLoader

    kb = KeyboardLayout.qwerty()
    ds = SwipeDataset(_corpus(), kb, augment_cfg=None)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    m = SwipeEncoder(EncoderConfig(d_model=16, dilations=(1,)))
    fit_normalization(m, loader)
    assert torch.isfinite(m.input_mean).all()
    assert (m.input_std > 0).all()


# --- layout reindexing ------------------------------------------------------

def test_reindex_reorders_to_target_alphabet():
    kb = KeyboardLayout.qwerty()
    shuffled = KeyboardLayout("shuf", "zyxwvutsrqponmlkjihgfedcba",
                              kb.centers[::-1].copy(), kb.radii[::-1].copy())
    fixed = shuffled.reindex(ALPHABET)
    assert fixed.letters == ALPHABET
    for ch in ALPHABET:
        assert np.allclose(fixed.center(ch), shuffled.center(ch))


def test_reindex_drops_extra_keys():
    kb = KeyboardLayout.qwerty()
    extended = KeyboardLayout(
        "plus", kb.letters + "'",
        np.vstack([kb.centers, [[0.95, 0.5]]]).astype(np.float32),
        np.vstack([kb.radii, [[0.05, 0.16]]]).astype(np.float32),
    )
    assert len(extended.reindex(ALPHABET)) == 26


def test_reindex_missing_letter_raises():
    kb = KeyboardLayout.qwerty()
    partial = KeyboardLayout("part", "abc", kb.centers[:3], kb.radii[:3])
    with pytest.raises(KeyError):
        partial.reindex(ALPHABET)


def test_encoder_accepts_any_layout_geometry():
    """A 5x8 grid must work as well as 3x10 -- the model never sees the grid."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, 1, size=(26, 2)).astype(np.float32)
    radii = np.full((26, 2), 0.06, dtype=np.float32)
    odd = KeyboardLayout("odd", ALPHABET, centers, radii)
    ds = SwipeDataset(_corpus(), odd, augment_cfg=None)
    x, _ = ds[0]
    assert x.shape == (features.N_POINTS, 32)
    assert torch.isfinite(x).all()
