import numpy as np
import pytest

from swipe_typing import augment, layout
from swipe_typing.augment import AugmentConfig
from swipe_typing.layout import KeyboardLayout
from conftest import make_swipe


def nearest_keys(points, kb):
    """Decode a trajectory to the run-compressed key sequence it passes over."""
    d = np.linalg.norm(points[:, None, :] - kb.centers[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    out = []
    for i in idx:
        ch = kb.letters[i]
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def collapse(s):
    """Drop consecutive duplicates.

    A doubled letter is geometrically indistinguishable in a swipe -- the
    gesture for "hello" dwells on 'l' once -- so the label must be collapsed
    before comparing it against a decoded key run.
    """
    return "".join(ch for i, ch in enumerate(s) if i == 0 or ch != s[i - 1])


def is_subsequence(needle, haystack):
    it = iter(haystack)
    return all(ch in it for ch in collapse(needle))


def test_co_augmentation_preserves_decodability():
    """The core invariant: moving the trajectory and layout together keeps the
    gesture decodable to its own label.

    Note this is a subsequence check, not equality. Nearest-key assignment is
    not affine-invariant -- under shear or anisotropic scale a path can graze a
    neighbouring key it previously missed. What must survive is that every
    letter of the word is still hit, in order.
    """
    kb = KeyboardLayout.qwerty()
    rng = np.random.default_rng(0)
    cfg = AugmentConfig(time_reverse=0.0)
    for word in ("cat", "hello", "swipe", "the"):
        sw = make_swipe(word, n=200)
        assert is_subsequence(word, nearest_keys(sw.points, kb))
        for _ in range(20):
            aug_sw, aug_kb = augment.augment(sw, kb, cfg, rng)
            decoded = nearest_keys(aug_sw.points, aug_kb)
            assert is_subsequence(word, decoded), f"{word!r} lost in {decoded!r}"


def test_co_augmentation_is_an_exact_change_of_frame():
    """Trajectory and layout must move by the *same* matrix, so mapping the
    augmented pair back through one inverse recovers both originals."""
    kb = KeyboardLayout.qwerty()
    rng = np.random.default_rng(7)
    sw = make_swipe("swipe", n=64)
    cfg = AugmentConfig(time_reverse=0.0)
    for _ in range(10):
        mat = augment.sample_affine(cfg, rng)
        pts = augment.apply_affine(sw.points, mat)
        centers = augment.apply_affine(kb.centers, mat)
        inv = np.linalg.inv(mat)
        assert np.allclose(augment.apply_affine(pts, inv), sw.points, atol=1e-4)
        assert np.allclose(augment.apply_affine(centers, inv), kb.centers, atol=1e-4)


def test_augmenting_trajectory_alone_breaks_the_label():
    """Contrast case -- this is what co-augmentation exists to prevent.

    Same augmented gesture, but scored against the *un*-augmented layout: the
    label stops being recoverable. That gap is the whole reason the layout
    tensor has to travel with the trajectory.
    """
    kb = KeyboardLayout.qwerty()
    rng = np.random.default_rng(1)
    sw = make_swipe("swipe", n=200)
    broken = 0
    trials = 40
    for _ in range(trials):
        aug_sw, _ = augment.augment(sw, kb, AugmentConfig(time_reverse=0.0), rng)
        if not is_subsequence("swipe", nearest_keys(aug_sw.points, kb)):
            broken += 1
    assert broken > trials // 4, f"only {broken}/{trials} broke"


def test_time_reverse_reverses_word_and_path():
    kb = KeyboardLayout.qwerty()
    sw = make_swipe("cat", n=40)
    cfg = AugmentConfig(x_scale=(1, 1), y_scale=(1, 1), shear=(0, 0),
                        rotation_deg=(0, 0), translate=(0, 0), time_reverse=1.0)
    aug_sw, _ = augment.augment(sw, kb, cfg, np.random.default_rng(0))
    assert aug_sw.word == "tac"
    assert aug_sw.x[0] == pytest.approx(sw.x[-1], abs=1e-4)
    assert aug_sw.t[0] == 0
    assert aug_sw.t[-1] == sw.t[-1]
    assert np.all(np.diff(aug_sw.t) >= 0)


def test_no_time_reverse_by_default():
    kb = KeyboardLayout.qwerty()
    sw = make_swipe("cat")
    rng = np.random.default_rng(3)
    for _ in range(10):
        aug_sw, _ = augment.augment(sw, kb, augment.DEFAULT, rng)
        assert aug_sw.word == "cat"


def test_identity_config_is_a_no_op():
    kb = KeyboardLayout.qwerty()
    sw = make_swipe("cat")
    cfg = AugmentConfig(x_scale=(1, 1), y_scale=(1, 1), shear=(0, 0),
                        rotation_deg=(0, 0), translate=(0, 0))
    aug_sw, aug_kb = augment.augment(sw, kb, cfg, np.random.default_rng(0))
    assert np.allclose(aug_sw.points, sw.points, atol=1e-5)
    assert np.allclose(aug_kb.centers, kb.centers, atol=1e-5)


def test_scaled_strength_zero_is_identity():
    cfg = augment.DEFAULT.scaled(0.0)
    assert cfg.x_scale == (1.0, 1.0)
    assert cfg.y_scale == (1.0, 1.0)
    assert cfg.shear == (0.0, 0.0)
    assert cfg.rotation_deg == (0.0, 0.0)


def test_scaled_strength_one_is_unchanged():
    cfg = augment.DEFAULT.scaled(1.0)
    assert cfg.x_scale == pytest.approx(augment.DEFAULT.x_scale)
    assert cfg.rotation_deg == pytest.approx(augment.DEFAULT.rotation_deg)


def test_radii_stay_positive_under_flips():
    kb = KeyboardLayout.qwerty()
    sw = make_swipe("cat")
    cfg = AugmentConfig(flip_x=1.0, flip_y=1.0)
    _, aug_kb = augment.augment(sw, kb, cfg, np.random.default_rng(0))
    assert (aug_kb.radii > 0).all()


def test_augment_preserves_metadata_and_length():
    kb = KeyboardLayout.qwerty()
    sw = make_swipe("cat", session="donor-7")
    aug_sw, _ = augment.augment(sw, kb, augment.DEFAULT, np.random.default_rng(0))
    assert aug_sw.session == "donor-7"
    assert aug_sw.source == sw.source
    assert len(aug_sw) == len(sw)


def test_affine_reproducible_with_seed():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = augment.sample_affine(augment.DEFAULT, rng_a)
    b = augment.sample_affine(augment.DEFAULT, rng_b)
    assert np.allclose(a, b)


def test_jitter_perturbs_points():
    kb = KeyboardLayout.qwerty()
    sw = make_swipe("cat")
    cfg = AugmentConfig(x_scale=(1, 1), y_scale=(1, 1), shear=(0, 0),
                        rotation_deg=(0, 0), translate=(0, 0), jitter=0.01)
    aug_sw, _ = augment.augment(sw, kb, cfg, np.random.default_rng(0))
    assert not np.allclose(aug_sw.points, sw.points)
    assert np.abs(aug_sw.points - sw.points).max() < 0.1
