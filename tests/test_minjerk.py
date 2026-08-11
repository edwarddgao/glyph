import numpy as np
import pytest

from swipe_typing import minjerk
from swipe_typing.layout import KeyboardLayout
from swipe_typing.schema import is_plausible

MODEL = minjerk.MinJerkModel(
    m=330.0, n=0.3, log_sigma=0.4, offset_sigma_x=0.4, offset_sigma_y=0.2,
    aspect=2.4,
)


@pytest.fixture
def kb():
    return KeyboardLayout.qwerty()


def test_generate_is_plausible(kb):
    rng = np.random.default_rng(0)
    for word in ["hello", "a", "ll", "minimum", "sequoia"]:
        sw = minjerk.generate(MODEL, word, kb, rng)
        assert sw.word == word
        assert is_plausible(sw)
        assert sw.duration_ms > 0
        assert np.all(np.diff(sw.t) >= 0)


def test_gesture_visits_keys(kb):
    # With aiming noise off, the trajectory must pass through every key
    # center of the word, in order.
    quiet = minjerk.MinJerkModel(m=330.0, n=0.3, log_sigma=0.0,
                                 offset_sigma_x=0.0, offset_sigma_y=0.0,
                                 aspect=2.4)
    rng = np.random.default_rng(1)
    sw = minjerk.generate(quiet, "wise", kb, rng)
    pts = sw.points
    last = -1
    for c in "wise":
        d = np.linalg.norm(pts - kb.center(c), axis=1)
        hit = int(np.argmin(d))
        assert d[hit] < 0.02
        assert hit >= last
        last = hit


def test_dwell_at_via_points(kb):
    # Rest-to-rest segments: speed near via points is lower than mid-segment,
    # so uniform-time samples bunch at the keys.
    quiet = minjerk.MinJerkModel(m=330.0, n=0.3, log_sigma=0.0,
                                 offset_sigma_x=0.0, offset_sigma_y=0.0,
                                 aspect=2.4)
    rng = np.random.default_rng(2)
    sw = minjerk.generate(quiet, "or", kb, rng)
    step = np.linalg.norm(np.diff(sw.points, axis=0), axis=1)
    assert step[0] < step[len(step) // 2]
    assert step[-1] < step[len(step) // 2]


def test_glides_through_interior_vias(kb):
    # Global min-jerk rounds corners: speed at an interior via is well above
    # zero, unlike the rest-to-rest concatenation which parks at every key.
    quiet = minjerk.MinJerkModel(m=330.0, n=0.3, log_sigma=0.0,
                                 offset_sigma_x=0.0, offset_sigma_y=0.0,
                                 aspect=2.4)
    rng = np.random.default_rng(4)
    sw = minjerk.generate(quiet, "was", kb, rng)
    pts, t = sw.points, sw.t.astype(float)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.diff(t)
    mid = int(np.argmin(np.linalg.norm(pts - kb.center("a"), axis=1)))
    assert step[max(mid - 1, 0)] > 0.2 * step.max()


def test_segments_profile_parks_at_vias(kb):
    quiet = minjerk.MinJerkModel(m=330.0, n=0.3, log_sigma=0.0,
                                 offset_sigma_x=0.0, offset_sigma_y=0.0,
                                 aspect=2.4, profile="segments")
    rng = np.random.default_rng(5)
    sw = minjerk.generate(quiet, "was", kb, rng)
    pts, t = sw.points, sw.t.astype(float)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.diff(t)
    mid = int(np.argmin(np.linalg.norm(pts - kb.center("a"), axis=1)))
    assert step[max(mid - 1, 0)] < 0.2 * step.max()


def test_generate_many_deterministic(kb):
    a = minjerk.generate_many(MODEL, ["hi", "there"], kb, seed=3)
    b = minjerk.generate_many(MODEL, ["hi", "there"], kb, seed=3)
    assert all(np.array_equal(x.x, y.x) for x, y in zip(a, b))


def test_model_roundtrip(tmp_path, kb):
    p = tmp_path / "m.json"
    MODEL.save(p)
    loaded = minjerk.MinJerkModel.load(p)
    assert loaded == MODEL
