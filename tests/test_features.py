import numpy as np
import pytest

from swipe_typing import features
from swipe_typing.schema import Swipe
from conftest import make_swipe


def test_encode_shape_and_dtype(swipe):
    f = features.encode(swipe)
    assert f.shape == (features.N_POINTS, features.N_FEATURES)
    assert f.dtype == np.float32
    assert len(features.FEATURE_NAMES) == features.N_FEATURES


def test_encode_is_finite_everywhere(swipe):
    assert np.isfinite(features.encode(swipe)).all()


def test_encode_batch(swipe):
    batch = features.encode_batch([swipe, make_swipe("dog")])
    assert batch.shape == (2, features.N_POINTS, features.N_FEATURES)


def test_resample_hits_endpoints():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    t = np.array([0, 100])
    out = features.resample(pts, t, n=16)
    assert out.shape == (16, 2)
    assert out[0] == pytest.approx([0.0, 0.0], abs=1e-5)
    assert out[-1] == pytest.approx([1.0, 1.0], abs=1e-5)


def test_resample_handles_duplicate_timestamps():
    """Touch logs repeat milliseconds; interpolation must stay well posed."""
    pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    t = np.array([10, 10, 10])
    out = features.resample(pts, t, n=8)
    assert np.isfinite(out).all()


def test_resample_single_point_is_repeated():
    out = features.resample(np.array([[0.3, 0.7]]), np.array([0]), n=8)
    assert out.shape == (8, 2)
    assert np.allclose(out, [0.3, 0.7])


def test_resample_empty():
    out = features.resample(np.zeros((0, 2)), np.zeros(0), n=8)
    assert out.shape == (8, 2)


def test_arclength_mode_is_uniformly_spaced():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    # Deliberately non-uniform timing; arc-length must ignore it.
    out = features.resample(pts, np.array([0, 1000]), n=9, mode="arclength")
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert steps.std() < 1e-5


def test_arclength_differs_from_time_for_varying_speed():
    n = 30
    # Move fast then slow: time- and arc-length-uniform sampling must diverge.
    x = np.concatenate([np.linspace(0, 0.9, 5), np.linspace(0.9, 1.0, n - 5)])
    y = np.zeros(n)
    sw = Swipe(word="ab", x=x, y=y, t=(np.arange(n) * 16).astype(np.int32),
               aspect=1.0, session="s", source="t")
    a = features.resample(sw.points, sw.t, n=16, mode="time")
    b = features.resample(sw.points, sw.t, n=16, mode="arclength")
    assert not np.allclose(a, b, atol=1e-3)


def test_unknown_resample_mode():
    with pytest.raises(ValueError):
        features.resample(np.zeros((3, 2)), np.arange(3), mode="nope")


def test_aspect_correct_scales_x_only():
    pts = np.array([[0.5, 0.5]])
    out = features.aspect_correct(pts, 2.0)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.5)


def test_aspect_correct_passthrough_when_unknown():
    pts = np.array([[0.5, 0.5]])
    assert np.allclose(features.aspect_correct(pts, 0.0), pts)
    assert np.allclose(features.aspect_correct(pts, float("nan")), pts)


def test_aspect_changes_curvature():
    """The reason aspect is carried at all: it changes measured geometry."""
    sw = make_swipe("cat")
    wide = features.encode(sw)
    sw_sq = make_swipe("cat", aspect=1.0)
    square = features.encode(sw_sq)
    curv_i = features.FEATURE_NAMES.index("curvature")
    assert not np.allclose(wide[:, curv_i], square[:, curv_i], atol=1e-3)


def test_straight_line_has_near_zero_curvature():
    n = 40
    sw = Swipe(word="ab", x=np.linspace(0.1, 0.9, n), y=np.full(n, 0.5),
               t=(np.arange(n) * 16).astype(np.int32), aspect=1.0,
               session="s", source="t")
    f = features.encode(sw)
    curv = f[:, features.FEATURE_NAMES.index("curvature")]
    assert np.abs(curv).max() < 1e-2


def test_speed_matches_velocity_norm(swipe):
    f = features.encode(swipe)
    vx = f[:, features.FEATURE_NAMES.index("vx")]
    vy = f[:, features.FEATURE_NAMES.index("vy")]
    speed = f[:, features.FEATURE_NAMES.index("speed")]
    assert np.allclose(np.hypot(vx, vy), speed, atol=1e-4)


def test_short_swipe_does_not_crash():
    sw = Swipe(word="i", x=[0.7, 0.71], y=[0.16, 0.17], t=[0, 20],
               aspect=2.0, session="s", source="t")
    f = features.encode(sw)
    assert f.shape == (features.N_POINTS, features.N_FEATURES)
    assert np.isfinite(f).all()


def test_encode_deterministic(swipe):
    assert np.array_equal(features.encode(swipe), features.encode(swipe))


def test_dwell_does_not_blow_up_curvature():
    """Regression: speed -> 0 at a key corner made curvature reach ~1e16 on
    real data, which silently destroys downstream feature normalization."""
    n = 60
    x = np.concatenate([np.linspace(0.1, 0.5, 20), np.full(20, 0.5),
                        np.linspace(0.5, 0.9, 20)])
    y = np.concatenate([np.linspace(0.5, 0.5, 20), np.full(20, 0.5),
                        np.linspace(0.5, 0.16, 20)])
    sw = Swipe(word="ab", x=x, y=y, t=(np.arange(n) * 16).astype(np.int32),
               aspect=2.0, session="s", source="t")
    f = features.encode(sw)
    assert np.isfinite(f).all()
    assert np.abs(f).max() <= features.CURVATURE_CLIP


def test_all_repeated_points_are_finite():
    n = 30
    sw = Swipe(word="ab", x=np.full(n, 0.5), y=np.full(n, 0.5),
               t=(np.arange(n) * 16).astype(np.int32), aspect=2.0,
               session="s", source="t")
    f = features.encode(sw)
    assert np.isfinite(f).all()
    assert np.abs(f).max() <= features.CURVATURE_CLIP


def test_curvature_bounded_by_clip(swipe):
    curv = features.encode(swipe)[:, features.FEATURE_NAMES.index("curvature")]
    assert np.abs(curv).max() <= features.CURVATURE_CLIP


def test_float32_variance_is_representable(swipe):
    """A channel large enough to overflow float32 when squared is unusable as
    a model input, even though it is technically 'finite'."""
    f = features.encode(swipe)
    assert np.isfinite(np.square(f.astype(np.float32)).sum(axis=0)).all()
