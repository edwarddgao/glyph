import numpy as np
import pytest

from swipe_typing import cache
from swipe_typing.schema import Swipe, is_plausible
from conftest import make_swipe


def test_round_trip_preserves_fields(tmp_path):
    original = [make_swipe("cat", session="a"), make_swipe("dog", session="b")]
    cache.write(original, tmp_path)
    back = list(cache.read(tmp_path))
    assert len(back) == 2
    for a, b in zip(original, back):
        assert a.word == b.word
        assert a.session == b.session
        assert a.source == b.source
        assert b.aspect == pytest.approx(a.aspect, rel=1e-6)
        assert np.allclose(a.x, b.x, atol=1e-6)
        assert np.allclose(a.y, b.y, atol=1e-6)
        assert np.array_equal(a.t, b.t)


def test_sharding(tmp_path):
    swipes = [make_swipe("cat", session=f"s{i}") for i in range(25)]
    files = cache.write(swipes, tmp_path, shard_rows=10)
    assert len(files) == 3
    assert len(list(cache.read(tmp_path))) == 25


def test_write_empty(tmp_path):
    assert cache.write([], tmp_path) == []
    assert list(cache.read(tmp_path)) == []


def test_stats(tmp_path):
    swipes = [make_swipe("cat", session="a"), make_swipe("cat", session="b"),
              make_swipe("dog", session="a")]
    cache.write(swipes, tmp_path)
    s = cache.stats(tmp_path)
    assert s["swipes"] == 3
    assert s["unique_words"] == 2
    assert s["sessions"] == 2


def test_read_single_file(tmp_path):
    files = cache.write([make_swipe("cat")], tmp_path)
    assert len(list(cache.read(files[0]))) == 1


# --- schema -----------------------------------------------------------------

def test_ragged_swipe_rejected():
    with pytest.raises(ValueError):
        Swipe(word="cat", x=[0, 1], y=[0], t=[0, 1], aspect=1.0,
              session="s", source="t")


def test_points_and_duration(swipe):
    assert swipe.points.shape == (len(swipe), 2)
    assert swipe.duration_ms == int(swipe.t[-1])


@pytest.mark.parametrize("kwargs", [
    dict(word=""),                                    # empty label
    dict(word="ca7"),                                 # non-alpha label
])
def test_is_plausible_rejects_bad_labels(kwargs):
    sw = make_swipe("cat")
    sw.word = kwargs["word"]
    assert not is_plausible(sw)


def test_is_plausible_rejects_too_few_points():
    sw = Swipe(word="it", x=[0.1, 0.2], y=[0.1, 0.2], t=[0, 10],
               aspect=1.0, session="s", source="t")
    assert not is_plausible(sw, min_points=4)


def test_is_plausible_rejects_wild_coordinates():
    sw = make_swipe("cat")
    sw.x = sw.x + 5.0
    assert not is_plausible(sw)


def test_is_plausible_rejects_nonpositive_duration():
    n = 10
    sw = Swipe(word="cat", x=np.linspace(0, 1, n), y=np.linspace(0, 1, n),
               t=np.zeros(n, dtype=np.int32), aspect=1.0, session="s", source="t")
    assert not is_plausible(sw)


def test_is_plausible_accepts_normal_swipe(swipe):
    assert is_plausible(swipe)


def test_is_plausible_allows_modest_overshoot():
    """Users overshoot past edge keys; that is real data, not corruption."""
    sw = make_swipe("cat")
    sw.x = sw.x + 0.1
    assert is_plausible(sw)
