import json

import numpy as np
import pytest

from swipe_typing import layout
from swipe_typing.layout import KeyboardLayout

# Verbatim excerpt of futo-org/swipe.futo.org :: swipe-5/layouts/qwerty.json.
# The canonical grid is only trustworthy if it matches the corpus's own layout
# description, so we assert against the real numbers rather than our own.
FUTO_QWERTY_KEYS = {
    "a": (0.10046728971962617, 0.5),
    "b": (0.6004672897196262, 0.8333333333333334),
    "c": (0.40046728971962614, 0.8333333333333334),
    "d": (0.30046728971962616, 0.5),
    "e": (0.25, 0.16666666666666669),
    "f": (0.40046728971962614, 0.5),
    "g": (0.5004672897196262, 0.5),
}
FUTO_RADII = (0.05, 0.16666666666666669)


@pytest.mark.parametrize("char,expected", FUTO_QWERTY_KEYS.items())
def test_key_center_matches_futo_layout(char, expected):
    got = layout.key_center(char)
    assert got == pytest.approx(expected, abs=1e-3)


def test_qwerty_layout_shape_and_radii():
    kb = KeyboardLayout.qwerty()
    assert len(kb) == 26
    assert kb.centers.shape == (26, 2)
    assert kb.radii.shape == (26, 2)
    assert kb.radii[0] == pytest.approx(FUTO_RADII, abs=1e-3)


def test_all_keys_inside_unit_square():
    kb = KeyboardLayout.qwerty()
    lo = kb.centers - kb.radii
    hi = kb.centers + kb.radii
    assert lo.min() >= -1e-6
    assert hi.max() <= 1.0 + 1e-6


def test_rows_are_distinct_and_ordered():
    ys = [layout.key_center(row[0])[1] for row in layout.ROWS]
    assert ys == sorted(ys)
    assert len(set(ys)) == len(layout.ROWS)


def test_non_letter_rejected():
    with pytest.raises(KeyError):
        layout.key_center("1")


def test_from_futo_json(tmp_path):
    spec = {
        "name": "tiny",
        "letters": "abc",
        "keys": [
            {"letter": "a", "cx": 0.1, "cy": 0.5, "rx": 0.05, "ry": 0.16},
            {"letter": "b", "cx": 0.6, "cy": 0.83, "rx": 0.05, "ry": 0.16},
            {"letter": "c", "cx": 0.4, "cy": 0.83, "rx": 0.05, "ry": 0.16},
        ],
    }
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(spec))
    kb = KeyboardLayout.from_futo_json(p)
    assert kb.name == "tiny"
    assert kb.letters == "abc"
    assert kb.center("b") == pytest.approx([0.6, 0.83], abs=1e-6)
    assert kb.covers("cab")
    assert not kb.covers("cat")


def test_layout_length_mismatch_rejected():
    with pytest.raises(ValueError):
        KeyboardLayout("bad", "abc", np.zeros((2, 2)), np.zeros((2, 2)))


def test_ideal_trace_passes_through_key_centers():
    kb = KeyboardLayout.qwerty()
    trace = layout.ideal_trace("cat", points_per_key=8)
    for ch in "cat":
        d = np.linalg.norm(trace - kb.center(ch), axis=1).min()
        assert d < 1e-5, f"trace misses key {ch!r}"


def test_ideal_trace_single_letter():
    assert layout.ideal_trace("i").shape == (1, 2)


def test_ideal_trace_rejects_empty():
    with pytest.raises(ValueError):
        layout.ideal_trace("123")
