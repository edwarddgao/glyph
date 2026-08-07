import json

import numpy as np
import pytest

from swipe_typing.sources import futo, how_we_swipe


# --- FUTO -------------------------------------------------------------------

def _futo_record(word="The", n=20, session="s1"):
    xs = np.linspace(0.45, 0.60, n)
    ys = np.linspace(0.18, 0.52, n)
    return {
        "id": 1,
        "session": session,
        "timestamp": 1724390287154,
        "word": word,
        "canvas_width": 422.0,
        "canvas_height": 170.3125,
        "orientation": "portrait-primary",
        "data": [{"t": 1724390286378 + i * 16, "x": float(x), "y": float(y)}
                 for i, (x, y) in enumerate(zip(xs, ys))],
        "sentence": "The end",
        "word_idx": 0,
        "potentially_invalid_sentence": False,
        "distance": 17.0,
    }


def test_futo_parse_record_basics():
    sw = futo.parse_record(_futo_record(), split="train")
    assert sw.word == "the"          # lowercased
    assert sw.source == "futo"
    assert sw.split == "train"
    assert len(sw) == 20
    assert sw.t[0] == 0              # rebased to zero
    assert sw.t[-1] == 19 * 16
    assert sw.aspect == pytest.approx(422.0 / 170.3125, rel=1e-6)


def test_futo_coordinates_pass_through_unscaled():
    """FUTO's canvas *is* the letter grid, so no rescaling may occur."""
    rec = _futo_record()
    sw = futo.parse_record(rec, split="train")
    assert sw.x[0] == pytest.approx(rec["data"][0]["x"], abs=1e-6)
    assert sw.y[0] == pytest.approx(rec["data"][0]["y"], abs=1e-6)


@pytest.mark.parametrize("raw,expected", [
    ("The", "the"), ("don't", "dont"), ("HELLO", "hello"),
    ("well-known", "wellknown"), ("42", ""), ("", ""),
])
def test_futo_normalize_word(raw, expected):
    assert futo.normalize_word(raw) == expected


def test_futo_record_without_letters_is_dropped():
    assert futo.parse_record(_futo_record(word="42"), split="train") is None


def test_futo_record_without_points_is_dropped():
    rec = _futo_record()
    rec["data"] = []
    assert futo.parse_record(rec, split="train") is None


def test_futo_dual_finger_record_is_skipped_not_crashed():
    """swipe-5 stores two-finger gestures as {"L": [...], "R": [...]}.

    A different input modality, not corruption -- it must be skipped cleanly
    rather than raising on ``p["x"]``.
    """
    rec = _futo_record()
    rec["data"] = {"L": [[{"x": 0.4, "y": 0.15, "t": 1}]],
                   "R": [[{"x": 0.6, "y": 0.15, "t": 1}]]}
    rec["dual_finger"] = 1
    assert futo.parse_record(rec, split="test") is None


@pytest.mark.parametrize("data", [
    None, "", [{}], [{"x": 1}], [[0.1, 0.2]], {"L": []},
])
def test_futo_malformed_data_is_skipped(data):
    rec = _futo_record()
    rec["data"] = data
    assert futo.parse_record(rec, split="test") is None


def test_futo_iter_skips_bad_lines_and_flagged(tmp_path):
    good = _futo_record()
    flagged = _futo_record(word="bad")
    flagged["potentially_invalid_sentence"] = True
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps(good) + "\n"
        + "{ not json\n"
        + "\n"
        + json.dumps(flagged) + "\n"
    )
    words = [s.word for s in futo.iter_swipes(p)]
    assert words == ["the"]

    kept = [s.word for s in futo.iter_swipes(p, keep_flagged=True)]
    assert sorted(kept) == ["bad", "the"]


def test_futo_swipe5_extra_fields_land_in_meta():
    rec = _futo_record(word="Buvo")
    rec.update({"language": "lt", "layout": "lithuanian_qwerty", "dual_finger": 0})
    sw = futo.parse_record(rec, split="train", source="futo/swipe-5")
    assert sw.meta["language"] == "lt"
    assert sw.meta["layout"] == "lithuanian_qwerty"


def test_futo_unknown_split_raises():
    with pytest.raises(KeyError):
        futo.download("swipe-2", "test")


# --- How We Swipe -----------------------------------------------------------

def test_hws_segments_gestures(hws_log):
    swipes = list(how_we_swipe.iter_swipes(hws_log, keep_flagged=True))
    # two gestures for 'it'; the stray touchend must not produce a third
    assert len(swipes) == 2
    assert all(s.word == "it" for s in swipes)
    assert all(s.source == "how_we_swipe" for s in swipes)


def test_hws_extra_trailing_columns_parsed_as_flags(hws_log):
    swipes = list(how_we_swipe.iter_swipes(hws_log, keep_flagged=True))
    assert swipes[0].flagged is False
    assert swipes[1].flagged is True   # second attempt has trailing "1"s


def test_hws_flagged_excluded_by_default(hws_log):
    swipes = list(how_we_swipe.iter_swipes(hws_log))
    assert len(swipes) == 1
    assert swipes[0].flagged is False


def test_hws_maps_into_canonical_grid(hws_log):
    """Touch-down on 'i' must land near the canonical 'i' key."""
    from swipe_typing.layout import KeyboardLayout

    sw = next(iter(how_we_swipe.iter_swipes(hws_log)))
    kb = KeyboardLayout.qwerty()
    start = np.array([sw.x[0], sw.y[0]])
    # within one key half-width/height of the 'i' center
    assert np.all(np.abs(start - kb.center("i")) < kb.radii[kb.index("i")] * 1.5)


def test_hws_vertical_rescale_applied(hws_log):
    """y must be divided by the letter-grid span, not left as keyb fraction."""
    sw = next(iter(how_we_swipe.iter_swipes(hws_log)))
    raw = 32 / 215
    assert sw.y[0] == pytest.approx(raw / how_we_swipe._GRID_SPAN, abs=1e-5)
    assert sw.y[0] > raw


def test_hws_aspect_accounts_for_partial_bottom_row(hws_log):
    sw = next(iter(how_we_swipe.iter_swipes(hws_log)))
    assert sw.aspect == pytest.approx(360 / (215 * how_we_swipe._GRID_SPAN), rel=1e-5)
    assert sw.aspect > 360 / 215


def test_hws_time_rebased(hws_log):
    sw = next(iter(how_we_swipe.iter_swipes(hws_log)))
    assert sw.t[0] == 0
    assert sw.duration_ms == 100


def test_hws_sentence_underscores_restored(hws_log):
    sw = next(iter(how_we_swipe.iter_swipes(hws_log)))
    assert sw.sentence == "it is"


def test_hws_session_is_participant_id(hws_log):
    sw = next(iter(how_we_swipe.iter_swipes(hws_log)))
    assert sw.session == "user0000000000000000000000"


def test_hws_participants(hws_log):
    people = how_we_swipe.load_participants(hws_log)
    assert people["user0000000000000000000000"]["age"] == 21
