import numpy as np
import pytest

from swipe_typing import layout
from swipe_typing.schema import Swipe


def make_swipe(word="cat", n=40, aspect=2.38, source="test", **kw):
    """A plausible swipe tracing the ideal path of ``word``."""
    trace = layout.ideal_trace(word, points_per_key=max(n // len(word), 2))
    idx = np.linspace(0, len(trace) - 1, n)
    x = np.interp(idx, np.arange(len(trace)), trace[:, 0])
    y = np.interp(idx, np.arange(len(trace)), trace[:, 1])
    t = (np.arange(n) * 16).astype(np.int32)
    return Swipe(word=word, x=x, y=y, t=t, aspect=aspect,
                 session=kw.pop("session", "s0"), source=source, **kw)


@pytest.fixture
def swipe():
    return make_swipe()


@pytest.fixture
def hws_log(tmp_path):
    """A How We Swipe log exercising the format's real quirks.

    - a clean gesture
    - a gesture with extra trailing error-flag columns (13+ fields)
    - a retry: two gestures for the same target word
    - a stray touchend with no matching touchstart
    """
    header = ("sentence timestamp keyb_width keyb_height event x_pos y_pos "
              "x_radius y_radius angle word is_err")
    kw, kh = 360, 215
    rows = [header]

    def ev(event, x, y, t, word, sent, extra=""):
        return (f"{sent} {t} {kw} {kh} {event} {x} {y} 0.5 0.5 0 {word} 0{extra}")

    # 'it' -> i is row0 col7, t is row0 col4
    rows.append(ev("touchstart", 269, 32, 1000, "it", "it_is"))
    for k in range(6):
        rows.append(ev("touchmove", 269 - k * 8, 32 + k, 1005 + k * 10, "it", "it_is"))
    rows.append(ev("touchend", 161, 32, 1100, "it", "it_is"))

    # same word again (retry), with extra trailing flag columns
    rows.append(ev("touchstart", 269, 34, 2000, "it", "it_is", extra=" 1"))
    for k in range(6):
        rows.append(ev("touchmove", 269 - k * 8, 34, 2005 + k * 10, "it", "it_is",
                       extra=" 1 1"))
    rows.append(ev("touchend", 161, 34, 2100, "it", "it_is", extra=" 1 1 1"))

    # stray touchend with no open gesture
    rows.append(ev("touchend", 100, 100, 3000, "is", "it_is"))

    d = tmp_path / "swipelogs"
    d.mkdir()
    (d / "user0000000000000000000000.log").write_text("\n".join(rows) + "\n")
    (d / "user0000000000000000000000.json").write_text('{"age": 21, "gender": "Female"}')
    return d
