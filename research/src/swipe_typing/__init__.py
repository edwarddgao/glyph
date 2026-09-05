"""Normalized loaders and features for public swipe/gesture-typing corpora."""

from . import augment, cache, features, layout
from .layout import KeyboardLayout, ideal_trace, key_center
from .schema import Swipe, is_plausible
from .sources import futo, how_we_swipe

__all__ = [
    "KeyboardLayout",
    "Swipe",
    "augment",
    "cache",
    "features",
    "futo",
    "how_we_swipe",
    "ideal_trace",
    "is_plausible",
    "key_center",
    "layout",
]

__version__ = "0.1.0"
