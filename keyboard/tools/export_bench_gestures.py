#!/usr/bin/env python3
"""Gestures for the replay benchmark: recorded swipes, canonical coordinates,
whole sentences, so every keyboard sees byte-identical input.

    ../research/.venv/bin/python tools/export_bench_gestures.py --futo 150

Sources:
  capture  the iPhone capture study's Block A swipes (research/iphone/data),
           latest upload per (session, sentence), all sets — the same 543
           words the study scored offline
  futo     complete sentences from futo/validation (every word present,
           3–12 words), a fixed random sample

Output: Resources/bench_gestures.json
  {"sentences": [{"source", "session", "set", "tag", "words": [...],
                  "gestures": [{"x": [...], "y": [...], "t": [ms...]}, ...]}]}
"""
from __future__ import annotations

import argparse, collections, json, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYBOARD = HERE.parent
RESEARCH = KEYBOARD.parent / "research"
sys.path.insert(0, str(RESEARCH / "src"))
from swipe_typing import cache  # noqa: E402


def norm(w): return re.sub(r"[^a-z]", "", w.lower())


def capture_sentences():
    latest = {}
    for f in (RESEARCH / "iphone/data").glob("capture_*.json"):
        p = json.loads(f.read_text())
        k = (p.get("session", "anon"), p["sentence"])
        if k not in latest or p["ts"] > latest[k]["ts"]:
            latest[k] = p
    out = []
    for (sess, sent), p in sorted(latest.items()):
        gs = sorted(p["gestures"], key=lambda g: g["word_idx"])
        words = [norm(g["word"]) for g in gs]
        if not words or any(not w for w in words):
            continue
        out.append({"source": "capture", "session": sess, "set": p.get("set", 1), "tag": p.get("tag", ""),
                    "words": words,
                    "gestures": [{"x": g["x"], "y": g["y"], "t": g["t"]} for g in gs]})
    return out


def futo_sentences(n, seed=0):
    groups = collections.defaultdict(list)
    for sw in cache.read("data/canonical/futo/validation" if False else str(RESEARCH / "data/canonical/futo/validation")):
        groups[(sw.session, sw.sentence)].append(sw)
    full = []
    for (sess, sent), sws in groups.items():
        sws.sort(key=lambda s: s.word_idx)
        ref = [norm(w) for w in sent.split()]
        if len(sws) != len(ref) or not 3 <= len(ref) <= 12:
            continue
        if any(norm(s.word) != r for s, r in zip(sws, ref)):
            continue
        full.append((sess, sent, sws))
    random.Random(seed).shuffle(full)
    out = []
    for sess, sent, sws in full[:n]:
        out.append({"source": "futo", "session": sess[:12], "set": 0, "tag": "futo",
                    "words": [norm(s.word) for s in sws],
                    "gestures": [{"x": [round(float(v), 4) for v in s.x], "y": [round(float(v), 4) for v in s.y],
                                  "t": [int(v) for v in s.t]} for s in sws]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--futo", type=int, default=150)
    a = ap.parse_args()
    cap = capture_sentences()
    fu = futo_sentences(a.futo)
    data = {"sentences": cap + fu}
    path = KEYBOARD / "Resources/bench_gestures.json"
    path.write_text(json.dumps(data))
    nw = lambda ss: sum(len(s["words"]) for s in ss)
    print(f"capture: {len(cap)} sentences, {nw(cap)} words; futo: {len(fu)} sentences, {nw(fu)} words -> {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
