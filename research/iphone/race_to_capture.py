#!/usr/bin/env python3
"""Turn SwipeRacer records into capture-shaped files, and summarize the players.

The app's SwipeRacer game posts one `race_*.json` per finished sentence: the
prompted sentence, and every swipe attempted on each word (canonical touches,
the word it was prompted for, the attempt number, whether it traced the word —
the decoder-independent `GestureTrace` cost, #81's label-filter rule — and what
the shipped decoder read: first-pass list, fused choice, `decoder_right`). This writes one `capture_*_race.json`
per sentence in Block A's shape (kind "capture", one gesture per word carrying
word/word_idx) — the accepted attempt when there is one, else the last attempt
— so decode_capture.py, fused_rescore.py, adapt_user.py and the bench exporter
consume race data unchanged. Extra fields on each gesture (`attempt`,
`accepted`, `n_attempts`) are kept for filtering.

It also prints, per player, what the game measured directly: words, first-swipe
trace rate (did the finger trace the word), the shipped stack's top-1 on those
first swipes (`decoder_right`, the per-person accuracy read the benchmark
needs), swipes per word and wpm.

    .venv/bin/python iphone/race_to_capture.py [--data iphone/data] [--all-attempts]

--all-attempts writes every attempt (one gesture per attempt, all labeled with
the prompted word) into a separate `capture_*_race_all.json`, for training.

Single-letter words ("i", "a") are tapped in the game, as on any keyboard; the
taps are recorded (`input: "tap"`) but are not swipes and appear in neither the
files nor the swipe columns. Which recorded swipes train is decided later, on
the stored `trace_cost` — the game's cut of 6 is #81's default, not a decision.
"""
from __future__ import annotations

import argparse, json, re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--all-attempts", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    races = sorted(data.glob("race_*.json"))
    if not races:
        print("no race_*.json in", data); return
    per = defaultdict(lambda: dict(sent=0, words=0, first=0, dec=0, attempts=0, skipped=0, taps=0, secs=0.0, chars=0, nick="", device="", lm=None))
    written = 0
    for f in races:
        p = json.loads(f.read_text())
        if p.get("kind") != "race":
            continue
        sess = p.get("session", "anon")
        s = per[sess]
        s["sent"] += 1; s["nick"] = p.get("nick") or s["nick"]; s["device"] = p.get("device", ""); s["lm"] = p.get("lm")
        s["secs"] += p.get("ms", 0) / 1000; s["chars"] += len(p["sentence"])
        by_word = defaultdict(list)
        for g in p.get("gestures", []):
            if g.get("input") == "tap":          # single-letter words are tapped; no path, not swipe data
                s["taps"] += 1; continue
            by_word[g["word_idx"]].append(g)
        gestures, all_g = [], []
        for w in p.get("words", []):
            if len(re.sub(r"[^a-z]", "", w["word"].lower())) == 1:
                continue                          # tapped word: excluded from swipe counts and files
            i = w["word_idx"]; atts = sorted(by_word.get(i, []), key=lambda g: g["attempt"])
            s["words"] += 1; s["attempts"] += len(atts); s["skipped"] += bool(w.get("skipped"))
            s["first"] += bool(atts) and atts[0]["accepted"]
            s["dec"] += bool(atts) and atts[0].get("decoder_right", atts[0]["accepted"])
            if not atts:
                continue
            pick = next((g for g in atts if g["accepted"]), atts[-1])
            def shape(g):
                return {"word": w["word"], "word_idx": i, "x": g["x"], "y": g["y"], "t": g["t"], "aspect": g.get("aspect", p.get("grid", {}).get("aspect", 2.44)),
                        "attempt": g["attempt"], "accepted": g["accepted"], "n_attempts": len(atts),
                        "trace_cost": g.get("trace_cost"), "decoder_right": g.get("decoder_right"),
                        "first_pass": g.get("first_pass", []), "fused": g.get("fused", "")}
            gestures.append(shape(pick))
            all_g.extend(shape(g) for g in atts)
        base = {"kind": "capture", "session": sess, "ts": p["ts"], "sentence": p["sentence"], "tag": p.get("tag", ""),
                "set": p.get("set", 1), "ua": p.get("ua", "swipe-app-race"), "device": p.get("device", ""), "source_kind": "race"}
        (data / f"capture_{sess}_{p['ts']}_race.json").write_text(json.dumps(dict(base, gestures=gestures), indent=1))
        if a.all_attempts:
            (data / f"capture_{sess}_{p['ts']}_race_all.json").write_text(json.dumps(dict(base, gestures=all_g), indent=1))
        written += 1
    print(f"{written} race sentences -> capture_*_race.json in {data}\n")
    print(f"{'player':<12} {'device':<12} {'sent':>4} {'words':>5} {'traced 1st':>10} {'decoder 1st':>11} {'swipes/word':>11} {'skipped':>7} {'wpm':>5}  lm")
    for sess, s in sorted(per.items(), key=lambda kv: -kv[1]["words"]):
        wpm = s["chars"] / 5 / (s["secs"] / 60) if s["secs"] else 0
        name = (s["nick"] or sess)[:12]
        print(f"{name:<12} {s['device'][:12]:<12} {s['sent']:>4} {s['words']:>5} {s['first'] / max(s['words'], 1) * 100:>9.1f}% "
              f"{s['dec'] / max(s['words'], 1) * 100:>10.1f}% {s['attempts'] / max(s['words'], 1):>11.2f} {s['skipped']:>7} {wpm:>5.0f}  {s['lm']}")


if __name__ == "__main__":
    main()
