#!/usr/bin/env python3
"""Turn keyboard-captured gestures into a labeled capture corpus.

The Swipe keyboard's capture mode posts one `kbcapture_*.json` per swipe (raw
touches in canonical coordinates, first-pass candidates, fused choice,
committed word, text context). The capture page's Block B, run with the Swipe
keyboard, posts one `native_*.json` per prompted sentence (prompt, committed
text, timestamp at "next"). This joins them: the gestures posted between two
consecutive "next" events belong to the later sentence, and each gesture's
committed word is aligned to the prompt's words to label it.

Output: `capture_<session>_<ts>_kb.json` files in the same shape as Block A's
uploads (kind "capture", one per sentence, gestures carrying word/word_idx),
so decode_capture.py, fused_rescore.py, adapt_user.py and the bench exporter
consume them unchanged. Sentences whose gesture count does not match the
prompt after alignment are reported and skipped.

    .venv/bin/python iphone/join_kbcapture.py [--data iphone/data] [--since 2026-09-04]
"""
from __future__ import annotations

import argparse, json, re, time
from collections import defaultdict
from pathlib import Path


def norm(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def align_words(ref: list[str], hyp: list[str]) -> list[int | None]:
    """For each hyp position, the ref index it aligns to (Levenshtein, matches preferred), or None."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1))
    out = [None] * m
    i, j = n, m
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            out[j - 1] = i - 1; i -= 1; j -= 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--since", default=None, help="ignore uploads before this date (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    since = time.mktime(time.strptime(a.since, "%Y-%m-%d")) * 1000 if a.since else 0

    gestures = sorted((json.loads(f.read_text()) for f in data.glob("kbcapture_*.json")), key=lambda p: p["ts"])
    gestures = [g for g in gestures if g["ts"] >= since]
    prompts = sorted((json.loads(f.read_text()) for f in data.glob("native_*.json")), key=lambda p: p["ts"])
    prompts = [p for p in prompts if p.get("keyboard") == "swipe" and p["ts"] >= since]
    print(f"{len(gestures)} keyboard gestures, {len(prompts)} Swipe-keyboard prompt records")
    if not prompts:
        return

    # A prompt record's window is (previous prompt ts, this ts]; the first window opens at its
    # own ts minus the sentence's duration ("ms_shown" when logged) or 60 s.
    written = skipped = 0
    prev_ts = 0
    per_session = defaultdict(int)
    for p in prompts:
        lo = prev_ts if prev_ts else p["ts"] - max(int(p.get("ms_shown", 0) or 0) + 2000, 60_000)
        hi = p["ts"] + 1500                      # the final swipe's upload may land just after "next"
        prev_ts = p["ts"]
        window = [g for g in gestures if lo < g["ts"] <= hi]
        ref = [norm(w) for w in p["sentence"].split()]
        committed = [norm(g.get("committed", "")) for g in window]
        idx = align_words(ref, committed)
        labeled = []
        used = set()
        for g, i in zip(window, idx):
            if i is None or i in used or not ref[i]:
                continue
            used.add(i)
            labeled.append({"word": ref[i], "word_idx": i, "x": g["x"], "y": g["y"], "t": g["t"], "aspect": g.get("aspect", 2.4),
                            "committed": g.get("committed"), "first_pass": g.get("first_pass"), "fused": g.get("fused"),
                            "candidates": g.get("candidates"), "context_before": g.get("context_before")})
        # words the user tapped (single letters) legitimately have no gesture
        expected = [i for i, w in enumerate(ref) if len(w) > 1]
        missing = [ref[i] for i in expected if i not in used]
        if len(missing) > max(1, len(expected) // 4):
            skipped += 1
            print(f"  skip {p['sentence']!r}: {len(window)} gestures in window, unlabeled {missing}")
            continue
        out = {"kind": "capture", "session": p.get("session", "anon"), "set": p.get("set", 0), "ts": p["ts"],
               "sentence": p["sentence"], "tag": p.get("tag", ""), "ua": "swipe-keyboard", "typed": p.get("typed"),
               "gestures": sorted(labeled, key=lambda g: g["word_idx"])}
        name = f"capture_{out['session']}_{p['ts']}_kb.json"
        if not a.dry_run:
            (data / name).write_text(json.dumps(out, indent=1))
        written += 1
        per_session[out["session"]] += len(labeled)
    print(f"wrote {written} sentences ({skipped} skipped); gestures per session: {dict(per_session)}")
    print("next: .venv/bin/python iphone/decode_capture.py   (scores these with the offline stack alongside the web-page captures)")


if __name__ == "__main__":
    main()
