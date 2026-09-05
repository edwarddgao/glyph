#!/usr/bin/env python3
"""Sample modern, conversational English text for the post-encoder evaluation.

    python scripts/fetch_text_domains.py --n 1500

The swipe corpora's text is Common Voice (Wikipedia-style read speech) and
How We Swipe's four-word bags — neither is what people type on a phone. This
pulls fixed samples from public sets with a messaging-like register, one file
of normalized sentences per domain under data/text_domains/:

  tweets      cardiffnlp/tweet_eval (sentiment): real tweets 2017–20, handles/URLs stripped
  reddit      sentence-transformers/reddit-title-body: comment/post bodies, split to sentences
  wildchat    allenai/WildChat-1M: the user's turns to a chatbot, 2023–24, English
  dialog      li2017dailydialog/daily_dialog: everyday dialogue (crowd-written)
  movies      cornell_movie_dialog: film dialogue lines

Normalization matches the corpora: lowercase, letters-only tokens (a sentence
with a digit, URL or non-letter token is dropped rather than mangled), 3–12
words, deduplicated. Sentences are sampled with a fixed seed, streaming, so
nothing large is downloaded.
"""
from __future__ import annotations

import argparse, random, re
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "text_domains"
URL = re.compile(r"https?://\S+|www\.\S+")
HANDLE = re.compile(r"[@#]\w+")
SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> list[str]:
    """Sentences of 3–12 letters-only words, or [] if the text does not qualify."""
    text = URL.sub(" ", HANDLE.sub(" ", text.replace("\n", " ")))
    out = []
    for sent in SPLIT.split(text):
        toks = sent.strip().split()
        if not 3 <= len(toks) <= 12:
            continue
        words = []
        ok = True
        for t in toks:
            w = t.strip(".,!?;:\"'()[]…-—*").lower().replace("’", "'")
            w = w.replace("'", "")            # don't -> dont, like the corpora
            if not w:
                continue
            if not w.isascii() or not w.isalpha():
                ok = False; break
            words.append(w)
        if ok and 3 <= len(words) <= 12:
            out.append(" ".join(words))
    return out


def take(iterable, field, n, seed, per_item=3):
    rng = random.Random(seed)
    seen, out = set(), []
    for item in iterable:
        text = item.get(field) if isinstance(item, dict) else None
        if not text or not isinstance(text, str):
            continue
        cands = normalize(text)
        rng.shuffle(cands)
        for s in cands[:per_item]:
            if s not in seen:
                seen.add(s); out.append(s)
        if len(out) >= n * 4:            # oversample, then subsample for a fixed set
            break
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    from datasets import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "tweets": lambda: take(load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train", streaming=True), "text", a.n, 0),
        "reddit": lambda: take(load_dataset("sentence-transformers/reddit-title-body", split="train", streaming=True), "body", a.n, 1),
        "wildchat": lambda: take(
            ({"text": conv["conversation"][0]["content"]} for conv in load_dataset("allenai/WildChat-1M", split="train", streaming=True)
             if conv.get("language") == "English" and conv.get("conversation")),
            "text", a.n, 2, per_item=2),
        "dialog": lambda: take(({"text": u} for d in load_dataset("li2017dailydialog/daily_dialog", split="train", streaming=True, trust_remote_code=True) for u in d["dialog"]), "text", a.n, 3, per_item=1),
        "movies": lambda: take(({"text": u} for d in load_dataset("cornell_movie_dialog", split="train", streaming=True, trust_remote_code=True) for u in (d.get("utterance", {}) or {}).get("text", [])), "text", a.n, 4, per_item=1),
    }
    for name, job in jobs.items():
        if a.only and name != a.only:
            continue
        try:
            sents = job()
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}")
            continue
        (OUT / f"{name}.txt").write_text("\n".join(sents) + "\n")
        print(f"{name}: {len(sents)} sentences, {sum(len(s.split()) for s in sents)} words -> {OUT / (name + '.txt')}")
        for s in sents[:3]:
            print("   ", s)


if __name__ == "__main__":
    main()
