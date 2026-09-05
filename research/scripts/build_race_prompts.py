#!/usr/bin/env python3
"""Build SwipeRacer's prompt pool: what text should players swipe?

    .venv/bin/python scripts/build_race_prompts.py [--n 3000] [--everyday-frac 0.4] [--out ../keyboard/Resources/race_prompts.json]

What the encoder learns from a swipe is the mapping from a path to a letter
sequence; sentence meaning is the LM's business. So the pool is chosen for
*word* coverage, not sentence variety, from real modern text people type
(`data/text_domains/`: tweets, reddit, WildChat — fetch_text_domains.py), with
three constraints:

  in-lexicon   every word is in train+wf320k, so the decoder's verdict recorded
               with each swipe is meaningful and the word is real English;
  clean        no sentence with a word on the blocklist below (players are
               strangers) or with a word so rare (zipf < 1.8) it is likely noise;
  4–9 words    long enough to be a sentence, short enough for one race step;
  swipeable    every token is something a person would swipe rather than tap —
               no initialisms or vowel-less strings (hc, nfl, btw), two-letter
               tokens from a list of real words, one-letter only i / a.

Selection is greedy by coverage gain: each candidate sentence scores the sum
over its words of band_weight / (1 + times the word is already in the pool),
with tail words (zipf < 3.5) weighted 3, mid (3.5–5) 1.5, head (≥ 5) 0.3 — so
the pool spends its budget on words the model sees rarely, and "the" stops
paying after a handful of repeats. Natural frequency is still represented:
every sentence carries its head words too, so the pool's word distribution is
Zipf flattened, not uniform. Each sentence is tagged `everyday` (all words
zipf ≥ 3.5) or `tail`, and the game races 3 everyday + 2 tail; the tag is in
every record, so the everyday half can serve as an unbiased test set while the
tail half buys coverage.
"""
from __future__ import annotations

import argparse, json, random, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from wordfreq import zipf_frequency  # noqa: E402
from swipe_typing.layout import ALPHABET  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402

# Words that must not appear in a prompt shown to a stranger. Short and blunt on
# purpose; extend as needed.
BLOCK = set("""fuck fucking fucked fucker fuckin shit shitty bullshit bitch bitches asshole ass dick cock pussy cunt
whore slut nigga nigger faggot fag retard retarded rape raped rapist nazi hitler kill killed killing murder suicide
porn sex sexy sexual nude naked penis vagina orgasm cum horny damn hell crap piss pissed bastard douche jerk moron
idiot stupid dumb terrorist bomb shoot shooting gun guns drugs cocaine heroin meth weed
death deaths dead die died dying killer police cops arrested prison jail war torture abuse abused assault victim victims
cancer disease tumor hospital funeral corpse blood bleeding hate hatred racist racism slave slavery""".split())


# People swipe words and tap abbreviations. A prompt must be something one would
# swipe: one-letter tokens only "i"/"a" (tapped in the game), two-letter tokens
# only real words, and nothing without a vowel (nfl, hc, btw, http…).
TWO_LETTER = set("""am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we
ah eh ha uh um ya yo ye yes id im la ex ok""".split())


def swipeable(w: str) -> bool:
    if len(w) == 1: return w in ("i", "a")
    if len(w) == 2: return w in TWO_LETTER
    if not re.search(r"[aeiouy]", w): return False
    return w not in ("http", "https", "www")


def band(z: float) -> str:
    return "tail" if z < 3.5 else ("mid" if z < 5.0 else "head")


BAND_W = {"tail": 3.0, "mid": 1.5, "head": 0.3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--everyday-frac", type=float, default=0.4, help="share of the pool with every word zipf >= 3.5 (the game races 3 everyday : 2 tail)")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=9)
    ap.add_argument("--min-zipf", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT.parent / "keyboard/Resources/race_prompts.json"))
    a = ap.parse_args()

    lex = build_lexicon("train+wf320k", ROOT / "data/canonical", ALPHABET, 1.0)
    cands = []
    seen = set()
    for f in sorted((ROOT / "data/text_domains").glob("*.txt")):
        for line in f.read_text().splitlines():
            words = line.strip().lower().split()
            if not (a.min_words <= len(words) <= a.max_words): continue
            if any(not re.fullmatch(r"[a-z]+", w) for w in words): continue
            if any(w not in lex for w in words): continue
            if any(w in BLOCK for w in words): continue
            if any(not swipeable(w) for w in words): continue
            zs = [zipf_frequency(w, "en") for w in words]
            if min(zs) < a.min_zipf: continue
            key = " ".join(words)
            if key in seen: continue
            seen.add(key)
            cands.append({"text": key, "source": f.stem, "words": words, "zipf": zs})
    print(f"{len(cands)} candidate sentences after filters", flush=True)

    rng = random.Random(a.seed); rng.shuffle(cands)
    counts: Counter = Counter()
    chosen = []

    def pick(remaining, k):
        # greedy coverage gain; `counts` is shared so the tail stratum does not re-buy words the everyday one has
        while len(chosen) < k and remaining:
            best_i, best_gain = -1, -1.0
            for i, c in enumerate(remaining):
                gain = sum(BAND_W[band(z)] / (1 + counts[w]) for w, z in zip(c["words"], c["zipf"]))
                if gain > best_gain: best_gain, best_i = gain, i
            c = remaining.pop(best_i)
            for w in c["words"]: counts[w] += 1
            chosen.append(c)

    # Two strata, so the everyday share is fixed rather than crowded out by the tail's higher gains.
    everyday = [c for c in cands if min(c["zipf"]) >= 3.5]
    tail = [c for c in cands if min(c["zipf"]) < 3.5]
    print(f"  {len(everyday)} everyday, {len(tail)} tail candidates", flush=True)
    pick(everyday, int(round(a.n * a.everyday_frac)))
    pick(tail, a.n)
    rng.shuffle(chosen)

    out = {"version": 1, "sentences": []}
    for i, c in enumerate(chosen):
        out["sentences"].append({"id": i, "text": c["text"], "source": c["source"],
                                 "tag": "tail" if min(c["zipf"]) < 3.5 else "everyday",
                                 "zipf": [round(z, 2) for z in c["zipf"]]})
    Path(a.out).write_text(json.dumps(out))

    words = Counter(w for c in chosen for w in c["words"])
    bands = Counter(band(z) for c in chosen for z in c["zipf"])
    distinct = Counter(band(zipf_frequency(w, "en")) for w in words)
    tags = Counter(s["tag"] for s in out["sentences"])
    lens = Counter(len(c["words"]) for c in chosen)
    bigrams = {w[i:i + 2] for w in words for i in range(len(w) - 1)}
    lex_bigrams = {w[i:i + 2] for w in list(lex._counts)[:50000] for i in range(len(w) - 1)}
    print(f"pool: {len(chosen)} sentences ({dict(tags)}), {sum(words.values())} word tokens, {len(words)} distinct words")
    print(f"tokens by band  head {bands['head']}  mid {bands['mid']}  tail {bands['tail']}")
    print(f"distinct by band head {distinct['head']}  mid {distinct['mid']}  tail {distinct['tail']}")
    print(f"most repeated: {words.most_common(8)}")
    print(f"sentence lengths: {dict(sorted(lens.items()))}")
    print(f"letter bigrams covered: {len(bigrams)} of {len(lex_bigrams)} in the 50k most frequent lexicon words")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
