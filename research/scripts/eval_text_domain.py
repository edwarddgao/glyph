#!/usr/bin/env python3
"""The post-encoder stack on modern conversational text, with synthetic gestures.

    python scripts/eval_text_domain.py --domains tweets,reddit,wildchat,dialog,movies \
        --lm distilgpt2 --n 800

For every domain file in data/text_domains/ (from fetch_text_domains.py):

  1. lexicon coverage — the share of words outside the train+wf320k trie
     (unreachable by construction; reported, not averaged into accuracy),
  2. one synthetic gesture per word from the domain-randomized minimum-jerk
     generator (#57: profile coin-flip, dwell pauses, tremor, timing jitter),
  3. the shipped stack: runs/full CTC + trie beam top-8, then the fused
     sentence search (fused_rescore.sentence_decode) at every commitment lag,
  4. accuracy overall and on sentence-initial words, plus the n-best ceiling.

Absolute numbers on synthetic gestures are not comparable to real-gesture
numbers; the point is the *LM and commitment-policy gains per text domain*
and the OOV rate, which depend on the text, not the gesture realism. A
`futo` row (real validation sentences re-synthesized the same way) anchors
the synthetic protocol against the corpus the stack was tuned on, and
`--real-futo` adds the same sentences with their real gestures.
"""
from __future__ import annotations

import argparse, collections, re, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "iphone"))
from swipe_typing import cache, minjerk  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_search  # noqa: E402
from swipe_typing.model.data import SwipeCorpus  # noqa: E402
from eval_decoder import build_lexicon, load_model, run_encoder  # noqa: E402
import fused_rescore as fr  # noqa: E402


def load_sentences(path: Path, n: int) -> list[list[str]]:
    sents = [l.split() for l in path.read_text().splitlines() if l.strip()]
    return sents[:n]


def futo_sentences(n: int, seed: int = 0):
    """Complete futo/validation sentences with their real gestures."""
    groups = collections.defaultdict(list)
    for sw in cache.read(str(ROOT / "data/canonical/futo/validation")):
        groups[(sw.session, sw.sentence)].append(sw)
    full = []
    for (sess, sent), sws in groups.items():
        sws.sort(key=lambda s: s.word_idx)
        ref = [re.sub(r"[^a-z]", "", w.lower()) for w in sent.split()]
        if len(sws) == len(ref) and 3 <= len(ref) <= 12 and all(re.sub(r"[^a-z]", "", s.word.lower()) == r for s, r in zip(sws, ref)):
            full.append(sws)
    rng = np.random.default_rng(seed); rng.shuffle(full)
    return full[:n]


def corpus_from_gestures(sentences, gestures_per_sentence):
    xs, ys, ts, offsets, words, aspects, sents, widx = [], [], [], [0], [], [], [], []
    for words_s, gs in zip(sentences, gestures_per_sentence):
        for j, (w, g) in enumerate(zip(words_s, gs)):
            xs.extend(np.asarray(g.x, np.float32)); ys.extend(np.asarray(g.y, np.float32)); ts.extend(np.asarray(g.t, np.int32))
            offsets.append(len(xs)); words.append(w); aspects.append(float(g.aspect)); sents.append(" ".join(words_s)); widx.append(j)
    return SwipeCorpus(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ts, np.int32),
                       np.asarray(offsets, np.int64), words, np.asarray(aspects, np.float32), sentences=sents, word_idx=widx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="tweets,reddit,wildchat,dialog,movies")
    ap.add_argument("--n", type=int, default=800, help="sentences per domain")
    ap.add_argument("--lm", default="distilgpt2")
    ap.add_argument("--lm-device", default="mps" if torch.backends.mps.is_available() else "cpu",
                    help="LM device; the encoder/beam stay on CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-futo", action="store_true", help="also decode the futo anchor with its real gestures")
    ap.add_argument("--profile", default="random"); ap.add_argument("--dwell-prob", type=float, default=0.6)
    ap.add_argument("--tremor", type=float, default=0.20); ap.add_argument("--seg-jitter", type=float, default=0.25)
    a = ap.parse_args()

    device = torch.device("cpu")
    kb = KeyboardLayout.qwerty()
    lexicon = build_lexicon("train+wf320k", ROOT / "data/canonical", ALPHABET, 1.0)
    model, alphabet, key_units, mode = load_model(str(ROOT / "runs/full/encoder.pt"), device)
    cfg = BeamConfig(beam_width=64, alpha=0.8, beta=1.2, top_k=8)
    gen = minjerk.MinJerkModel.load(ROOT / "runs/minjerk_model.json")
    gen.profile, gen.dwell_prob, gen.tremor, gen.seg_jitter = a.profile, a.dwell_prob, a.tremor, a.seg_jitter
    lm_dev = torch.device(a.lm_device)
    lm = fr.LMScorer(a.lm, lm_dev)
    if lm_dev.type == "cpu":
        lm.model = lm.model.float()   # fp16 matmuls are not implemented on CPU

    rows = []
    domains = [("futo(synth)", None)] + [(d, ROOT / "data/text_domains" / f"{d}.txt") for d in a.domains.split(",")]
    if a.real_futo:
        domains.insert(0, ("futo(real)", None))
    futo = futo_sentences(a.n, a.seed)
    for name, path in domains:
        t0 = time.time()
        if name.startswith("futo"):
            sentences = [[re.sub(r"[^a-z]", "", s.word.lower()) for s in sws] for sws in futo]
        else:
            if not path.exists():
                print(f"{name}: missing {path}"); continue
            sentences = load_sentences(path, a.n)
        # lexicon coverage; keep sentences whose words are all in-lexicon for the accuracy rows
        allw = [w for s in sentences for w in s]
        oov = [w for w in allw if w not in lexicon]
        oov_rate = len(oov) / max(len(allw), 1)
        kept = [s for s in sentences if all(w in lexicon for w in s)]
        if name == "futo(real)":
            keep_sws = [sws for sws, s in zip(futo, sentences) if all(w in lexicon for w in s)]
            corpus = corpus_from_gestures(kept, keep_sws)
        else:
            rng = np.random.default_rng(a.seed)
            gests = [[minjerk.generate(gen, w, kb, rng) for w in s] for s in kept]
            corpus = corpus_from_gestures(kept, gests)
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode, key_units=key_units)
        lp, refs = run_encoder(model, make_loader(ds, batch_size=256, shuffle=False, num_workers=0), device, alphabet)
        cands = [beam_search(l, lexicon, alphabet, cfg)[:8] for l in lp]
        groups = collections.defaultdict(list)
        for i in range(len(refs)):
            groups[(corpus.sentences[i], i - corpus.word_idx[i])].append(i)
        res = {k: [0, 0] for k in ["first", "streaming", "lookahead1", "joint", "ceiling"]}
        first_pos = {k: [0, 0] for k in ["first", "streaming", "lookahead1", "joint"]}
        for idx in groups.values():
            slots = [(refs[i], cands[i]) for i in idx]
            outs = {"first": [cands[i][0][0] if cands[i] else "" for i in idx],
                    "streaming": fr.sentence_decode(slots, lm, 0),
                    "lookahead1": fr.sentence_decode(slots, lm, 1),
                    "joint": fr.sentence_decode(slots, lm, None)}
            for k, out in outs.items():
                for j, i in enumerate(idx):
                    ok = out[j] == refs[i]
                    res[k][0] += ok; res[k][1] += 1
                    if j == 0: first_pos[k][0] += ok; first_pos[k][1] += 1
            for i in idx:
                res["ceiling"][0] += refs[i] in [w for w, *_ in cands[i]]; res["ceiling"][1] += 1
        n_words = res["first"][1]
        pct = lambda r: r[0] / max(r[1], 1) * 100
        row = dict(domain=name, sentences=len(kept), words=n_words, oov=oov_rate * 100,
                   first=pct(res["first"]), streaming=pct(res["streaming"]), look1=pct(res["lookahead1"]), joint=pct(res["joint"]),
                   ceiling=pct(res["ceiling"]), first_word_nolm=pct(first_pos["first"]), first_word_look1=pct(first_pos["lookahead1"]))
        rows.append(row)
        print(f"{name:<12} {len(kept):>4} sents {n_words:>5} words | OOV {row['oov']:4.1f}% | first pass {row['first']:5.1f} | "
              f"streaming {row['streaming']:5.1f} | lookahead-1 {row['look1']:5.1f} | joint {row['joint']:5.1f} | ceiling@8 {row['ceiling']:5.1f} | "
              f"first word: no-LM {row['first_word_nolm']:5.1f} → look1 {row['first_word_look1']:5.1f}  ({time.time() - t0:.0f}s)", flush=True)
        oov_common = collections.Counter(oov).most_common(8)
        print(f"             OOV examples: {', '.join(w for w, _ in oov_common)}")
    print("\nLM gain (lookahead-1 − first pass) and headroom share ((look1−first)/(ceiling−first)):")
    for r in rows:
        share = (r["look1"] - r["first"]) / max(r["ceiling"] - r["first"], 1e-9) * 100
        print(f"  {r['domain']:<12} +{r['look1'] - r['first']:4.1f} pts   {share:4.0f}% of headroom   OOV {r['oov']:4.1f}%")


if __name__ == "__main__":
    main()
