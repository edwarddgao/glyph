#!/usr/bin/env python3
"""#8's matched-size in-domain comparison, re-scored on label-filtered eval sets.

    python scripts/rerun_matched_size_clean.py

#80 found ~3-4 pts of How We Swipe's error rate is label mismatch (foreign
words, aborted gestures, swiped misspellings recorded under the prompt), so
#8's "5.5 pts intrinsic" -- the in-domain greedy gap between a model trained
on 60k HWS swipes and one trained on 60k FUTO swipes, each read on its own
held-out users -- absorbed that noise. The two checkpoints still exist
(runs/scratch_hws, runs/scratch_futo); this re-reads them on the *full*
held-out sets (#8 used 20 batches) with and without a decoder-independent
label filter applied identically to both corpora:

    drop if the label is outside the English lexicon, or wordfreq scores it as
    es/de/fr/pt/it/nl and not English, or the gesture travelled under half a
    key per letter transition (aborted), or the label's training-free geometry
    cost per letter exceeds --cost-thr (the gesture does not trace the label).

The filter uses the label and the raw gesture only -- never a decoder output --
so the number it produces is not circular.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus, SwipeDataset, decode, make_loader
from swipe_typing.geomllm import GestureDP, GeomConfig
from eval_decoder import build_lexicon, load_model, pick_device
from finetune_transfer import split_by_user


def label_filter(swipes, kb, eng, cost_thr):
    from wordfreq import zipf_frequency
    cfg = GeomConfig(time_weight=1.25)
    kw = 2 * float(kb.radii[0][0])
    keep, why = [], {"non_english": 0, "foreign": 0, "aborted": 0, "untraced": 0}
    for sw in swipes:
        w = sw.word
        if w not in eng:
            why["non_english"] += 1
            continue
        if max(zipf_frequency(w, l) for l in ("es", "de", "fr", "pt", "it", "nl")) >= 3.0 and zipf_frequency(w, "en") < 2.5:
            why["foreign"] += 1
            continue
        path = float(np.linalg.norm(np.diff(sw.points, axis=0), axis=1).sum()) / kw
        if len(w) >= 4 and path < 0.5 * (len(w) - 1):
            why["aborted"] += 1
            continue
        if GestureDP(sw.points, sw.t, kb, cfg).word_cost(w) / len(w) > cost_thr:
            why["untraced"] += 1
            continue
        keep.append(sw)
    return keep, why


@torch.no_grad()
def greedy(model, corpus, kb, device, alphabet, key_units, mode):
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode, key_units=key_units)
    preds, refs = [], []
    for x, targets, lengths in make_loader(ds, batch_size=256, shuffle=False, num_workers=2):
        lp = model(x.to(device))
        preds.extend(decode.greedy_decode(lp, model.cfg.blank, alphabet))
        refs.extend(decode.target_strings(targets, lengths, alphabet))
    return decode.score(preds, refs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--cost-thr", type=float, default=6.0)
    args = ap.parse_args()
    root = Path(args.cache)
    kb = KeyboardLayout.qwerty()
    device = pick_device("auto")
    models = {n: load_model(f"runs/{n}/encoder.pt", device) for n in ("scratch_hws", "scratch_futo")}
    alphabet = models["scratch_hws"][1]
    eng = set(build_lexicon("train+wf320k", root, alphabet, 1.0)._counts)

    _, hws_test = split_by_user(root / "how_we_swipe/test", alphabet, 0.7, None)
    from swipe_typing import cache
    hws_sw = [sw for i, sw in enumerate(cache.read(root / "how_we_swipe/test"))]
    from finetune_transfer import user_bucket
    hws_sw = [sw for sw in hws_sw if user_bucket(sw.session) >= 0.7]
    futo_sw = [sw for i, sw in enumerate(cache.read(root / "futo/validation")) if i < 20000]
    sets = {"hws held-out users": hws_sw, "futo val": futo_sw}
    print(f"label filter: English lexicon, wordfreq foreign, aborted, geometry cost/letter > {args.cost_thr}\n")
    for name, sws in sets.items():
        kept, why = label_filter(sws, kb, eng, args.cost_thr)
        print(f"== {name}: {len(sws):,} swipes; dropped {len(sws) - len(kept):,} ({100 * (len(sws) - len(kept)) / len(sws):.2f}%): {why}")
        full = SwipeCorpus.from_swipes(sws, alphabet)
        clean = SwipeCorpus.from_swipes(kept, alphabet)
        for mn, (model, alph, ku, mode) in models.items():
            a = greedy(model, full, kb, device, alph, ku, mode)
            b = greedy(model, clean, kb, device, alph, ku, mode)
            print(f"   {mn:>12}: greedy top-1 all {100 * a['wacc']:.2f} (n={a['n']:,})  clean {100 * b['wacc']:.2f} (n={b['n']:,})   CER {a['cer']:.3f} -> {b['cer']:.3f}")
        print()


if __name__ == "__main__":
    main()
