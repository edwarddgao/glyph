#!/usr/bin/env python3
"""The composed cell on the iPhone capture: everything the notebook says helps
off-domain, in one score formula.

Per swipe, candidates = trie-beam top-8 UNION geometry-trie top-8 (the #73
proposal channel — attacks list-coverage misses). Every candidate, injected or
not, is scored by the same acoustic:

    ctc_full(w) + alpha*log_uni(w) + beta*len(w) - gamma*geom_cost(w)

with geom_cost the dwell-weighted GestureDP alignment (#71, time_weight in the
flat 1.0-1.5 band) and gamma=0.5, #73's off-domain optimum — this capture IS
off-domain. Then the delta-form gpt2-xl sentence beam (mu=0.8) on top, at all
three commitment lags. Run for both the canonical and perm-MMI encoders.

Usage:  .venv/bin/python capture/composed_stack.py --lm gpt2-xl
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT.parent / "scripts"))
sys.path.insert(0, str(ROOT))

from swipe_typing.geomllm import GeomConfig, GestureDP              # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout            # noqa: E402
from swipe_typing.model import SwipeDataset, make_loader            # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_search         # noqa: E402
from eval_decoder import build_lexicon, load_model, pick_device, run_encoder  # noqa: E402
from eval_geom_trie import build_trie, decode as geom_decode        # noqa: E402
from decode_capture import (build_corpus, latest_per_sentence,      # noqa: E402
                            norm_word, word_align)
from fused_rescore import LMScorer, sentence_decode                 # noqa: E402

ALPHA, BETA, GAMMA, TIME_W, M = 0.8, 1.2, 0.5, 1.25, 8


def ctc_score(lp: np.ndarray, word: str, alphabet: str, blank: int) -> float:
    x = torch.from_numpy(lp).unsqueeze(1)
    tgt = torch.tensor([[alphabet.index(c) for c in word]])
    return -float(F.ctc_loss(x, tgt, torch.tensor([lp.shape[0]]),
                             torch.tensor([len(word)]), blank=blank,
                             reduction="sum", zero_infinity=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="gpt2-xl")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--encoders", default="canonical",
                    help="comma list: canonical, perm25mmi, or name=path")
    ap.add_argument("--sets", default="",
                    help="restrict eval to these phrase sets, e.g. 1,4,7")
    args = ap.parse_args()
    enc_table = {"canonical": "runs/full/encoder.pt",
                 "perm25mmi": "runs/perm25mmi/encoder_ep0.pt"}

    device = pick_device("auto")
    kb = KeyboardLayout.qwerty()
    captures = latest_per_sentence("capture")
    if args.sets:
        keep = {int(s) for s in args.sets.split(",")}
        captures = {k: v for k, v in captures.items()
                    if v.get("set", 1) in keep}
        print(f"eval restricted to sets {sorted(keep)}: "
              f"{len(captures)} sentences")
    corpus, meta = build_corpus(captures, ALPHABET)
    lexicon = build_lexicon("train+wf320k", ROOT.parent / "data/canonical",
                            ALPHABET, 1.0)

    def uni(w: str) -> float:
        node = lexicon.node_for(w)
        return node.logp if node is not None and node.is_word else -25.0

    # ---- geometry channel: dwell-weighted DP + trie proposals (shared) ----
    gcfg = GeomConfig(time_weight=TIME_W)
    trie_root = build_trie(lexicon.counts().keys())
    dps, geom_props = [], []
    for i in range(len(corpus)):
        dp = GestureDP(corpus.points(i), corpus.times(i), kb, gcfg)
        dps.append(dp)
        cands = geom_decode(dp, trie_root, kb, beam=2000)[:50]
        ranked = sorted(((w, -g + 0.5 * uni(w)) for w, g in cands),
                        key=lambda t: -t[1])
        geom_props.append([w for w, _ in ranked[:M]])
    print(f"geometry proposals ready ({len(corpus)} swipes)")

    lm = LMScorer(args.lm, device)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["key"]].append(i)
    for s in groups:
        groups[s].sort(key=lambda i: corpus.word_idx[i])

    # ---- native condition: per-word correctness, aligned to ref positions ----
    natives = latest_per_sentence("native")
    native_ok: dict[tuple[str, int], bool] = {}
    for (sess, s), p in natives.items():
        ref = [norm_word(w) for w in s.split()]
        hyp = [norm_word(w) for w in p["typed"].split() if norm_word(w)]
        _c, _n, pairs = word_align(ref, hyp)
        pos = 0
        for r, h in pairs:
            if r is not None:
                native_ok[(sess + "|" + s, pos)] = (r == h)
                pos += 1
    nat_hit = sum(native_ok.values())
    print(f"native (QuickPath): {nat_hit}/{len(native_ok)} "
          f"= {nat_hit / len(native_ok):.1%}")

    enc_list = []
    for e in args.encoders.split(","):
        if "=" in e:
            name, path = e.split("=", 1)
            enc_list.append((name, path))
        else:
            enc_list.append((e, enc_table[e]))

    for enc_name, ckpt in enc_list:
        model, alphabet, key_units, mode = load_model(
            str(ROOT.parent / ckpt), device)
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                          key_units=key_units, shape_only=model.cfg.shape_only)
        loader = make_loader(ds, batch_size=64, shuffle=False, num_workers=0)
        log_probs, refs = run_encoder(model, loader, device, alphabet)
        blank = model.cfg.blank

        cfg = BeamConfig(beam_width=64, alpha=ALPHA, beta=BETA, top_k=M)
        cand_lists = []
        for i, lp in enumerate(log_probs):
            beam_words = [w for w, _ in beam_search(lp, lexicon, alphabet, cfg)[:M]]
            merged = list(dict.fromkeys(beam_words + geom_props[i]))
            scored = [(w, ctc_score(lp, w, alphabet, blank)
                       + ALPHA * uni(w) + BETA * len(w)
                       - args.gamma * dps[i].word_cost(w)) for w in merged]
            scored.sort(key=lambda t: -t[1])
            cand_lists.append(scored)

        cover = sum(refs[i] in [w for w, _ in cand_lists[i]]
                    for i in range(len(refs)))
        fp = sum(cand_lists[i][0][0] == refs[i] for i in range(len(refs)))
        print(f"\n#### encoder {enc_name}  (gamma={args.gamma}, "
              f"time_weight={TIME_W})")
        print(f"  first pass {fp / len(refs):.1%}  "
              f"truth-in-list {cover / len(refs):.1%}  "
              f"(was 90.3% coverage without geometry)")

        for lag, name in [(0, "streaming"), (1, "lookahead-1"), (None, "joint")]:
            hit, tag_hit, errs = 0, defaultdict(lambda: [0, 0]), []
            ours_ok: dict[int, bool] = {}
            for s, idx in sorted(groups.items()):
                slots = [(refs[i], cand_lists[i]) for i in idx]
                out = sentence_decode(slots, lm, lag)
                for w, i in zip(out, idx):
                    ok = w == refs[i]
                    ours_ok[i] = ok
                    hit += ok
                    tag_hit[meta[i]["tag"]][0] += ok
                    tag_hit[meta[i]["tag"]][1] += 1
                    if not ok and lag is None:
                        errs.append((refs[i], w, meta[i]["tag"],
                                     "in-list" if refs[i] in
                                     [c for c, _ in cand_lists[i]] else "MISS"))
            cells = "  ".join(f"{t} {c}/{n}={c / n:.0%}"
                              for t, (c, n) in sorted(tag_hit.items()))
            print(f"  fused {name:<12} top-1 {hit / len(refs):.1%} "
                  f"({hit}/{len(refs)})   {cells}")
            mcnemar_vs_native(name, ours_ok, native_ok, corpus, meta)
            if lag is None:
                per_set(ours_ok, native_ok, corpus, meta, natives)
            if errs:
                for r, p, tag, st in errs[:25]:
                    print(f"     {r:<12} -> {p:<12} [{tag}, {st}]")
                if len(errs) > 25:
                    print(f"     ... {len(errs) - 25} more")


def per_set(ours_ok, native_ok, corpus, meta, natives):
    """Joint-mode accuracy by phrase set, ours vs native, plus swipe tempo."""
    ours = defaultdict(lambda: [0, 0])
    nat = defaultdict(lambda: [0, 0])
    sent_set = {}
    for i, ok in ours_ok.items():
        s = meta[i]["set"]
        ours[s][0] += ok
        ours[s][1] += 1
        sent_set[meta[i]["key"]] = s
    for (sess, s), p in natives.items():
        st = p.get("set", 1)
        for pos in range(len(s.split())):
            k = (sess + "|" + s, pos)
            if k in native_ok:
                nat[st][0] += native_ok[k]
                nat[st][1] += 1
    # median per-word swipe duration per set, as a sloppiness proxy
    dur = defaultdict(list)
    for i in range(len(corpus)):
        t = corpus.times(i)
        if len(t):
            dur[meta[i]["set"]].append(int(t[-1]))
    print("      per-set (joint):  set  ours      native    med-swipe-ms")
    for s in sorted(ours):
        o, on = ours[s]
        c, cn = nat.get(s, (0, 0))
        med = int(np.median(dur[s])) if dur[s] else 0
        print(f"        set {s}:  {o}/{on} = {o / on:.0%}   "
              f"{c}/{cn} = {c / cn:.0%}   {med}ms")


def mcnemar_vs_native(name, ours_ok, native_ok, corpus, meta):
    """Paired word-level comparison against the native keyboard."""
    from scipy.stats import binomtest
    for label, sel in [("all", lambda m: True),
                       ("tail", lambda m: m["tag"] == "tail")]:
        b = c = both = n = 0
        for i, ok in ours_ok.items():
            k = (meta[i]["key"], int(corpus.word_idx[i]))
            if k not in native_ok or not sel(meta[i]):
                continue
            n += 1
            if ok and not native_ok[k]:
                b += 1
            elif not ok and native_ok[k]:
                c += 1
            elif ok:
                both += 1
        if b + c == 0:
            continue
        p = binomtest(b, b + c, 0.5).pvalue
        print(f"      vs native [{label:>4}]: ours-only-right {b}, "
              f"native-only-right {c}  (n={n}, McNemar p={p:.3g})")


if __name__ == "__main__":
    main()
