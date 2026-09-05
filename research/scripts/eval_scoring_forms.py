#!/usr/bin/env python3
"""Which fused-scoring form is robust across text registers?

    python scripts/eval_scoring_forms.py --n 300 --lm distilgpt2

The shipped score is a hand-assembled algebra fitted on Common Voice:

  current   ctc + α·uni + β·len + μ·(logP_LM(w|ctx) − marginal_LM(w))        α=0.8 β=1.2 μ=0.8
  nouni     ctc + β·len + μ·(logP_LM(w|ctx) − marginal_LM(w))                 (the unigram removed)
  bayes     ctc + β·len + μ·logP_LM(w|ctx)                                    (shallow fusion, LM owns the prior)
  ilm       (ctc − λ·ilm(w)) + β·len + μ·logP_LM(w|ctx)                       (ASR-style: internal LM subtracted)

`ilm(w)` is the encoder's own prior: the CTC score of `w` under a gesture-free
input (the training-mean feature vector, #78's "mean" ablation for the AR
decoder, done here for the CTC encoder). All weights are fitted ONCE on real
FUTO validation gestures by grid over lookahead-1 accuracy, then held fixed on
every other domain — synthetic gestures on FUTO / tweets / reddit / WildChat
text and the real iPhone capture gestures. A robust form is one whose fixed
weights are close to each domain's own optimum. Reports lookahead-1 accuracy,
first-word accuracy, and the fixed-vs-per-domain-best gap.
"""
from __future__ import annotations

import argparse, collections, itertools, json, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "iphone"))
from swipe_typing import cache, minjerk  # noqa: E402
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeDataset, make_loader  # noqa: E402
from swipe_typing.model.beam import BeamConfig, beam_candidates  # noqa: E402
from swipe_typing.model.data import SwipeCorpus  # noqa: E402
from eval_decoder import build_lexicon, load_model, run_encoder  # noqa: E402
import fused_rescore as fr  # noqa: E402
from eval_text_domain import futo_sentences, corpus_from_gestures, load_sentences  # noqa: E402

BEAM, M = 8, 8


@torch.no_grad()
def ilm_table(model, words, alphabet, device):
    """log P_ctc(word | gesture-free input): the encoder's internal prior."""
    x = model.input_mean.expand(1, 64, -1).clone()          # normalizes to zeros
    lp = model(x.to(device))[0].cpu()                        # (64, 27)
    blank = model.cfg.blank
    out = {}
    char = {c: i for i, c in enumerate(alphabet)}
    lpT = lp[None].transpose(0, 1)                           # (T, 1, C)
    for w in words:
        tgt = torch.tensor([[char[c] for c in w]])
        loss = F.ctc_loss(lpT, tgt, torch.tensor([64]), torch.tensor([len(w)]), blank=blank, reduction="sum", zero_infinity=True)
        out[w] = -float(loss)
    return out


def decode(slots, lm, form, w, ilm, lag=1):
    """slots: [(ref, [(word, ctc, uni, len)])]; form/weights select the score."""
    states = [((), 0.0)]
    for t, (_ref, cands) in enumerate(slots):
        ctxs = [" ".join(s) for s, _ in states]
        lm.fill([(c, cw) for c in ctxs for cw, *_ in cands])
        priors = {cw: lm.prior(cw) for cw, *_ in cands} if form in ("current", "nouni") else {}
        exp = {}
        for (words, cum), ctx in zip(states, ctxs):
            for cw, ctc, uni, ln in cands:
                l = lm.cache[(ctx, cw)]
                if form == "current":
                    sc = ctc + w["alpha"] * uni + w["beta"] * ln + w["mu"] * (l - priors[cw])
                elif form == "nouni":
                    sc = ctc + w["beta"] * ln + w["mu"] * (l - priors[cw])
                elif form == "bayes":
                    sc = ctc + w["beta"] * ln + w["mu"] * l
                elif form == "ilm":
                    sc = ctc - w["lam"] * ilm[cw] + w["beta"] * ln + w["mu"] * l
                else:
                    raise ValueError(form)
                wt = words + (cw,)
                if wt not in exp or cum + sc > exp[wt]:
                    exp[wt] = cum + sc
        states = sorted(exp.items(), key=lambda kv: -kv[1])[:BEAM]
        if lag is not None and t - lag >= 0:
            j = t - lag; keep = states[0][0][j]
            states = [s for s in states if s[0][j] == keep] or states[:1]
    return list(states[0][0])


GRIDS = {
    "current": [dict(alpha=0.8, beta=1.2, mu=mu) for mu in (0.4, 0.6, 0.8, 1.0, 1.2)],
    "nouni": [dict(beta=b, mu=mu) for b in (0.6, 1.2) for mu in (0.4, 0.6, 0.8, 1.0, 1.2)],
    "bayes": [dict(beta=b, mu=mu) for b in (0.6, 1.2, 1.8) for mu in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)],
    "ilm": [dict(lam=lam, beta=b, mu=mu) for lam in (0.25, 0.5, 1.0) for b in (0.6, 1.2) for mu in (0.3, 0.4, 0.5, 0.6, 0.8)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lm", default="distilgpt2")
    ap.add_argument("--lm-device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--domains", default="tweets,reddit,wildchat")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cpu"); kb = KeyboardLayout.qwerty()
    lexicon = build_lexicon("train+wf320k", ROOT / "data/canonical", ALPHABET, 1.0)
    model, alphabet, key_units, mode = load_model(str(ROOT / "runs/full/encoder.pt"), device)
    cfg = BeamConfig(beam_width=64, alpha=0.8, beta=1.2, top_k=M)
    gen = minjerk.MinJerkModel.load(ROOT / "runs/minjerk_model.json")
    gen.profile, gen.dwell_prob, gen.tremor, gen.seg_jitter = "random", 0.6, 0.20, 0.25
    lm_dev = torch.device(a.lm_device); lm = fr.LMScorer(a.lm, lm_dev)
    if lm_dev.type == "cpu": lm.model = lm.model.float()

    # ---- build per-domain slot lists: [(ref, [(word, ctc, uni, len)])] per sentence ----
    def slots_for(corpus):
        ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode, key_units=key_units)
        lp, refs = run_encoder(model, make_loader(ds, batch_size=256, shuffle=False, num_workers=0), device, alphabet)
        cl = []
        for l in lp:
            c = beam_candidates(l, lexicon, alphabet, cfg)
            c.sort(key=lambda t: t[1] + cfg.alpha * t[2] + cfg.beta * t[3], reverse=True)
            cl.append([(w, ac, uni, n) for w, ac, uni, n in c[:M]])
        groups = collections.defaultdict(list)
        for i in range(len(refs)): groups[(corpus.sentences[i], i - corpus.word_idx[i])].append(i)
        return [[(refs[i], cl[i]) for i in idx] for idx in groups.values()]

    domains = {}
    futo = futo_sentences(a.n, a.seed)
    fsent = [[re.sub(r"[^a-z]", "", s.word.lower()) for s in sws] for sws in futo]
    keep = [(s, sws) for s, sws in zip(fsent, futo) if all(w in lexicon for w in s)]
    domains["futo(real)"] = slots_for(corpus_from_gestures([s for s, _ in keep], [sws for _, sws in keep]))
    rng = np.random.default_rng(a.seed)
    domains["futo(synth)"] = slots_for(corpus_from_gestures([s for s, _ in keep], [[minjerk.generate(gen, w, kb, rng) for w in s] for s, _ in keep]))
    for d in a.domains.split(","):
        sents = [s for s in load_sentences(ROOT / "data/text_domains" / f"{d}.txt", a.n) if all(w in lexicon for w in s)]
        domains[d] = slots_for(corpus_from_gestures(sents, [[minjerk.generate(gen, w, kb, rng) for w in s] for s in sents]))
    # real iPhone capture gestures (all 96 sentences)
    bench = [s for s in json.load(open(ROOT.parent / "keyboard/Resources/bench_gestures.json"))["sentences"] if s["source"] == "capture"]
    from swipe_typing.schema import Swipe  # noqa: E402
    cap_sents, cap_g = [], []
    for s in bench:
        if not all(w in lexicon for w in s["words"]): continue
        cap_sents.append(s["words"])
        cap_g.append([Swipe(x=np.asarray(g["x"], np.float32), y=np.asarray(g["y"], np.float32), t=np.asarray(g["t"], np.int32),
                            word=w, aspect=2.2, source="capture", session=s["session"], sentence=" ".join(s["words"]), word_idx=j)
                      for j, (g, w) in enumerate(zip(s["gestures"], s["words"]))])
    domains["capture(real)"] = slots_for(corpus_from_gestures(cap_sents, cap_g))

    allwords = sorted({w for sl in domains.values() for sent in sl for _, c in sent for w, *_ in c})
    ilm = ilm_table(model, allwords, alphabet, device)
    print(f"domains: " + ", ".join(f"{k} {sum(len(s) for s in v)} words" for k, v in domains.items()) + f"; ilm over {len(allwords)} words", flush=True)

    def score(slots, form, w):
        ok = first_ok = first_n = 0; n = 0
        for sent in slots:
            out = decode(sent, lm, form, w, ilm)
            for j, ((ref, _), o) in enumerate(zip(sent, out)):
                n += 1; ok += o == ref
                if j == 0: first_n += 1; first_ok += o == ref
        return ok / n * 100, first_ok / max(first_n, 1) * 100

    def first_pass(slots):
        n = ok = 0
        for sent in slots:
            for ref, c in sent: n += 1; ok += bool(c) and c[0][0] == ref
        return ok / n * 100

    # ---- fit on futo(real), evaluate everywhere at fixed weights; also each domain's own best ----
    results = {}
    for form, grid in GRIDS.items():
        t0 = time.time()
        fit = [(score(domains["futo(real)"], form, w)[0], i) for i, w in enumerate(grid)]
        best_i = max(fit)[1]; wbest = grid[best_i]
        row = {"weights": wbest, "domains": {}}
        for name, slots in domains.items():
            fixed, first = score(slots, form, wbest)
            own = max(score(slots, form, w)[0] for w in grid) if name != "futo(real)" else fixed
            row["domains"][name] = dict(fixed=fixed, own_best=own, first_word=first)
        results[form] = row
        print(f"{form:<8} fitted on futo(real): {wbest}  ({time.time() - t0:.0f}s)", flush=True)

    names = list(domains)
    print("\nfirst pass (current acoustic ranking, no LM):  " + "  ".join(f"{n} {first_pass(domains[n]):5.1f}" for n in names))
    print(f"\n{'form':<8} " + " ".join(f"{n:>14}" for n in names) + "   | mean gap to each domain's own best (robustness)")
    for form, row in results.items():
        cells = " ".join(f"{row['domains'][n]['fixed']:6.1f}({row['domains'][n]['own_best'] - row['domains'][n]['fixed']:+.1f})" for n in names)
        gap = np.mean([row["domains"][n]["own_best"] - row["domains"][n]["fixed"] for n in names if n != "futo(real)"])
        print(f"{form:<8} {cells}   | {gap:.2f}")
    print("\nfirst-word accuracy at fixed weights:")
    for form, row in results.items():
        print(f"{form:<8} " + " ".join(f"{row['domains'][n]['first_word']:14.1f}" for n in names))
    print("\n(lookahead-1 accuracy; cell = fixed FUTO-fitted weights, parenthesis = how much that domain's own best weights would add)")


if __name__ == "__main__":
    main()
