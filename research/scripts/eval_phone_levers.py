#!/usr/bin/env python3
"""Pre-ship lever audit: what the shipped phone stack has not yet tried.

    .venv/bin/python scripts/eval_phone_levers.py firstpass  [--ckpts ...]
    .venv/bin/python scripts/eval_phone_levers.py offset
    .venv/bin/python scripts/eval_phone_levers.py fused      [--ckpts runs/...] [--beam 32]
    .venv/bin/python scripts/eval_phone_levers.py tscale
    .venv/bin/python scripts/eval_phone_levers.py lmladder   [--lms distilgpt2 gpt2 ...]

Everything is read on the two replay sets (`keyboard/Resources/bench_gestures.json`:
542 real-iPhone words, 1,337 FUTO words), which is the cross-domain evidence the
phone's model choice is made on. Variants are paired against the shipped
configuration word by word (exact McNemar).

firstpass  every candidate checkpoint through the trie beam (64) — top-1, truth
           in top-8 / top-16 / beam, everyday / tail / first word, for the shipped
           ranking (α 0.6, λ 0.25) and α 0.4. Lists are cached for `fused`.
offset     is there a systematic touch offset on the iPhone gestures? Start and
           end point vs the key centre, in key units; then the shipped encoder
           re-read with the mean offset removed.
fused      the sentence stage's knobs on the shipped lists: list depth M, μ, the
           first-word μ, α, λ, fused beam — one at a time, then combined.
tscale     the iPhone gestures are ~2× faster than the corpus donors': does the
           encoder care about absolute speed? Gesture times scaled ×2 / ×1.5 / ×0.5.
lmladder   other LMs on the shipped lists, same delta form and marginal prior.
Results: research/iphone/README.md, "Pre-ship lever audit"; log #85.
"""
from __future__ import annotations

import argparse, collections, json, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "iphone"))
from swipe_typing.layout import ALPHABET, KeyboardLayout, key_center  # noqa: E402
from swipe_typing.model import SwipeDataset  # noqa: E402
from swipe_typing.model.ar import FlatTrie, ar_beam  # noqa: E402
from swipe_typing.model.data import SwipeCorpus  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from probe_ilm_fusion import ilm_scores  # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-edwarddgao-Documents-swipe/15ccc021-b230-4007-86f7-b5e34f7d8fe5/scratchpad/levers")
SHIPPED = "runs/ar_mixed_s1/ar_decoder.pt"
DEFAULT_CKPTS = [SHIPPED, "runs/ar_mixed_mmi/ar_decoder_ep0.pt", "runs/ar_clean_s1/ar_decoder.pt", "runs/ar_mmi/ar_decoder_ep0.pt",
                 "runs/ar_enc_conformer/ar_decoder.pt", "runs/ar_enc_conformer_s2/ar_decoder.pt", "runs/ar_enc_n128/ar_decoder.pt",
                 "runs/ar_d192/ar_decoder.pt", "runs/ar_d256/ar_decoder.pt", "runs/ar_full_cont6/ar_decoder.pt", "runs/ar_perm25/ar_decoder.pt"]
ALPHA, BETA, LAM, MU, FBEAM, M = 0.6, 1.2, 0.25, 0.8, 8, 8


def mcnemar(a_ok, b_ok):
    """exact two-sided McNemar on paired correctness vectors; returns (b_only, a_only, p)."""
    b = int(sum(1 for x, y in zip(a_ok, b_ok) if y and not x)); c = int(sum(1 for x, y in zip(a_ok, b_ok) if x and not y))
    n = b + c
    if n == 0: return b, c, 1.0
    k = min(b, c); p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, p


def load_sets():
    bench = json.load(open(ROOT.parent / "keyboard/Resources/bench_gestures.json"))["sentences"]
    sets = {}
    for source in ("capture", "futo"):
        sents = [s for s in bench if s["source"] == source]
        rows = []
        for si, s in enumerate(sents):
            for j, (w, g) in enumerate(zip(s["words"], s["gestures"])):
                rows.append(dict(sid=si, j=j, word=w, tag=s["tag"], x=np.asarray(g["x"], np.float32), y=np.asarray(g["y"], np.float32), t=np.asarray(g["t"], np.int32)))
        sets[source] = rows
    return sets


def features(rows, m, mode, kb, shift=(0.0, 0.0)):
    xs, ys, ts, off = [], [], [], [0]
    for r in rows:
        xs.extend(r["x"] + shift[0]); ys.extend(r["y"] + shift[1]); ts.extend(r["t"]); off.append(len(xs))
    corpus = SwipeCorpus(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ts, np.int32), np.asarray(off), [r["word"] for r in rows], np.full(len(rows), 2.44, np.float32))
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode, key_units=True, n_points=m.cfg.n_frames)
    return torch.cat([x[None] for x, _ in ds])


def mean_memory(m, mode, kb, alphabet):
    val = SwipeCorpus.load(ROOT / "data/canonical/futo/validation", alphabet, limit=2000)
    with torch.no_grad():
        return m.encode(torch.cat([x[None] for x, _ in SwipeDataset(val, kb, augment_cfg=None, resample_mode=mode, key_units=True, n_points=m.cfg.n_frames)])).mean(0, keepdim=True)


def run_first_pass(ckpt, sets, trie, kb, device, beam=64, shift=(0.0, 0.0)):
    """-> {source: [(cands, ilm_lookup)]}: cands per word as (word, ar, uni, len), ilm dict."""
    m, alphabet, mode = load_ar(str(ROOT / ckpt), device)
    mem = mean_memory(m, mode, kb, alphabet)
    out = {}
    for source, rows in sets.items():
        t0 = time.time()
        feats = features(rows, m, mode, kb, shift)
        with torch.no_grad():
            cands = ar_beam(m, feats, trie, alphabet, beam_width=beam)
        allw = sorted({cw for c in cands for cw, *_ in c})
        ilm = ilm_scores(m, [w for w in allw if len(w) <= m.cfg.max_word_len], alphabet, device, mem, batch=1024)
        out[source] = (cands, ilm)
        print(f"    {ckpt} {source}: {len(cands)} words, {time.time() - t0:.0f}s", flush=True)
    return out


def ranked(c, ilm, alpha=ALPHA, lam=LAM):
    return sorted(((cw, ar + alpha * u + BETA * n - lam * ilm.get(cw, 0.0)) for cw, ar, u, n in c), key=lambda t: -t[1])


def summarize(rows, cands, ilm, alpha, lam):
    top1 = []; in8 = in16 = inb = 0; by = collections.defaultdict(lambda: [0, 0]); first = [0, 0]
    for r, c in zip(rows, cands):
        rk = [w for w, _ in ranked(c, ilm, alpha, lam)]
        ok = bool(rk) and rk[0] == r["word"]; top1.append(ok)
        in8 += r["word"] in rk[:8]; in16 += r["word"] in rk[:16]; inb += r["word"] in rk
        by[r["tag"]][0] += ok; by[r["tag"]][1] += 1
        if r["j"] == 0: first[0] += ok; first[1] += 1
    n = len(rows)
    return dict(top1=top1, acc=100 * sum(top1) / n, in8=100 * in8 / n, in16=100 * in16 / n, inbeam=100 * inb / n,
                by={k: 100 * v[0] / v[1] for k, v in by.items()}, first=100 * first[0] / first[1])


def stage_firstpass(a, sets, trie, kb, device):
    CACHE.mkdir(parents=True, exist_ok=True)
    base = None
    print(f"\n{'checkpoint':<34} {'α':>4} {'iPhone':>7} {'in8':>5} {'in16':>5} {'beam':>5} {'evry':>5} {'tail':>5} {'1st':>5}  | {'FUTO':>5} {'in8':>5} {'beam':>5} {'1st':>5}  | vs shipped (iPhone: +/−, p ; FUTO: +/−, p)")
    for ckpt in a.ckpts:
        if not (ROOT / ckpt).exists(): print(f"  (missing {ckpt})"); continue
        cache = CACHE / (ckpt.replace("/", "_") + f".b{a.beam}.pkl")
        if cache.exists(): res = pickle.load(open(cache, "rb"))
        else:
            res = run_first_pass(ckpt, sets, trie, kb, device, beam=a.beam); pickle.dump(res, open(cache, "wb"))
        for alpha in (0.6, 0.4):
            s = {src: summarize(sets[src], *res[src], alpha, LAM) for src in sets}
            if base is None and ckpt == a.ckpts[0] and alpha == 0.6: base = s
            cmp = ""
            if base is not None and not (ckpt == a.ckpts[0] and alpha == 0.6):
                bc, bf = mcnemar(base["capture"]["top1"], s["capture"]["top1"]), mcnemar(base["futo"]["top1"], s["futo"]["top1"])
                cmp = f"{bc[0]}/{bc[1]} p={bc[2]:.2f} ; {bf[0]}/{bf[1]} p={bf[2]:.2f}"
            c, f = s["capture"], s["futo"]
            print(f"{ckpt.replace('runs/', '').replace('/ar_decoder', ''):<34} {alpha:4.1f} {c['acc']:7.1f} {c['in8']:5.1f} {c['in16']:5.1f} {c['inbeam']:5.1f} {c['by'].get('everyday', 0):5.1f} {c['by'].get('tail', 0):5.1f} {c['first']:5.1f}  | {f['acc']:5.1f} {f['in8']:5.1f} {f['inbeam']:5.1f} {f['first']:5.1f}  | {cmp}", flush=True)


def stage_offset(a, sets, trie, kb, device):
    for src, rows in sets.items():
        ds_ = []; de_ = []
        for r in rows:
            w = r["word"]
            sx, sy = key_center(w[0]); ex, ey = key_center(w[-1])
            ds_.append(((r["x"][0] - sx) * 10, (r["y"][0] - sy) * 3)); de_.append(((r["x"][-1] - ex) * 10, (r["y"][-1] - ey) * 3))
        ds_ = np.array(ds_); de_ = np.array(de_)
        print(f"{src}: start offset mean ({ds_[:, 0].mean():+.3f}, {ds_[:, 1].mean():+.3f}) median ({np.median(ds_[:, 0]):+.3f}, {np.median(ds_[:, 1]):+.3f}) keys; "
              f"end mean ({de_[:, 0].mean():+.3f}, {de_[:, 1].mean():+.3f}) median ({np.median(de_[:, 0]):+.3f}, {np.median(de_[:, 1]):+.3f}); "
              f"|start| > 1 key: {100 * (np.hypot(*ds_.T) > 1).mean():.0f}%")
        if src == "capture":
            mean_dx = (ds_[:, 0].mean() + de_[:, 0].mean()) / 2 / 10; mean_dy = (ds_[:, 1].mean() + de_[:, 1].mean()) / 2 / 3
    res0 = pickle.load(open(CACHE / (SHIPPED.replace("/", "_") + ".b64.pkl"), "rb"))
    base = {src: summarize(sets[src], *res0[src], ALPHA, LAM) for src in sets}
    for label, shift in (("remove mean offset (x,y)", (-mean_dx, -mean_dy)), ("remove y only", (0.0, -mean_dy)), ("y −0.1 key", (0.0, -0.1 / 3)), ("y +0.1 key", (0.0, 0.1 / 3))):
        res = run_first_pass(SHIPPED, sets, trie, kb, device, shift=shift)
        s = {src: summarize(sets[src], *res[src], ALPHA, LAM) for src in sets}
        bc, bf = mcnemar(base["capture"]["top1"], s["capture"]["top1"]), mcnemar(base["futo"]["top1"], s["futo"]["top1"])
        print(f"  {label:<28} shift=({shift[0] * 10:+.3f},{shift[1] * 3:+.3f}) keys: iPhone {s['capture']['acc']:.1f} (base {base['capture']['acc']:.1f}; {bc[0]}/{bc[1]} p={bc[2]:.2f})  FUTO {s['futo']['acc']:.1f} (base {base['futo']['acc']:.1f}; {bf[0]}/{bf[1]} p={bf[2]:.2f})", flush=True)


def fused_decode(slots, lm, mu=MU, mu0=None, beam=FBEAM, lag=1):
    mu0 = mu if mu0 is None else mu0
    states = [((), 0.0)]
    for t, (_ref, cands) in enumerate(slots):
        ctxs = [" ".join(s) for s, _ in states]
        lm.fill([(c, cw) for c in ctxs for cw, _ in cands])
        priors = {cw: lm.prior(cw) for cw, _ in cands}
        exp = {}
        for (words, cum), ctx in zip(states, ctxs):
            for cw, ac in cands:
                delta = lm.cache[(ctx, cw)] - priors[cw]
                sc = ac + (mu0 if t == 0 else mu) * delta
                wt = words + (cw,)
                if wt not in exp or cum + sc > exp[wt]: exp[wt] = cum + sc
        states = sorted(exp.items(), key=lambda kv: -kv[1])[:beam]
        if lag is not None and t - lag >= 0:
            j = t - lag; keep = states[0][0][j]
            states = [s for s in states if s[0][j] == keep] or states[:1]
    return list(states[0][0])


def stage_tscale(a, sets, trie, kb, device):
    """The iPhone gestures are ~2× faster than the corpus donors'. Does the encoder care about absolute speed?"""
    res0 = pickle.load(open(CACHE / (SHIPPED.replace("/", "_") + ".b64.pkl"), "rb"))
    base = {src: summarize(sets[src], *res0[src], ALPHA, LAM) for src in sets}
    for k in (2.0, 1.5, 0.5):
        scaled = {src: [dict(r, t=(r["t"] * k).astype(np.int32)) for r in rows] for src, rows in sets.items()}
        res = run_first_pass(SHIPPED, scaled, trie, kb, device)
        s = {src: summarize(sets[src], *res[src], ALPHA, LAM) for src in sets}
        bc, bf = mcnemar(base["capture"]["top1"], s["capture"]["top1"]), mcnemar(base["futo"]["top1"], s["futo"]["top1"])
        print(f"  t × {k}: iPhone {s['capture']['acc']:.1f} (base {base['capture']['acc']:.1f}; {bc[0]}/{bc[1]} p={bc[2]:.2f})  FUTO {s['futo']['acc']:.1f} (base {base['futo']['acc']:.1f}; {bf[0]}/{bf[1]} p={bf[2]:.2f})", flush=True)


def make_lm(name, device):
    import fused_rescore as fr
    lm = fr.LMScorer(name, torch.device(device))
    if device == "cpu": lm.model = lm.model.float()
    if lm.bos is None: lm.bos = lm.tok.eos_token_id  # Qwen-style: no BOS token
    return lm


def stage_fused(a, sets, trie, kb, device):
    if a.stage == "lmladder":
        base = None
        for name in a.lms:
            print(f"== {name}", flush=True)
            try: lm = make_lm(name, a.lm_device)
            except Exception as e: print(f"  (skipped: {e})"); continue
            base = fused_grid(a, sets, lm, base, variants_override=[(name, dict())])
        return
    fused_grid(a, sets, make_lm("distilgpt2", a.lm_device), None)


def fused_grid(a, sets, lm, base, variants_override=None):
    lists = {}
    for ckpt in a.ckpts:
        cache = CACHE / (ckpt.replace("/", "_") + f".b{a.beam}.pkl")
        if not cache.exists(): print(f"  (no cached lists for {ckpt}; run firstpass)"); continue
        lists[ckpt] = pickle.load(open(cache, "rb"))

    def sentences(ckpt, src, alpha, lam, m):
        cands, ilm = lists[ckpt][src]; rows = sets[src]
        slots = collections.defaultdict(list)
        for r, c in zip(rows, cands): slots[r["sid"]].append((r["word"], ranked(c, ilm, alpha, lam)[:m]))
        return [slots[k] for k in sorted(slots)]

    def read(ckpt, alpha=ALPHA, lam=LAM, m=M, mu=MU, mu0=None, beam=FBEAM):
        out = {}
        for src in sets:
            ok = []; by = collections.defaultdict(lambda: [0, 0]); first = [0, 0]
            for sl in sentences(ckpt, src, alpha, lam, m):
                dec = fused_decode(sl, lm, mu=mu, mu0=mu0, beam=beam)
                for j, ((ref, _), o) in enumerate(zip(sl, dec)):
                    ok.append(o == ref)
                    if j == 0: first[0] += o == ref; first[1] += 1
            # tags per word in sentence order
            tags = [r["tag"] for r in sorted(sets[src], key=lambda r: (r["sid"], r["j"]))]
            for t_, o in zip(tags, ok): by[t_][0] += o; by[t_][1] += 1
            out[src] = dict(ok=ok, acc=100 * sum(ok) / len(ok), by={k: 100 * v[0] / v[1] for k, v in by.items()}, first=100 * first[0] / first[1])
        return out

    variants = [("shipped (M8 μ0.8 α0.6 λ0.25 beam8)", dict()),
                ("M 16", dict(m=16)), ("M 24", dict(m=24)),
                ("μ 0.6", dict(mu=0.6)), ("μ 1.0", dict(mu=1.0)), ("μ 1.2", dict(mu=1.2)),
                ("first-word μ 0.4", dict(mu0=0.4)), ("first-word μ 0", dict(mu0=0.0)),
                ("α 0.4", dict(alpha=0.4)), ("λ 0", dict(lam=0.0)),
                ("fused beam 16", dict(beam=16)), ("fused beam 4", dict(beam=4)),
                ("M 16, μ 1.0", dict(m=16, mu=1.0)), ("M 16, first-word μ 0.4", dict(m=16, mu0=0.4)),
                ("M 24, μ 1.0, first-word μ 0.4", dict(m=24, mu=1.0, mu0=0.4))]
    if variants_override is not None: variants = variants_override
    if base is None:
        print(f"\n{'variant':<34} {'iPhone':>7} {'evry':>5} {'tail':>5} {'1st':>5}  | {'FUTO':>5} {'1st':>5}  | vs shipped (iPhone +/−, p ; FUTO +/−, p)")
    for ckpt in lists:
        if len(lists) > 1: print(f"-- lists from {ckpt}")
        for name, kw in (variants if ckpt == a.ckpts[0] else variants[:1] + [("M 16, μ 1.0", dict(m=16, mu=1.0))]):
            t0 = time.time(); s = read(ckpt, **kw)
            if base is None: base = s
            bc, bf = mcnemar(base["capture"]["ok"], s["capture"]["ok"]), mcnemar(base["futo"]["ok"], s["futo"]["ok"])
            c, f = s["capture"], s["futo"]
            print(f"{name:<34} {c['acc']:7.1f} {c['by'].get('everyday', 0):5.1f} {c['by'].get('tail', 0):5.1f} {c['first']:5.1f}  | {f['acc']:5.1f} {f['first']:5.1f}  | {bc[0]}/{bc[1]} p={bc[2]:.2f} ; {bf[0]}/{bf[1]} p={bf[2]:.2f}   ({time.time() - t0:.0f}s)", flush=True)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["firstpass", "offset", "fused", "tscale", "lmladder"])
    ap.add_argument("--lms", nargs="+", default=["distilgpt2", "gpt2", "gpt2-medium", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3.5-0.8B-Base"])
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    ap.add_argument("--beam", type=int, default=64)
    ap.add_argument("--lm-device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device("cpu"); kb = KeyboardLayout.qwerty()
    lex = build_lexicon("train+wf320k", ROOT / "data/canonical", ALPHABET, 1.0); trie = FlatTrie(lex, ALPHABET)
    sets = load_sets()
    if a.stage == "firstpass":
        oov = {src: [r["word"] for r in rows if r["word"] not in lex] for src, rows in sets.items()}
        for src, ws in oov.items(): print(f"{src}: {len(sets[src])} words, {len(ws)} not in lexicon: {ws}")
    {"firstpass": stage_firstpass, "offset": stage_offset, "fused": stage_fused, "tscale": stage_tscale, "lmladder": stage_fused}[a.stage](a, sets, trie, kb, device)


if __name__ == "__main__":
    main()
