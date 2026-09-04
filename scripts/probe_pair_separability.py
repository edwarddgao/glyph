#!/usr/bin/env python3
"""Are the fused stack's misranked in-list errors separable from the gesture?

    python scripts/probe_pair_separability.py --hyps runs/hyps_base_val.npz

#74/#75 established that the encoder is saturated under this data, which is
not the same as the residual being non-acoustic. After the out-of-lexicon
words (~1.2 pts) and the never-surfaced words (~0.6) the largest bucket is
words the search *had* in its list and ranked below a rival (~3.0). This
probe asks, pair by pair, whether those rivals are distinguishable at all
from the gesture:

  1. taxonomy — for each val word: correct / misranked (truth in list) /
     not surfaced. Within misranked, split by whether the AR acoustic score
     alone preferred the truth (the prior or LM overrode it) or the rival
     (the acoustics were wrong).
  2. confusion pairs — unordered {truth, hyp} pairs by count.
  3. dedicated two-word classifiers per frequent pair, trained on FUTO
     *train* gestures of exactly those two words (the encoder's own input
     features, no augmentation), evaluated on every *validation* gesture of
     the pair (session-disjoint by construction). Balanced accuracy and AUC
     are prior-free readouts of separability; accuracy on the specific
     misranked instances says whether the signal the stack missed is there.
     Three learners so no single inductive bias carries the answer:
     regularised logistic regression, a small MLP, and 1-NN on resampled
     positions (nonparametric).

Reading: pairwise balanced accuracy near 50% means the two words are swiped
identically and the bucket is irreducible without new information; well above
the stack's own rate on the same instances means acoustic signal exists that
the encoder does not use.
"""
from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from swipe_typing import features
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus, SwipeDataset


def balanced_acc(y, pred):
    accs = [np.mean(pred[y == c] == c) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(accs))


def auc(y, score):
    """Rank AUC of score for class 1."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_linear(Xtr, ytr, hidden=0, epochs=300, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    d = Xtr.shape[1]
    if hidden:
        model = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(),
                                    torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1))
    else:
        model = torch.nn.Linear(d, 1)
    # class-balanced loss so the unigram ratio of the pair does not leak in
    pos = ytr_t.mean().clamp(1e-3, 1 - 1e-3)
    w = torch.where(ytr_t == 1, 0.5 / pos, 0.5 / (1 - pos))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logit = model(Xtr).squeeze(1)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            logit, ytr_t, reduction="none") * w).mean()
        loss.backward()
        opt.step()
    model.eval()
    return lambda X: model(torch.from_numpy(X).float()).squeeze(1).detach().numpy()


def knn1(Xtr, ytr, Xte):
    # 1-NN, euclidean, chunked
    out = np.empty(len(Xte), dtype=np.int64)
    tr = torch.from_numpy(Xtr).float()
    for s in range(0, len(Xte), 512):
        q = torch.from_numpy(Xte[s:s + 512]).float()
        d = torch.cdist(q, tr)
        out[s:s + 512] = ytr[d.argmin(1).numpy()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="fused_bundle.pkl")
    ap.add_argument("--hyps", default="runs/hyps_base_val.npz")
    ap.add_argument("--cache", default="data/canonical")
    ap.add_argument("--min-pair-count", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--max-train-per-word", type=int, default=4000)
    ap.add_argument("--min-train-per-word", type=int, default=30)
    args = ap.parse_args()

    kb = KeyboardLayout.qwerty()
    bundle = pickle.load(open(args.bundle, "rb"))
    refs = bundle["refs"]
    lists = bundle["lists"]
    hyps = np.load(args.hyps, allow_pickle=True)["joint"]
    n = len(refs)
    assert len(hyps) == n
    print(f"bundle {args.bundle} ({bundle['checkpoint']}), hyps {args.hyps}, n={n}")
    print(f"joint top-1: {np.mean([h == r for h, r in zip(hyps, refs)]):.4f}")

    # --- 1. taxonomy -------------------------------------------------------
    misranked, not_surfaced = [], []
    ac_right = ac_wrong = 0
    for i, (r, h) in enumerate(zip(refs, hyps)):
        if r == h:
            continue
        L = {w: (ar, uni) for w, ar, uni, _ in lists[i]}
        if r in L:
            misranked.append(i)
            if h in L and L[r][0] > L[h][0]:
                ac_right += 1
            else:
                ac_wrong += 1
        else:
            not_surfaced.append(i)
    err = len(misranked) + len(not_surfaced)
    print(f"\nerrors {err} = misranked {len(misranked)} "
          f"({100 * len(misranked) / n:.2f} pts) + not surfaced {len(not_surfaced)} "
          f"({100 * len(not_surfaced) / n:.2f} pts)")
    print(f"  misranked where AR acoustic score alone preferred truth: {ac_right} "
          f"({100 * ac_right / max(len(misranked), 1):.0f}%) — prior/LM overrode the acoustics")
    print(f"  misranked where AR acoustic score preferred the rival:   {ac_wrong} "
          f"({100 * ac_wrong / max(len(misranked), 1):.0f}%) — acoustically misranked")

    # --- 2. confusion pairs ---------------------------------------------------
    pair_idx = defaultdict(list)
    for i in misranked:
        pair_idx[frozenset((refs[i], hyps[i]))].append(i)
    pairs = sorted(pair_idx.items(), key=lambda kv: -len(kv[1]))
    print(f"\n{len(pairs)} distinct confusion pairs; top 25:")
    for p, ix in pairs[:25]:
        a, b = sorted(p)
        print(f"  {a:>10} / {b:<10} {len(ix):>3}")
    covered = sum(len(ix) for p, ix in pairs if len(ix) >= args.min_pair_count)
    print(f"pairs with count >= {args.min_pair_count}: "
          f"{sum(len(ix) >= args.min_pair_count for _, ix in pairs)}, "
          f"covering {covered} of {len(misranked)} misranked")

    # --- 3. dedicated pairwise classifiers -------------------------------------
    chosen = [(tuple(sorted(p)), ix) for p, ix in pairs
              if len(ix) >= args.min_pair_count][: args.max_pairs]
    vocab = {w for p, _ in chosen for w in p}

    t0 = time.time()
    val = SwipeCorpus.load(Path(args.cache) / "futo/validation", kb.letters)
    assert list(val.words[:n]) == list(refs), "bundle order != corpus order"
    train = SwipeCorpus.load(Path(args.cache) / "futo/train", kb.letters)
    print(f"\ncorpora loaded ({time.time() - t0:.0f}s): val {len(val.words):,}, "
          f"train {len(train.words):,}")
    tr_idx, va_idx = defaultdict(list), defaultdict(list)
    for i, w in enumerate(train.words):
        if w in vocab:
            tr_idx[w].append(i)
    for i, w in enumerate(val.words):
        if w in vocab:
            va_idx[w].append(i)

    ds_tr = SwipeDataset(train, kb, augment_cfg=None)
    ds_va = SwipeDataset(val, kb, augment_cfg=None)
    feat_cache = {}

    def feats(ds, corpus, idx, tag):
        rows, pos = [], []
        for i in idx:
            key = (tag, i)
            if key not in feat_cache:
                x, _ = ds[i]
                xy = features.resample(corpus.points(i), corpus.times(i),
                                       n=features.N_POINTS, mode="time")
                feat_cache[key] = (x.numpy().reshape(-1), xy.reshape(-1))
            f, p = feat_cache[key]
            rows.append(f)
            pos.append(p)
        return np.stack(rows), np.stack(pos)

    rng = np.random.default_rng(0)
    print(f"\n{'pair':>22} {'ntr':>9} {'nval':>5} | {'LR':>5} {'MLP':>5} {'1NN':>5} "
          f"{'AUC':>5} | {'stack':>5} | misranked recovered LR/MLP/1NN")
    tot_mis = tot_lr = tot_mlp = tot_nn = 0
    ba_rows = []
    for (a, b), mis in chosen:
        if min(len(tr_idx[a]), len(tr_idx[b])) < args.min_train_per_word:
            print(f"{a + '/' + b:>22}  skipped (train {len(tr_idx[a])}/{len(tr_idx[b])})")
            continue
        ia = rng.permutation(tr_idx[a])[: args.max_train_per_word]
        ib = rng.permutation(tr_idx[b])[: args.max_train_per_word]
        Xtr, Ptr = feats(ds_tr, train, list(ia) + list(ib), "tr")
        ytr = np.array([0] * len(ia) + [1] * len(ib))
        va = va_idx[a] + va_idx[b]
        Xva, Pva = feats(ds_va, val, va, "va")
        yva = np.array([0] * len(va_idx[a]) + [1] * len(va_idx[b]))
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd

        lr = fit_linear(Xtr_n, ytr)
        mlp = fit_linear(Xtr_n, ytr, hidden=128, epochs=400, seed=1)
        s_lr, s_mlp = lr(Xva_n), mlp(Xva_n)
        p_lr, p_mlp = (s_lr > 0).astype(int), (s_mlp > 0).astype(int)
        p_nn = knn1(Ptr, ytr, Pva)
        ba = (balanced_acc(yva, p_lr), balanced_acc(yva, p_mlp), balanced_acc(yva, p_nn))
        au = auc(yva, s_mlp)

        # the stack on the same instances: val-slice gestures of a or b whose
        # hyp is a or b (a two-way decision it actually faced)
        slice_ix = [i for i in va if i < n and hyps[i] in (a, b)]
        stack_acc = (np.mean([hyps[i] == refs[i] for i in slice_ix])
                     if slice_ix else float("nan"))

        # the misranked instances themselves
        pos_in_va = {i: k for k, i in enumerate(va)}
        mk = [pos_in_va[i] for i in mis]
        truth = yva[mk]
        rec = (int((p_lr[mk] == truth).sum()), int((p_mlp[mk] == truth).sum()),
               int((p_nn[mk] == truth).sum()))
        tot_mis += len(mis)
        tot_lr += rec[0]
        tot_mlp += rec[1]
        tot_nn += rec[2]
        ba_rows.append((len(mis),) + ba)
        print(f"{a + '/' + b:>22} {len(ia):>4}/{len(ib):<4} {len(va):>5} | "
              f"{ba[0]:.3f} {ba[1]:.3f} {ba[2]:.3f} {au:.3f} | {stack_acc:.3f} | "
              f"{rec[0]}/{rec[1]}/{rec[2]} of {len(mis)}")

    if tot_mis:
        w = np.array([r[0] for r in ba_rows], dtype=float)
        ba_w = [float((w * np.array([r[k] for r in ba_rows])).sum() / w.sum())
                for k in (1, 2, 3)]
        print(f"\nweighted (by misranked count) balanced accuracy: "
              f"LR {ba_w[0]:.3f}  MLP {ba_w[1]:.3f}  1NN {ba_w[2]:.3f}")
        print(f"misranked instances recovered: LR {tot_lr}/{tot_mis} "
              f"({100 * tot_lr / tot_mis:.0f}%)  MLP {tot_mlp}/{tot_mis} "
              f"({100 * tot_mlp / tot_mis:.0f}%)  1NN {tot_nn}/{tot_mis} "
              f"({100 * tot_nn / tot_mis:.0f}%)   [chance 50%, stack 0% by construction]")


if __name__ == "__main__":
    main()
