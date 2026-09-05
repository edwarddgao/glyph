#!/usr/bin/env python3
"""Per-user adaptation ceiling: what would this user's own practice data buy?

    .venv/bin/python scripts/probe_user_adapt.py [--epochs 40] [--lr 1e-4]

The shipped encoder (`runs/ar_mixed_s1`) never saw the 543 real-iPhone
gestures. Two-fold by sentence: fine-tune on one half (with the training
augmentation), read the trie-beam first pass on the other half, both ways,
paired against the un-tuned model on the same words; the 1,337-word FUTO
replay set is read alongside to see what the adaptation costs everyone else.
Two arms: the user's swipes alone, and the user's swipes with a 4:1 replay of
FUTO training swipes (the usual guard against drift). This is the regime the
practice run's records would feed — the sloppy 30% that no training-data lever
in the log reaches (#58, README "Which encoder generalizes to a real iPhone?").
"""
from __future__ import annotations

import argparse, copy, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from swipe_typing.layout import ALPHABET, KeyboardLayout  # noqa: E402
from swipe_typing.model import SwipeDataset  # noqa: E402
from swipe_typing.model.ar import FlatTrie, ar_loss  # noqa: E402
from swipe_typing.model.data import SwipeCorpus, collate  # noqa: E402
from swipe_typing.augment import DEFAULT as DEFAULT_AUG  # noqa: E402
from eval_decoder import build_lexicon  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
import eval_phone_levers as L  # noqa: E402


def corpus_of(rows):
    xs, ys, ts, off = [], [], [], [0]
    for r in rows:
        xs.extend(r["x"]); ys.extend(r["y"]); ts.extend(r["t"]); off.append(len(xs))
    return SwipeCorpus(np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ts, np.int32), np.asarray(off), [r["word"] for r in rows], np.full(len(rows), 2.44, np.float32))


def finetune(ckpt_path, rows, replay, kb, epochs, lr, device, out):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m, alphabet, mode = load_ar(ckpt_path, device)
    m.train()
    user = SwipeDataset(corpus_of(rows), kb, augment_cfg=DEFAULT_AUG, resample_mode=mode, key_units=True, n_points=m.cfg.n_frames)
    sets = [user]
    if replay is not None:
        sets.append(SwipeDataset(replay, kb, augment_cfg=DEFAULT_AUG, resample_mode=mode, key_units=True, n_points=m.cfg.n_frames))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for ep in range(epochs):
        for ds in sets: ds.seed = ep * 7919  # fresh augmentation draw per epoch
        idx = torch.randperm(len(user), generator=g).tolist()
        rep_idx = torch.randperm(len(sets[1]), generator=g).tolist() if replay is not None else []
        bs = 32; tot = 0.0; n = 0
        for b in range(0, len(idx), bs):
            batch = [user[i] for i in idx[b:b + bs]]
            if replay is not None:
                k = 4 * len(batch); batch += [sets[1][i] for i in rep_idx[:k]]; rep_idx = rep_idx[k:]
            x, tg, ln = collate(batch)
            loss = ar_loss(m, x.to(device), tg, ln)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        if ep % 10 == 9 or ep == epochs - 1: print(f"      epoch {ep + 1}: loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    m.eval()
    ckpt = dict(ckpt); ckpt["model"] = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.save(ckpt, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    kb = KeyboardLayout.qwerty(); cpu = torch.device("cpu")
    lex = build_lexicon("train+wf320k", ROOT / "data/canonical", ALPHABET, 1.0); trie = FlatTrie(lex, ALPHABET)
    sets = L.load_sets(); cap = sets["capture"]
    base = L.run_first_pass(L.SHIPPED, sets, trie, kb, cpu)
    base_s = {src: L.summarize(sets[src], *base[src], L.ALPHA, L.LAM) for src in sets}
    print(f"base: iPhone {base_s['capture']['acc']:.1f}  FUTO {base_s['futo']['acc']:.1f}")
    replay = SwipeCorpus.load(ROOT / "data/canonical/futo_clean/train", ALPHABET, limit=20000)
    out_dir = L.CACHE / "adapt"; out_dir.mkdir(parents=True, exist_ok=True)
    for arm, rep in (("user only", None), ("user + FUTO replay 1:4", replay)):
        print(f"== {arm}")
        cap_ok = [None] * len(cap); futo_ok = []
        by = {"everyday": [0, 0], "tail": [0, 0]}
        for fold in (0, 1):
            train = [r for r in cap if r["sid"] % 2 == fold]; test_idx = [i for i, r in enumerate(cap) if r["sid"] % 2 != fold]
            print(f"   fold {fold}: train {len(train)} swipes, test {len(test_idx)}")
            path = finetune(str(ROOT / L.SHIPPED), train, rep, kb, a.epochs, a.lr, torch.device(a.device), out_dir / f"{arm[:4].strip()}_{fold}.pt")
            res = L.run_first_pass(str(path), {"capture": [cap[i] for i in test_idx], "futo": sets["futo"]}, trie, kb, cpu)
            s_cap = L.summarize([cap[i] for i in test_idx], *res["capture"], L.ALPHA, L.LAM)
            for i, ok in zip(test_idx, s_cap["top1"]): cap_ok[i] = ok
            s_futo = L.summarize(sets["futo"], *res["futo"], L.ALPHA, L.LAM); futo_ok.append(s_futo["top1"])
            print(f"      held-out half: {s_cap['acc']:.1f} (everyday {s_cap['by'].get('everyday', 0):.1f} tail {s_cap['by'].get('tail', 0):.1f})  FUTO {s_futo['acc']:.1f}", flush=True)
        b, c, p = L.mcnemar(base_s["capture"]["top1"], cap_ok)
        for r, ok in zip(cap, cap_ok): by[r["tag"]][0] += ok; by[r["tag"]][1] += 1
        fb = [L.mcnemar(base_s["futo"]["top1"], f) for f in futo_ok]
        print(f"   {arm}: iPhone {100 * sum(cap_ok) / len(cap_ok):.1f} vs base {base_s['capture']['acc']:.1f} ({b}/{c}, p={p:.3f}); everyday {100 * by['everyday'][0] / by['everyday'][1]:.1f} tail {100 * by['tail'][0] / by['tail'][1]:.1f}; "
              f"FUTO per fold {100 * np.mean(futo_ok[0]):.1f} / {100 * np.mean(futo_ok[1]):.1f} vs {base_s['futo']['acc']:.1f} (p={fb[0][2]:.2f} / {fb[1][2]:.2f})", flush=True)


if __name__ == "__main__":
    main()
