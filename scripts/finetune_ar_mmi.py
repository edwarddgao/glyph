#!/usr/bin/env python3
"""MMI fine-tune for the AR decoder: train the sequence scores for the search.

    python scripts/finetune_ar_mmi.py --out runs/ar_mmi

The #26 recipe ported to the AR head, where it is *simpler* than it was for
CTC: the model scores any word exactly via teacher forcing, so the loss is a
plain softmax over each swipe's rival set,

    loss = -log  exp(s(truth)) / sum_j exp(s(candidate_j))

with the truth always injected into the denominator (AR can score words the
beam missed; the CTC version needed the same injection to bound the loss).
Rivals come from the model's own dumped n-best lists (dump_ar_nbest.py).

As with #26, greedy may drop while everything under search rises; judge
checkpoints by beam top-1 via eval_ar_decoder.py, not the epoch metric. The
epoch metric here is a cheap proxy: re-ranking the *fixed* dumped validation
lists with the updated scores (the real beam moves too, and historically
moves further).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from eval_decoder import pick_device  # noqa: E402
from eval_ar_decoder import load_ar  # noqa: E402
from swipe_typing.model.ar import PAD, shift_targets  # noqa: E402


def batch_scores(model, memory_rows, words, alphabet, device):
    """Teacher-forced log P(word) for rows whose memory is precomputed."""
    char = {c: i for i, c in enumerate(alphabet)}
    targets = torch.cat([
        torch.tensor([char[c] for c in w], dtype=torch.long) for w in words
    ])
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    tgt_in, tgt_out = shift_targets(targets, lengths, model.cfg)
    logits = model.decode_step(memory_rows, tgt_in.to(device))
    lp = F.log_softmax(logits, dim=-1)
    tgt = tgt_out.to(device)
    mask = (tgt != PAD).float()
    picked = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ar_full/ar_decoder.pt")
    ap.add_argument("--nbest", default="data/nbest_ar")
    ap.add_argument("--train", default="futo_train.npz")
    ap.add_argument("--val", default="futo_validation.npz")
    ap.add_argument("--out", default="runs/ar_mmi")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=48,
                    help="swipes per step; each carries its rival set")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, alphabet, _mode = load_ar(args.checkpoint, device)
    model.train()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    data = np.load(Path(args.nbest) / args.train, allow_pickle=False)
    feats = data["features"]
    cands, valid, words = data["candidates"], data["valid"], data["words"]
    n = len(words)
    print(f"train lists: {n:,} swipes")

    # Rival sets: dumped candidates plus the truth, deduped, truth first.
    sets: list[list[str]] = []
    for i in range(n):
        rivals = [str(w) for w, v in zip(cands[i], valid[i]) if v]
        w = str(words[i])
        sets.append([w] + [r for r in rivals if r != w])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    val = np.load(Path(args.nbest) / args.val, allow_pickle=False)

    @torch.no_grad()
    def rerank_proxy(limit=20000):
        """Top-1 when the fixed dumped val lists are re-ranked by new scores."""
        model.eval()
        correct = 0
        m = min(limit, len(val["words"]))
        for s in range(0, m, 128):
            idx = range(s, min(s + 128, m))
            rows, row_words, row_of = [], [], []
            for i in idx:
                for w, v in zip(val["candidates"][i], val["valid"][i]):
                    if v:
                        rows.append(i - s)
                        row_words.append(str(w))
                        row_of.append(i)
            x = torch.from_numpy(val["features"][s:min(s + 128, m)]).to(device)
            memory = model.encode(x)
            sc = batch_scores(model, memory[rows], row_words, alphabet, device)
            sc = sc.cpu().numpy()
            best: dict[int, tuple[float, str]] = {}
            for r, w, i in zip(sc, row_words, row_of):
                if i not in best or r > best[i][0]:
                    best[i] = (r, w)
            correct += sum(b[1] == str(val["words"][i])
                           for i, b in best.items())
        model.train()
        return correct / m

    step = 0
    order = np.random.default_rng(0).permutation(n)
    for epoch in range(args.epochs):
        t0, running, seen = time.time(), 0.0, 0
        for s in range(0, n, args.batch_size):
            batch = order[s:s + args.batch_size]
            rows, row_words, row_of = [], [], []
            for j, i in enumerate(batch):
                for w in sets[i]:
                    rows.append(j)
                    row_words.append(w)
                    row_of.append(j)
            x = torch.from_numpy(feats[batch]).to(device)
            memory = model.encode(x)
            sc = batch_scores(model, memory[rows], row_words, alphabet, device)

            loss = 0.0
            row_of_t = torch.tensor(row_of, device=device)
            for j in range(len(batch)):
                s_j = sc[row_of_t == j]
                loss = loss - F.log_softmax(s_j, dim=0)[0]  # truth is row 0
            loss = loss / len(batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            running += float(loss.detach())
            seen += 1
            if step % args.log_every == 0:
                rate = seen * args.batch_size / (time.time() - t0)
                print(f"  e{epoch} step {step} loss {running / seen:.4f} "
                      f"{rate:.0f} swipes/s", flush=True)

        proxy = rerank_proxy()
        print(f"epoch {epoch}: loss {running / max(seen, 1):.4f}  "
              f"fixed-list rerank top-1 {proxy:.4f}", flush=True)
        torch.save(
            {"model": model.state_dict(), "cfg": ckpt["cfg"],
             "alphabet": alphabet, "args": ckpt.get("args", {})},
            out / f"ar_decoder_ep{epoch}.pt",
        )
    print(f"saved {out}/ar_decoder_ep*.pt")


if __name__ == "__main__":
    main()
