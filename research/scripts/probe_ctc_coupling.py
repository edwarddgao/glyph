"""Does CTC's factorized emission model fail on invariant input?

Congruence classes make the letter-sequence posterior *coupled*-multimodal:
an `is` gesture is exactly an `od` gesture, so ideal frame posteriors split
i/o at dwell 1 and s/d at dwell 2 -- but CTC frames are independent given the
features, so any split leaks equal mass onto the cross-terms `id` and `os`.
Alternatively the encoder resolves the class with its implicit LM (one-hot
`is`), absorbing the lexical prior into the emissions.

Probe: full-alignment CTC scores of truth vs congruent twin vs cross-terms,
under the canonical and the shape-only encoder, on real validation swipes.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus, SwipeDataset
from eval_decoder import load_model, pick_device

device = pick_device("auto")
kb = KeyboardLayout.qwerty()
corpus = SwipeCorpus.load("data/canonical/futo/validation", kb.letters,
                          limit=20000)

# truth -> candidates (congruent twin first, then cross-terms / near-twins)
PROBES = {
    "is": ["od", "id", "os", "of"],   # exact twin od; cross id/os; near-twin of
    "on": ["in"],                     # single-position class, both trained
    "in": ["on"],
    "he": ["gw", "ge", "hw"],         # twin gw untrained; cross ge/hw
    "at": ["du", "au", "dt"],
}
MAX_PER_WORD = 100

idx_by_word = {w: [] for w in PROBES}
for i, w in enumerate(corpus.words):
    if w in idx_by_word and len(idx_by_word[w]) < MAX_PER_WORD:
        idx_by_word[w].append(i)

sub_idx = sorted(i for lst in idx_by_word.values() for i in lst)

def emissions(ckpt):
    model, alphabet, key_units, mode = load_model(ckpt, device)
    ds = SwipeDataset(corpus, kb, augment_cfg=None, resample_mode=mode,
                      key_units=key_units, shape_only=model.cfg.shape_only)
    out = {}
    with torch.no_grad():
        for s in range(0, len(sub_idx), 256):
            batch = sub_idx[s:s + 256]
            x = torch.stack([ds[i][0] for i in batch])
            lp = model(x.to(device)).float().cpu().numpy()
            for j, i in enumerate(batch):
                out[i] = lp[j]
    return out, alphabet, model.cfg.blank

def ctc_score(lp, word, alphabet, blank):
    x = torch.from_numpy(lp).unsqueeze(1)
    tgt = torch.tensor([[alphabet.index(c) for c in word]])
    return -float(F.ctc_loss(x, tgt, torch.tensor([lp.shape[0]]),
                             torch.tensor([len(word)]), blank=blank,
                             reduction="sum", zero_infinity=True))

for name, ckpt in [("canonical", "runs/full/encoder.pt"),
                   ("shape-only", "runs/shape10/encoder.pt")]:
    lps, alphabet, blank = emissions(ckpt)
    print(f"\n==== {name} ====")
    print(f"  {'truth':<6}{'cand':<6}{'mean dLogP (truth-cand)':>26}"
          f"{'cand wins':>12}   n")
    for w, cands in PROBES.items():
        for cand in cands:
            deltas = []
            for i in idx_by_word[w]:
                st = ctc_score(lps[i], w, alphabet, blank)
                sc = ctc_score(lps[i], cand, alphabet, blank)
                deltas.append(st - sc)
            d = np.array(deltas)
            print(f"  {w:<6}{cand:<6}{d.mean():>20.1f} nats"
                  f"{(d < 0).mean():>11.1%}   {len(d)}")
