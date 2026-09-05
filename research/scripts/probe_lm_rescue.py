"""Can a context LM rescue the translation+scale collision classes?

Best-case bound for "the LM works harder" on a shape-only front end:
  - acoustics assumed PERFECT except among geometric colliders (words whose
    normalized templates sit within eps of the truth's) -- there the input is
    (near-)identical and the LM must decide alone;
  - the LM is gpt2-xl, the LM-ladder's best (#34), with ORACLE left AND right
    context (full sentence, truth everywhere except the slot);
  - decision = argmax joint sentence log-prob over {truth} U colliders.
Every one of those choices is generous to the LM, so the result is a ceiling.

Reports, per eps: token accuracy of the LM decision over the 20k val slice
(non-collider tokens count as correct), vs the unigram tie-break ceiling.
"""
import sys
import numpy as np
import torch
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from swipe_typing import features
from swipe_typing.layout import KeyboardLayout
from swipe_typing.model import SwipeCorpus
from eval_decoder import build_lexicon

ASPECT = 2.38
KEY_W = ASPECT / 10.0
N_PTS = 24
MAX_CANDS = 15
EPS_FRACS = (0.25, 0.5)
MODEL = "gpt2-xl"

kb = KeyboardLayout.qwerty()
lex = build_lexicon("train+wf320k", Path("data/canonical"), kb.letters, 1.0)
counts = lex.counts()
words = sorted(counts)
centers = {ch: kb.center(ch) for ch in kb.letters}

def template(word):
    pts = np.array([centers[c] for c in word], dtype=np.float32)
    pts[:, 0] *= ASPECT
    return features.resample(pts, None, n=N_PTS, mode="arclength")

T = np.stack([template(w) for w in words])
lo_, hi_ = T.min(1), T.max(1)
ls = np.maximum((hi_ - lo_).max(1), features.SHAPE_SCALE_FLOOR)
S = ((T - ((lo_ + hi_) / 2.0)[:, None, :]) / ls[:, None, None]).reshape(len(words), -1)
sqS = (S * S).sum(1)
cnt = np.array([counts[w] for w in words], dtype=np.float64)

corpus = SwipeCorpus.load("data/canonical/futo/validation", kb.letters, limit=20000)
val_counts = Counter(corpus.words)
val_words = sorted(w for w in val_counts if w in counts)
widx = {w: i for i, w in enumerate(words)}
rows = np.array([widx[w] for w in val_words])

# Collider sets per unique val word, at the widest eps (subset for tighter).
eps_max = max(EPS_FRACS) * KEY_W
colliders: dict[str, list[tuple[str, float]]] = {}
for s in range(0, len(rows), 512):
    idx = rows[s:s + 512]
    d2 = sqS[idx][:, None] + sqS[None, :] - 2.0 * (S[idx] @ S.T)
    d = np.sqrt(np.maximum(d2, 0) / N_PTS) * ls[idx][:, None]
    d[np.arange(len(idx)), idx] = np.inf
    for j in range(len(idx)):
        w = val_words[s + j]
        near = np.flatnonzero(d[j] < eps_max)
        if len(near):
            order = near[np.argsort(-cnt[near])][:MAX_CANDS]
            colliders[w] = [(words[k], float(d[j][k])) for k in order]
print(f"val words with colliders @{max(EPS_FRACS)} keyw: "
      f"{len(colliders):,}/{len(val_words):,}")

# Tokens needing an LM decision: (sentence, word_idx, truth) with context.
tok = []
n_no_ctx = 0
for i in range(len(corpus)):
    w = corpus.words[i]
    sent, wi = corpus.sentences[i], int(corpus.word_idx[i])
    has_ctx = bool(sent) and wi >= 0 and sent.split()[wi:wi + 1] == [w]
    if w in colliders and not has_ctx:
        n_no_ctx += 1
    tok.append((i, w, sent if has_ctx else None, wi))
print(f"collider tokens without usable context: {n_no_ctx}")

# Score every needed sentence variant once.
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokzr = AutoTokenizer.from_pretrained(MODEL)
tokzr.pad_token = tokzr.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

need: set[str] = set()
variants: dict[tuple[str, int], list[tuple[str, str]]] = {}
for i, w, sent, wi in tok:
    if w not in colliders or sent is None:
        continue
    key = (sent, wi)
    if key in variants:
        continue
    toks = sent.split()
    out = []
    for cand, _d in [(w, 0.0)] + colliders[w]:
        toks2 = toks.copy(); toks2[wi] = cand
        text = " ".join(toks2)
        out.append((cand, text)); need.add(text)
    variants[key] = out
print(f"unique sentence slots: {len(variants):,}  variants to score: {len(need):,}")

texts = sorted(need, key=len)
scores: dict[str, float] = {}
B = 48
with torch.no_grad():
    for s in range(0, len(texts), B):
        batch = texts[s:s + B]
        enc = tokzr(batch, return_tensors="pt", padding=True,
                    padding_side="right")
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device)
        logits = model(ids, attention_mask=mask).logits.float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        got = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
        for b, text in enumerate(batch):
            scores[text] = float(got[b].sum())
        if (s // B) % 50 == 0:
            print(f"  scored {s + len(batch):,}/{len(texts):,}", flush=True)

# Decisions.
for eps_frac in EPS_FRACS:
    eps = eps_frac * KEY_W
    n_correct = n_lm_wrong = n_uni_wrong = n_fallback_wrong = 0
    flips = Counter()
    for i, w, sent, wi in tok:
        cands = [c for c, d in colliders.get(w, []) if d < eps]
        if not cands:
            n_correct += 1
            continue
        uni_ok = all(counts[c] <= counts[w] for c in cands)
        if not uni_ok:
            n_uni_wrong += 1
        if sent is None:
            if uni_ok:
                n_correct += 1
            else:
                n_fallback_wrong += 1; n_lm_wrong += 1
            continue
        pool = [(w, scores[" ".join(sent.split()[:wi] + [w] + sent.split()[wi + 1:])])]
        for c in cands:
            toks2 = sent.split(); toks2[wi] = c
            pool.append((c, scores[" ".join(toks2)]))
        best = max(pool, key=lambda p: p[1])[0]
        if best == w:
            n_correct += 1
        else:
            n_lm_wrong += 1
            flips[(w, best)] += 1
    n = len(tok)
    print(f"\n== eps = {eps_frac} key widths ==")
    print(f"  unigram tie-break ceiling : {1 - n_uni_wrong / n:.2%}")
    print(f"  {MODEL} joint, oracle ctx : {1 - n_lm_wrong / n:.2%}  "
          f"(no-context fallback errors: {n_fallback_wrong})")
    print(f"  most common LM losses:")
    for (w, b), c in flips.most_common(10):
        print(f"    {w:<12} -> {b:<12} x{c}")
