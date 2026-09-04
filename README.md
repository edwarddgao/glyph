# swipe-typing

Normalized loaders, keyboard geometry, and features for the public swipe /
gesture-typing corpora. Every source is mapped into one coordinate space and one
record type, so a model trained on one corpus can be evaluated on another.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev,train,lexicon]"

# data
python scripts/fetch_how_we_swipe.py          # ~70MB from OSF, expands to ~920MB
python scripts/build_cache.py                 # normalize everything -> Parquet
python scripts/validate_alignment.py          # confirm the corpora agree

# model (each section below documents its own script)
python scripts/train_encoder.py --d-model 128 --dilations 1,2,4,8,1,2,4,8 --epochs 10
python scripts/eval_decoder.py --limit 20000

pytest
```

```python
from swipe_typing import cache, features, augment
from swipe_typing.layout import KeyboardLayout

kb = KeyboardLayout.qwerty()
for sw in cache.read("data/canonical/futo/train"):
    sw, kb_aug = augment.augment(sw, kb)      # transforms BOTH together
    x = features.encode(sw)                   # (64, 8) float32
```

## Headline result

Measured once on FUTO's **test** split — 48,711 swipes, 683 donor sessions,
zero session overlap with train or validation, never downloaded or inspected
until the configuration was frozen.

| stage | test top-1 | CER |
|---|---|---|
| greedy, no lexicon | 79.30% | 0.066 |
| + trie-constrained beam search | 92.40% | 0.033 |
| + acoustic rescorer | 92.73% | — |
| + context LM, streaming decoded context | 93.83% | — |
| + context LM, oracle left context | 94.27% | — |
| + deferred commitment, lookahead-1 | 94.34% | — |
| + deferred commitment, joint (CTC stack, freeze 3) | 94.62% | — |
| n-best@8 ceiling (CTC stack) | 97.50% | — |
| fused decoder, streaming (freeze 4) | 93.85% | — |
| fused decoder, lookahead-1 (freeze 4) | 94.72% | — |
| **fused decoder, joint (freeze 4)** | **95.20%** | — |
| deep-list ceiling@24 (freeze 4) | 98.22% | — |

Freeze four replaces the CTC encoder and the three-pass second stage with the
architecture built in #45–49: an autoregressive letter decoder (MMI
fine-tuned, #46/#48) whose trie-constrained beam feeds a *single* fused
sentence-level search — 24 candidates per swipe, delta-form gpt2-xl scoring
completions in-search (#49/#51), commitment expressed as a pruning lag. No
rescorer, no separate deferred pass, one score formula.

The two deferred-commitment rows condition the LM only on the stack's own
decoded words — unlike the 94.27 row, there is no oracle anywhere in them.
Lookahead-1 bounds display latency to one word (the previous word may silently
correct when the next swipe lands, 1.9% of the time); joint may revise any
word in the sentence (2.5%).

Cross-corpus, no fine-tuning: **80.4%** top-1 on How We Swipe (85k swipes,
different apparatus, users and year).

Everything runs on a laptop: a 1.32M-parameter encoder trained in ~65 minutes
on an M-series GPU plus a ~7-minute MMI fine-tune over its own beam's n-best
lists (#26), and a 436k rescorer.

The test split has now been read four times, once per frozen configuration:
the original stack (93.92, #19), the MMI fine-tune (94.27, #28), deferred
commitment (94.62, #35), and the fused AR decoder (95.20, #53). Every time,
every stage landed within ~1–2 SE of its validation estimate, on the
positive side — freeze four at +0.08/+0.09/+0.05 for
streaming/lookahead-1/joint against a prediction of 95.1–95.3 posted before
the read.

### The tuning did not overfit

Every weight in the stack — alpha/beta, lexicon composition, LM weight, rescorer
weight, beam width, prune threshold — was fitted on the same 20k validation
slice, roughly a dozen fits in total. Test was held back to price that.

| stage | validation | test | delta | 1 SE |
|---|---|---|---|---|
| greedy | 0.7840 | 0.7850 | +0.0010 | 0.0019 |
| beam top-1 | 0.9186 | 0.9200 | +0.0014 | 0.0012 |
| top-8 ceiling | 0.9705 | 0.9705 | +0.0000 | 0.0008 |
| + rescorer | 0.9240 | 0.9248 | +0.0008 | 0.0012 |
| + rescorer + LM | 0.9381 | 0.9392 | +0.0011 | 0.0011 |

Every stage lands within 1.2 SE, and every delta is *positive*. The prediction
going in was that the full stack would give back 0.1-0.3 points, since the
second-pass weights are the most-tuned part; it gained 0.11 instead.

The second freeze (MMI config, #26–28) repeated the pattern exactly:
first pass 92.31 validation → 92.39 test, full stack 94.10 → 94.27. The third
(deferred commitment, #30–35) again: streaming 93.67 → 93.83, lookahead-1
94.12 → 94.34, joint 94.54 → 94.62, with the prediction posted before the
read. Three configurations, every stage-level comparison positive.

The reason is visible in the earlier sweeps: every tuned surface was flat.
alpha/beta spanned under one point across the entire grid, the LM weight curve
was flat from 0.5 to 1.0, and accuracy rose monotonically with lexicon size.
Flat optima cannot be overfit — there was nothing sharp to latch onto. The
protocol still mattered, because that is only knowable after checking.

## Datasets

| source | swipes | words | sessions | lang | license |
|---|---|---|---|---|---|
| [FUTO `swipe.futo.org`](https://huggingface.co/datasets/futo-org/swipe.futo.org) | 1,043,789 (swipe-1) | 108,759 | 12k+ | en | MIT |
| [How We Swipe](https://osf.io/sj67f/) | ~109k | 11,318 | 1,338 | en | see OSF |

Recommended split of labour: **train on FUTO, evaluate on How We Swipe.** The
second corpus is 10x smaller and was collected on different apparatus, which is
exactly what makes it a good held-out set — it measures transfer rather than
memorization of one capture pipeline. `build_cache.py` therefore writes How We
Swipe as a `test` split by default.

FUTO's own train/dev/test split is preserved because it is partitioned **by donor
session**. Keep it that way; splitting by swipe leaks a user's motor idiosyncrasies
across the boundary and inflates validation accuracy.

Other FUTO configs are available via `--futo-config`:

- `swipe-2/3/4` — 28k / 38k / 50k additional donations
- `swipe-5` — 59k across **10 layouts** (azerty, dvorak, qwertz, shavian,
  toki pona, …) and **8 languages**. Directly useful for layout-agnostic
  training; extra `layout` / `language` / `dual_finger` fields land in `Swipe.meta`.

Not included here: the [Yandex Cup 2023 NeuroSwipe](https://www.kaggle.com/datasets/sharthz23/yandex-cup-2023-neuroswipe)
corpus (millions of curves, Russian layouts). Worth adding for layout-agnostic
pretraining; it needs a Kaggle credential, so it is left out of the automated fetch.

## Canonical coordinate space

```
x in [0, 1] spans the 10 key columns of the top row
y in [0, 1] spans the 3 letter rows      (top row center at 1/6)
row insets: 0.0, 0.05, 0.15 key-widths
```

This is not an invented convention — it is exactly the space FUTO's own
`swipe-5/layouts/*.json` are expressed in. Their `qwerty.json` puts `a` at
(0.1005, 0.5) with half-extents (0.05, 0.1667); `layout.key_center` reproduces
that to within 5e-4, and `tests/test_layout.py` asserts it against the verbatim
numbers.

`scripts/calibrate_layout.py` recovers the same grid independently from touch
data in both corpora, by taking the median touch-down point per first letter.

**Aspect ratio is deliberately not baked into the coordinates.** Squashing every
keyboard to a unit square distorts angles and curvature by a different factor per
device — FUTO's letter grid measures 2.38 wide-to-tall, How We Swipe's 2.08. Each
`Swipe` carries its `aspect` and `features.aspect_correct` restores physical
proportions before any geometric feature is computed.

### The two corpora do line up

`scripts/validate_alignment.py` on the built cache:

```
corpus                        letters  med |dx|  med |dy|     max  aspect
futo/validation                    25     0.101     0.162   0.563   2.384
how_we_swipe/test                  26     0.057     0.131   0.278   2.076

futo/validation vs how_we_swipe/test: 0.124 over 25 letters
```

Offsets are in key half-widths, so the two corpora's per-letter touch-down
medians sit **0.124 half-widths apart — about 6% of a key**, from entirely
independent collection pipelines.

The small consistent positive `dy` is not misalignment: users touch systematically
*below* key centers. That is behavioral signal, and the calibration script is
careful to derive How We Swipe's row pitch from a *difference* of medians so the
bias cancels rather than being fitted away.

## Source-specific handling

**FUTO** — coordinates pass through unscaled; the logged canvas *is* the letter
grid. Timestamps are absolute epoch ms, rebased to 0. Labels are lowercased and
stripped to letters (`"don't"` → `"dont"`; the keyboard has no apostrophe key, so
that really is the gesture performed).

**How We Swipe** — two format details the parser handles, both verified against
the release:

1. Rows are one touch *event*, not one swipe. Gestures are segmented on
   `touchstart` → `touchend`, and users retry failed words, so one target can
   produce several gestures. Stray `touchend`s with no open gesture are dropped.
2. Some lines carry extra trailing fields beyond the 12-column header (we observe
   up to 31). These are additional per-attempt error flags. Every field the loader
   needs sits at a fixed index 0–10, so parsing is positional and everything from
   index 11 on is treated as flags.

Its `keyb_height` also covers a partial bottom (space) row, so the letter grid is
only the top **82.56%** — measured, not assumed. Without that correction the two
corpora sit ~0.06 canonical units apart in y.

Attempts the source marks as errors are excluded by default (`--keep-flagged` to
retain; it is ~16% of How We Swipe gestures).

## Features

`features.encode` follows the FUTO Swipe paper (arXiv
[2606.25247](https://arxiv.org/abs/2606.25247)): resample to exactly 64 points,
then derive 8 channels with a Savitzky–Golay filter — `x, y, vx, vy, ax, ay,
speed, curvature`.

`mode="time"` (default) spaces points uniformly in time, preserving the
hesitation-at-corners cue that carries much of the signal. `mode="arclength"`
normalizes speed away, the classic SHARK2-style choice.

Curvature is `1/radius`, so it diverges wherever the gesture pauses — which
happens at every key corner. It is epsilon-guarded and clipped at
`CURVATURE_CLIP = 1e3`; unclipped it reaches ~1e16 on real data and overflows
float32 the moment anything squares it.

Acceleration keeps genuinely heavy tails (max ~1e4) because real touch data is
jerky. **Standardize channels over the training set before feeding a model** —
this library deliberately does not, so the statistics stay inspectable.

## Augmentation

`augment.augment(swipe, layout)` applies **one shared affine to the trajectory and
the key-center tensor together**. This is the central idea of the FUTO paper and
the reason a decoder generalizes across layouts: augmenting only the trajectory
teaches the model that a distorted gesture still maps to the same word on a fixed
keyboard, which is false.

`tests/test_augment.py` pins both directions — co-augmented gestures stay
decodable to their label, and the same gestures scored against the *un*-augmented
layout stop being decodable.

Time reversal also reverses the label (`"cat"` → `"tac"`); a backwards gesture is
the gesture for the backwards string. It is off by default.

## The encoder

```bash
python scripts/train_encoder.py --d-model 128 --dilations 1,2,4,8,1,2,4,8 --epochs 10
python scripts/eval_layout_transfer.py --checkpoint runs/full/encoder.pt
```

A dilated TCN over 64 frames, trained with CTC. Per frame it consumes

```
[ key affinity (26) | kinematics (6) ]  ->  logits over 27 symbols (a-z + blank)
```

**The model never sees absolute coordinates.** Position reaches it only as
Gaussian affinity to the keys of whatever layout is supplied at runtime,
expressed in each key's own half-extents. There is nothing layout-specific left
to memorize, so swapping the keyboard at inference works without retraining.

CTC fits because a gesture crosses its target keys *in order* — alignment is
monotonic — and blanks absorb the keys the path merely passes over. It also
collapses repeats for free, matching the physics: "hello" dwells on `l` once.

### Motion is measured in keys, not grid-heights

Aspect-corrected coordinates express velocity in grid-heights per second, which
is only comparable between layouts with the same number of rows. A 5-row
keyboard packs its rows into the same unit square, so an identical gesture reads
~1.65× slower — a distribution shift sitting exactly where cross-layout transfer
is being asked for.

Dividing by key size removes it, and it is the same normalization the affinity
channel already applies, so both feature groups land on one scale. The device
aspect ratio cancels out algebraically: `x * aspect / (2 * rx * aspect)`.
`--no-key-units` restores the old behaviour as an ablation.

### Metrics are lexicon-free

Greedy CTC decode, no trie and no language model. A lexicon would score much
higher and would also let a strong prior paper over a weak encoder. These
numbers are the encoder alone — the lexicon belongs in the decoding layer above
it.

Worth knowing when reading them: **17.8% of validation words never appear in
training**, and train/validation share **zero donor sessions**.

### Results

1.32M params, 10 epochs over 916,834 swipes, 30 min on an M-series laptop (MPS).

| eval set | what it tests | CER | word acc |
|---|---|---|---|
| futo/validation | unseen donor sessions, same corpus | 0.070 | **78.4%** |
| how_we_swipe | different corpus, devices, users | 0.126 | **64.4%** |

Cross-corpus transfer with no fine-tuning is the number that validates the
normalization work in the first half of this README — How We Swipe was collected
years earlier, on different hardware, by different people.

### Layout transfer

`eval_layout_transfer.py` on the same checkpoint, against real English gestures
performed on layouts absent from training:

| layout | grid | n | CER | word acc |
|---|---|---|---|---|
| qwerty | 3x10 | 28,267 | 0.140 | 71.1% |
| qwertz | 3x10 | 801 | 0.118 | **69.5%** |
| clearflow | 5x6 | 11,677 | 0.165 | 49.5% |
| kasroz | 5x6 | 1,058 | 0.185 | 43.0% |
| dvorak | 4x10 | 2,809 | 0.215 | 35.1% |

Two distinct findings, and they should not be blurred together:

**Letter permutation is essentially free.** qwertz keeps QWERTY's geometry and
moves the letters; the model loses 1.6 points against qwerty (69.5% vs 71.1%)
having never trained on it. It is not reading letter positions it memorized.

**Unseen grid geometry costs real accuracy but does not break the model.**
clearflow and kasroz are 5-row grids; the encoder only ever saw 3 rows, and
drops to 43-50%. Well above collapse, well below parity. dvorak is weakest at
35.1%.

So layout-agnosticism holds strongly for the permutation case and partially for
the geometry case — *at the encoder level*. The next table is why that
qualifier matters.

### Layout transfer with the lexicon on

The table above is lexicon-free greedy, which isolates the encoder but is not
what anything ships. `eval_layout_beam.py` runs the same held-out layouts
through the real first pass — trie-constrained beam + blended lexicon.
Measured on the 20-epoch encoder (whose greedy numbers differ slightly from
the 10-epoch table above):

| layout | grid | n | greedy | beam 100 top-1 | n-best hit@8 |
|---|---|---|---|---|---|
| qwerty (swipe-5) | 3x10 | 28,267 | 72.8% | 84.3% | 90.1% |
| qwertz | 3x10 | 801 | 68.0% | 86.1% | 94.1% |
| clearflow | 5x6 | 11,677 | 49.0% | **83.4%** | 93.8% |
| kasroz | 5x6 | 1,058 | 53.5% | **84.0%** | 94.7% |
| dvorak | 4x10 | 2,809 | 40.4% | 79.4% | 90.6% |

**The 20–30 point "geometry gap" was mostly a decode-mode artifact.** With the
trie on, clearflow lands 0.9 points *below* same-corpus qwerty and kasroz 0.3
below — versus 20+ at greedy. The encoder's logits on 5-row grids are noisier,
but the information survives and the search recovers it; clearflow and kasroz
even have *higher* n-best hit rates than swipe-5 qwerty. The earlier hypothesis
that closing the gap needs mixed-row-count training is therefore mostly moot.
What remains real: dvorak (−4.9), and the fact that the swipe-5 corpus as a
whole runs ~8 points below futo/validation (84.3 vs 92.1 at matched decode) —
different donation task, different donors. The right baseline for a transfer
claim is swipe-5 qwerty, not futo/validation; comparing across that line
conflates corpus shift with layout effect.

For calibration: the FUTO paper reports 96.84% on clearflow and 97.68% on
kasroz (beam 100, 162k AOSP wordlist *extended with the evaluation targets*).
Their clearflow number sits *above* their own qwerty test figure (92.94%), so
those donors swipe cleanly; relative to corpus, their transfer penalty and
ours are comparable. The absolute gap is *not* encoder quality — measured
head-to-head under matched decoding, their released encoder trails ours
(#25) — it is their scoring form and wordlist protocol.

*(This table did not survive #37–38: permutation-mixture training lifts
every unseen grid by 11–15 points and erases the dvorak deficit — see "The
encoder's implicit LM".)*

### Error analysis

```bash
python scripts/error_analysis.py --checkpoint runs/full/encoder.pt
```

Of the 4,320 errors on futo/validation:

| | share of errors | implication |
|---|---|---|
| prediction is not a real word | 84.7% | a lexicon rejects it outright |
| true word in-vocab and 1 edit away | 52.5% | lexicon + beam very likely recovers |
| predicted a *different* real word | 15.3% | needs a language model, not a lexicon |
| true word absent from the lexicon | 12.8% | a lexicon cannot help |

**The encoder is not the bottleneck for QWERTY accuracy — decoding is.** Roughly
half the remaining error is one edit from an in-vocabulary word, and 85% of
predictions aren't words at all. The 78.4% figure is what a lexicon-free decode
buys; almost no shipping keyboard decodes that way.

Two more findings worth having:

**Short words are not the problem.** Length ≤3 accounts for only 6.6% of errors,
and 2–3 letter words run 94–98% accurate. Errors concentrate at length 6–10
(62% of them). The `im` → `in` sample that suggested otherwise was unrepresentative.

**78% of substitutions are between physically adjacent keys** — `i`↔`o`, `r`↔`t`,
`s`↔`d`, all at exactly 1.0 key-widths. The gesture genuinely passes near both;
this is spatial ambiguity, not a model defect, and it is precisely what a
lexicon disambiguates.

**Doubled letters look like an encoder defect but mostly aren't.** Words
containing a repeat score 48.8% against 81.7% without — a 33-point gap. CTC
needs a blank frame between the two identical labels, but the gesture for
"hello" traces h-e-l-o exactly like "helo": there is no geometric signal
separating them, only dwell duration. This is intrinsic ambiguity in shape
writing, and the lexicon is what resolves it.

**Out-of-vocabulary words are the genuine encoder weakness**: 31.8% versus 80.4%
on in-vocabulary words. These are also the cases where a lexicon *cannot* help,
so this is where encoder work would actually pay.

The picture holds cross-corpus. On How We Swipe: 78.8% non-words, 45.1%
in-vocab-and-one-edit, doubled letters 44.2% vs 67.3%.

## The decoder

```bash
python scripts/eval_decoder.py --checkpoint runs/full/encoder.pt --limit 20000
```

Trie-constrained CTC prefix beam search (`model/beam.py`). Standard prefix beam
search with two changes: a prefix may only be extended by a character the
lexicon trie allows, so a non-word can never be emitted; and only prefixes on a
word-terminal node may finish, scored with a unigram prior (`alpha`) and a
length bonus (`beta`).

Prefix beam search rather than n-best rescoring because CTC assigns probability
to *label sequences*, and many alignments collapse to the same string — summing
over them is the point, and that is what the blank/non-blank split is doing.

### The lexicon sets the ceiling

Whatever is not in the lexicon cannot be produced, so `--lexicon` is switchable
and the ceiling is reported next to every number. The default blends the FUTO
training vocabulary with the top ~320k English words by frequency (`wordfreq`).

Measured on 2,000 validation swipes, top-1 word accuracy:

| lexicon | size | futo ceiling | futo | hws ceiling | hws |
|---|---|---|---|---|---|
| training vocab | 65k | 97.0% | 91.8% | 88.4% | 74.3% |
| wordfreq 50k | 48k | 95.5% | 90.1% | 93.1% | 77.4% |
| wordfreq 200k | 183k | 98.0% | 91.7% | 96.8% | 78.8% |
| train + wordfreq 320k | 302k | 99.1% | 92.0% | 98.0% | **79.3%** |

Two things worth taking from this. **A larger vocabulary does not degrade
precision** — I expected 300k words to admit enough confusable candidates to
cost accuracy, and it doesn't, because the frequency prior absorbs them. And
**general vocabulary matters far more cross-corpus than in-corpus**: +0.2 points
on futo/validation, whose own training vocabulary already covered 97% of it,
versus +5.0 on How We Swipe.

A trap worth recording: the Unix dictionary (`/usr/share/dict/words`) has 236k
entries but covers only 82.6% of futo/validation — it is an archaic list missing
common inflections. Size is not coverage.

### Results (n = 20,000 per split)

*(Historical: measured at the then-current beam 16/32 defaults. Current
end-to-end numbers are in the headline table; the deltas here show what the
lexicon itself was worth.)*

**futo/validation** — lexicon ceiling 98.8%

| decoder | CER | top-1 | top-4 |
|---|---|---|---|
| greedy, no lexicon | 0.070 | 78.4% | — |
| beam 16 | 0.043 | 91.1% | 95.3% |
| beam 32 | 0.039 | **91.6%** | **96.1%** |

**how_we_swipe** — lexicon ceiling 98.4%

| decoder | CER | top-1 | top-4 |
|---|---|---|---|
| greedy, no lexicon | 0.126 | 64.4% | — |
| beam 16 | 0.098 | 79.7% | 87.1% |
| beam 32 | 0.094 | **80.2%** | **88.2%** |

Lexicon-constrained decoding is worth **+13.2 points** on futo/validation and
**+15.8** on How We Swipe. Beam 32 reaches 93% of the achievable ceiling on
futo/validation and 82% on How We Swipe.

Decoding runs at roughly 1,000 swipes/s single-threaded at beam 32 — the trie
prunes hard enough that decode speed was never the bottleneck I expected.

`alpha`/`beta` were tuned on 2,000 validation swipes, but the entire grid
`alpha ∈ [0, 1.6] × beta ∈ [0, 2.0]` spans under one point of accuracy, so
neither value is load-bearing.

### What is left

Growing the vocabulary largely emptied the out-of-vocabulary bucket — it fell
from 41.3% of remaining errors to 14.8% on futo/validation, and from 42.4% to
8.4% on How We Swipe. What remains is almost entirely one thing:

| | futo/validation | how_we_swipe |
|---|---|---|
| out-of-vocabulary | 14.8% | 8.4% |
| real-word confusions | **85.2%** | **91.6%** |

`val`→`call`, `has`→`had`, `im`→`in`, `ok`→`on`. The gesture genuinely resembles
both words, and no lexicon can separate them — only context can.

*(Written before the context work; partly superseded. The LM was then measured
to saturate at ~27% of this headroom — see "How much is left for any language
model?" — and an acoustic rescorer covers part of the rest. The stacked result
is in the headline table.)*

## Encoder gains do not survive the lexicon

The 10-epoch schedule was cutting training short: training loss was still
falling monotonically when cosine annealing ended. A 20-epoch run with the
identical config confirms it — and then shows why it barely matters.

```bash
python scripts/train_encoder.py --d-model 128 --dilations 1,2,4,8,1,2,4,8 --epochs 20
```

| | 10 epochs | 20 epochs | delta | |
|---|---|---|---|---|
| futo/val, greedy (no lexicon) | 0.7840 | 0.8010 | **+1.70** | 6.0 SE |
| futo/val, **beam 64 + lexicon** | 0.9186 | 0.9204 | +0.18 | 0.9 SE — noise |
| how_we_swipe, greedy | 0.6440 | 0.6590 | **+1.50** | 4.5 SE |
| how_we_swipe, **beam 64 + lexicon** | 0.8020 | 0.8090 | +0.70 | 2.5 SE |
| futo/val top-8 ceiling | 0.9705 | 0.9700 | −0.05 | unchanged |

**A 1.7-point encoder gain becomes 0.18 points once the lexicon is applied — a
~10x attenuation — and the n-best ceiling does not move at all.** The lexicon
was already repairing exactly the errors a better-trained encoder fixes: both
turn malformed spellings into real words. What survives the lexicon is
real-word confusions, and those are not what more encoder training addresses.

Two consequences worth carrying:

- **Never evaluate encoder changes lexicon-free.** Greedy CER is the natural
  metric while training and it overstates end-to-end value by an order of
  magnitude here. Every encoder-side experiment should be judged after the
  decoder.
- **The attenuation is weaker where the encoder is genuinely the bottleneck.**
  How We Swipe keeps +0.70 of its +1.50, against futo/validation keeping +0.18
  of +1.70 — consistent with its much larger "in lexicon but never in n-best"
  bucket (7.6% vs 1.9%).

### Gains attenuate twice, not once

The 20-epoch encoder was left unpromoted earlier. Regenerating its n-best lists
and retraining the rescorer against it settles why, and the answer generalizes.

| stage | 10-epoch | 20-epoch | delta | |
|---|---|---|---|---|
| futo first pass | 0.9186 | 0.9204 | +0.18 | noise |
| futo + rescorer | 0.9240 | 0.9236 | −0.04 | noise |
| **futo full stack** | **0.9381** | 0.9369 | −0.12 | noise |
| how_we_swipe first pass | 0.8020 | 0.8088 | +0.68 | significant |
| how_we_swipe + rescorer | 0.8124 | 0.8210 | +0.86 | significant |
| **how_we_swipe full stack** | 0.8130 | **0.8218** | +0.88 | significant |

The stronger encoder leaves the rescorer less to fix: it recovers **11.3%** of
available headroom behind the weaker encoder and only **6.7%** behind the
stronger one. So a first-pass gain is absorbed first by the lexicon (~10x, shown
above) and then again by the second pass, and on futo/validation nothing
measurable survives to the end of the stack.

Where the encoder genuinely is the bottleneck the opposite happens and the gains
*compound*: How We Swipe turns +0.68 at the first pass into +0.88 end-to-end.

**Not promoted.** It is neutral on the primary metric and better only
cross-corpus, so the case rests entirely on How We Swipe. Promoting it would
also invalidate the test-split headline, which was measured with the 10-epoch
encoder, and re-measuring would mean a second look at a split deliberately
budgeted for one. Kept at `runs/full20` + `runs/rescorer20` and recommended if
cross-corpus robustness is what matters; `runs/full` stays canonical because it
is the configuration test actually verified.

## Context reranking

```bash
curl -sL -o data/lm/count_1w.txt https://www.norvig.com/ngrams/count_1w.txt
curl -sL -o data/lm/count_2w.txt https://www.norvig.com/ngrams/count_2w.txt
python scripts/eval_reranker.py --limit 20000
```

A word bigram with stupid backoff, reranking the top-8 beam candidates. Trained
on the **Google Web corpus n-gram tables, external to both eval corpora** — a
model fitted on FUTO's train-split sentences would have memorized much of its
validation set, since both transcribe Common Voice.

### Add contextual evidence only, never the raw probability

The obvious formulation — `acoustic + w·log P(word | context)` — makes accuracy
**strictly worse at every weight** (0.9186 → 0.9027 at w=1.0). The reason
generalizes beyond this project:

the beam score already contains a unigram prior (`alpha=0.8`), and a sparse
bigram backs off to unigram most of the time, so raw scoring mostly re-applies a
prior that is already there. The residual errors are exactly the cases where
word frequency *misleads* — the beam's top-1 is already the more frequent word —
so doubling down on frequency pushes the wrong way.

Measured on futo/validation, over the 1,037 recoverable cases (true word in the
n-best but not ranked first):

| | |
|---|---|
| raw `log P(w\|ctx)` prefers the **wrong** word | 56.8% — worse than chance |
| restricted to *observed* bigrams, prefers the **true** word | **74.2%** |
| true bigram present in the table at all | 31.3% |
| correct items exposed per recoverable item | **17.7×** |

The signal is real but only where a bigram was actually observed, and the
exposure ratio is punishing. So `rerank` adds `bigram_delta` — `log P(w|ctx) −
log P(w)`, exactly zero when the pair was never seen — which contributes
contextual evidence without re-applying frequency.

### Results (n = 20,000, beam 64, top-8)

| split | baseline | gated, oracle ctx | gated, **decoded ctx** | raw, decoded ctx |
|---|---|---|---|---|
| futo/validation | 0.9186 | 0.9226 | **0.9223** | 0.9025 |
| how_we_swipe | 0.8082 | 0.8145 | **0.8130** | 0.7785 |

**Error propagation is negligible**: conditioning on what was actually decoded
costs 0.0003–0.0015 against oracle context, so the streaming number is the
honest one and it is nearly free.

Gains are real but modest — **+0.4 / +0.6 points**, well short of the +3–4 I
expected. The binding constraint is LM sparsity: only 31.3% of the true bigrams
are in a 286k-entry table. A denser n-gram source or a small neural LM is the
obvious next step, and its value scales with that coverage number.

### How much is left for *any* language model?

```bash
python scripts/eval_neural_rerank.py --limit 5000 --lm gpt2
```

Rather than build a denser n-gram model and find out, rescore the *existing*
n-best lists with a pretrained neural LM — no training, one number.

| reranker | decoded-context top-1 | gain | share of headroom |
|---|---|---|---|
| none (beam 64) | 0.9228 | — | — |
| word bigram, gated | 0.9268 | +0.4 | 8% |
| distilgpt2 (82M) | 0.9346 | +1.2 | 24% |
| gpt2 (124M) | **0.9360** | **+1.3** | **27%** |
| n-best@8 ceiling | 0.9724 | +5.0 | 100% |

**The LM path is saturated.** A 51% parameter increase (82M → 124M) buys 0.14
points, so the curve is flat and a larger model will not change the picture.
Roughly **73% of the remaining headroom is not language-modelable** — it is
acoustic: `philipp`/`philip`, `wayne`/`warner`, near-identical paths where
context cannot help.

The delta formulation matters here too, and more sharply than for bigrams: at
weight 0.8, raw `log P(w|ctx)` gives 0.9248 against delta's 0.9386. Decoded
context costs 0.26 points versus oracle, partly error propagation and partly
because decoded text is lowercase, which is out of distribution for GPT-2.

### The saturation survives a 70× scale-up

```bash
python scripts/eval_neural_rerank.py --limit 5000 --lm gpt2-xl --nbest-cache nbest.pkl
python scripts/export_rerank_bundle.py --nbest-cache nbest.pkl
modal run scripts/modal_rerank.py        # one A10G per model, ~$1 total
```

#11 called the curve flat from a single step (82M → 124M). Rerun with the
ladder extended two ways: the GPT-2 family to 1.5B (pure scale, same 2019
data, run locally) and Qwen3.5 base models to 9B (modern data *and* scale,
one rented A10G each). Every model rescores the byte-identical frozen n-best
lists via `--nbest-cache`; same weight sweep, same delta formulation.

| LM | params | vintage | decoded top-1 | share of headroom |
|---|---|---|---|---|
| gpt2 | 124M | 2019 | 0.9360 | 27% |
| gpt2-medium | 355M | 2019 | 0.9362 | 27% |
| gpt2-large | 774M | 2019 | 0.9352 | 25% |
| gpt2-xl | 1.5B | 2019 | **0.9372** | 29% |
| Qwen3-0.6B-Base | 0.6B | 2025 | 0.9288 | 12% |
| Qwen3.5-0.8B-Base | 0.8B | 2026 | 0.9318 | 18% |
| Qwen3.5-2B-Base | 2B | 2026 | 0.9338 | 22% |
| Qwen3.5-4B-Base | 4B | 2026 | 0.9300 | 15% |
| Qwen3.5-9B-Base | 9B | 2026 | 0.9362 | 27% |

**No model at any scale converts more than 29% of the headroom.** From 124M
to 9B — 70× the parameters, two model generations — every result lands in a
0.84-point band, within which most differences are under 2 SE (1 SE ≈ 0.36
here; the 4B's dip below the 2B is noise, not an inversion). The best model
on the entire ladder is 2019's gpt2-xl, and Qwen3.5-9B ties gpt2-medium.

*(Both halves of this are now in doubt. The scale half is a claim about the
**role**: run the same ladder inside the search and the GPT-2 family stops
being flat. The Qwen half is a claim about the **scoring**: the delta form's
subtracted prior is estimated by a proxy that suits GPT-2 and not Qwen, and
correcting it moves the modern rungs 1–2.7 points — enough to reverse the
ordering in-search. The re-run with the fix is below.)*

**The re-run with the corrected prior (#72): every rung rises, the family
gap was the bug, and "no model converts more than 29%" falls with it.**
Same frozen lists, same harness, `--uncond marginal` — the delta form's
prior averaged over eight neutral contexts instead of read off the start
token:

| LM | decoded top-1 (bos → marginal) | share of headroom |
|---|---|---|
| gpt2-xl | 0.9372 → 0.9412 | 37% |
| Qwen3.5-0.8B-Base | 0.9318 → 0.9386 | 32% |
| Qwen3.5-2B-Base | 0.9338 → 0.9410 | 37% |
| Qwen3.5-4B-Base | 0.9300 → 0.9382 | 31% |
| Qwen3.5-9B-Base | 0.9362 → **0.9424** | **40%** |

The Qwen rungs gain +0.62 to +0.82 and every weight optimum snaps to 0.8
delta — the bos runs had scattered to 0.3–0.5, family-specific, the same
fabricated surface #66 caught in-search. But the correction is not
Qwen-specific in this role: gpt2-xl gains +0.40 where the fused search
measured it a no-op (−0.01). As a rescorer the delta term is the only LM
signal and its weight is high, so even GPT-2's 0.92-correlation proxy prior
leaks measurable error; in the fused search the α·unigram term stands
beside it and absorbs the difference. What survives of #34 is exactly its
scale claim: the corrected ladder spans 31–40% of headroom with 9B ahead of
gpt2-xl by 6 words in 5,000 (under 1 SE), so scale stays flat *as a second
pass* even with the scoring fixed, and in-search authority (#66: +0.32 at
p=0.037) remains the only measured regime where it buys real points. What
falls: the ≤29% conversion bound, and with it #11's "73% acoustic" softens
to ~60%.

**Distribution match beats capability** — *superseded by the re-run above:
the "modern-pretraining mismatch" was the prior proxy, not the pretraining.
Kept as first written:* the Qwen column is the sharper
finding: modern curated pretraining is mildly *mismatched* to lowercase,
casing-free conversational fragments, and it takes ~9B parameters to climb
back to what WebText gives a 124M model for free. Qwen3.5 beats Qwen3
size-for-size — the family improves toward the same asymptote, not past it.

**The literature agrees, where it has looked.** ASR rescoring on competitive
first passes converts ~3–11% of oracle headroom, flat in LM scale: BERT-base
≈ BERT-large and GPT-2 buys zero (arXiv:2204.00212); a 70M model within
0.2–0.7 WER of Llama2-7B (2406.18972); T5-3B → PaLM-540B worth ~0.5 WER
(2306.08133). The one ~50%-conversion result (1910.14659) sits on a weak
7.3%-WER first pass — saturation is a strong-first-pass phenomenon. On
keyboards: Gboard's production n-gram → neural-LM swap moved words-modified
by 0.26–1.19% (EMNLP-Industry 2024); 6.4% of a 40k lexicon has a
gesture-*identical* twin on QWERTY, independent of any LM (Smith, Bi & Zhai,
CHI 2015); and the FUTO paper ships its context LM unevaluated. No
gesture-keyboard LM-size ablation or headroom-conversion figure appears to
have been published. The one documented route past a selection ceiling is
right context and cross-hypothesis correction (Gboard post-correction,
−2.31 gesture WER; HyPoradise, NeurIPS 2023) — precisely the lever #30–33
measure here.

### The same ladder, climbed inside the search

```bash
python scripts/run_fused_local.py --bundle fused_bundle_val38.pkl \
    --lm Qwen/Qwen3.5-9B-Base --m 8 --mu 0.8 --delta --uncond marginal \
    --lags 0,1,joint --save-hyps runs/hyps_uncond38_qw9.npz
python scripts/compare_hyps.py runs/hyps_uncond38_xl.npz \
    runs/hyps_uncond38_qw9.npz
python scripts/probe_lm_fit.py --lms gpt2-xl,Qwen/Qwen3.5-9B-Base
```

The table above prices LM scale as a *second pass*. #51 measured the same
gpt2 → gpt2-xl swap *inside* the fused search and got +0.43 against that
role's ~+0.2, which reopened the ladder — from two endpoints only, and with
nothing above 1.5B, where the GPT-2 family ends.

Every rung decodes the byte-identical bundle at one config — AR-MMI deep
lists, M=8, delta form, μ=0.8, α=0.4, β=1.2, beam 8 — over a fixed 3-of-8
slice of validation (795 sentences, 7,539 words; no LM 92.53, ceiling@8
97.59, so 5.06 points of headroom). The arms rank the same lists, so the
comparisons are paired: McNemar over the words two arms decode differently,
not the ±0.16 unpaired SE.

| LM | params | vintage | streaming | lookahead-1 | joint | headroom |
|---|---|---|---|---|---|---|
| *(no LM)* | — | — | 92.53 | — | — | — |
| gpt2 | 124M | 2019 | 93.87 | 94.56 | 94.67 | 42% |
| gpt2-medium | 355M | 2019 | 93.96 | 94.81 | 95.12 | 51% |
| gpt2-large | 774M | 2019 | 94.24 | 95.07 | 95.22 | 53% |
| gpt2-xl | 1.5B | 2019 | 94.11 | 94.99 | 95.22 | 53% |
| Qwen3.5-0.8B-Base | 0.8B | 2026 | 93.90 | 94.77 | 95.04 | 50% |
| Qwen3.5-2B-Base | 2B | 2026 | 94.19 | 95.22 | 95.38 | 56% |
| Qwen3.5-4B-Base | 4B | 2026 | 93.90 | 94.80 | 95.26 | 54% |
| Qwen3.5-9B-Base | 9B | 2026 | 94.22 | 95.30 | **95.54** | **60%** |

**In-search conversion runs about double the second pass.** 42–60% of the
same headroom the rescoring ladder converts at 25–29%, which is #51's finding
at seven rungs instead of two.

**The gain tracks authority, not ranking.** Split gpt2 → gpt2-xl by how much
the LM is allowed to decide:

| commitment | gpt2 | gpt2-xl | delta | paired p |
|---|---|---|---|---|
| streaming (lag 0) | 93.87 | 94.11 | +0.24 | 0.11 |
| lookahead-1 | 94.56 | 94.99 | +0.42 | 0.0038 |
| joint | 94.67 | 95.22 | **+0.56** | 0.00012 |

The same swap more than doubles in value as the LM gains the power to keep
hypotheses alive across the sentence, and is not resolvable at all while it
may only rescore the current word. The cross-family gap has the identical
signature — Qwen-9B over gpt2-xl is +0.11 at streaming (p=0.54), +0.32 at
lookahead-1 (p=0.035), +0.32 at joint (p=0.037). Whatever separates two
readers here, it only becomes visible where the reader can prune.

**Above gpt2-xl the ladder does keep climbing — but only after fixing how
`delta` estimates its own prior.** The GPT-2 family saturates at 774M (large
and xl tie at 95.22), while the modern family passes it: Qwen3.5-9B leads by
+0.32 (p=0.037), 2B by +0.16 (p=0.31, a tie), and even 0.8B — half gpt2-xl's
size — draws level at 95.04. The 4B checkpoint's dip below its smaller sibling,
which #34 saw as a rescorer and the first pass here read as −0.64 (p=0.0016),
shrinks to −0.12 (p=0.44) once the prior is estimated properly: that anomaly
was mostly the same artifact in a third place. That is the opposite of what the first pass of
this experiment measured, and the difference is one line of scoring.

#### The prior term is family-specific, and it is worth 2.65 points

`delta` is defined as `log P(w|ctx) − log P(w)` — contextual evidence with the
LM's own prior removed (#10, #49) — and the implementation estimated `log P(w)`
as `log P(w | start token)`. Against the corpus unigram that proxy correlates
at **0.92 for gpt2 and 0.77 for Qwen3.5-2B**, because GPT-2 has a real BOS and
Qwen has none, so it gets `<|endoftext|>` instead. The term is subtracted from
every candidate, so a worse proxy quietly degrades every score the model
produces. Estimating the prior over a set of neutral contexts instead
(`--uncond marginal`) costs one extra cached forward per candidate word and
changes each rung by:

| LM | bos proxy | marginal | delta | paired p |
|---|---|---|---|---|
| gpt2 | 94.60 | 94.67 | +0.07 | 0.63 |
| gpt2-medium | 94.83 | 95.12 | +0.29 | 0.012 |
| gpt2-large | 95.13 | 95.22 | +0.09 | 0.44 |
| gpt2-xl | 95.24 | 95.22 | −0.01 | 1.0 |
| Qwen3.5-0.8B-Base | 92.39 | 95.04 | **+2.65** | 6.8e-33 |
| Qwen3.5-2B-Base | 94.16 | 95.38 | **+1.22** | 5.5e-14 |
| Qwen3.5-4B-Base | 93.53 | 95.26 | **+1.74** | 1.1e-19 |
| Qwen3.5-9B-Base | 94.39 | 95.54 | **+1.15** | 8.3e-13 |

The GPT-2 rungs move by rounding error and every Qwen rung by 1–2.7 points.
Three separate "findings" from the first pass are that one asymmetry wearing
three hats: the modern-family deficit; a *sharp* μ surface where this project's
tuned surfaces are otherwise flat, with Qwen peaking at 0.4 and GPT-2 at 0.8
(fixed, every model wants 0.8 again); and Qwen-0.8B scoring below the no-LM
floor. A systematic error in a shared term does not add noise — it manufactures
structure, and the structure it manufactured here was plausible enough to
survive a written-up, committed conclusion.

The lesson generalizes past this stack: **any scoring form with a subtracted
prior inherits a hidden dependence on how that prior is estimated, and the
estimate can be good for one model family and bad for another.** #34's
rescoring ladder uses the same convention over the same checkpoints, so its
"distribution match beats capability" reading is likely measuring this too and
wants re-running before it is relied on.

#### Fit to the text and usefulness in the search are not the same thing

The obvious explanation for the first pass's ordering was that 2019 web text
simply matches lowercase, unpunctuated Common Voice prompts better. Measured,
that is not what separates these models. Per-word NLL over the reference
sentences, excluding the start position the convention above distorts:

| LM | nll/word | in-search rank |
|---|---|---|
| gpt2 | 6.17 | 4th |
| gpt2-xl | **5.62** | 3rd |
| Qwen3.5-2B-Base | 6.01 | 2nd |
| Qwen3.5-9B-Base | 6.50 | **1st** |

The ranking is inverted: the model that fits this text *worst* ranks *best* in
the search. Which is what the delta form is for — it removes the prior on
purpose, so absolute calibration to a deformatted corpus is beside the point
and what survives is contextual contrast. It also explains why the prior term
turned out to dominate: it is the one place where a model's absolute
distribution leaks back into a score meant to be free of it.

Two surface-form notes fall out of the same probe. Casing and punctuating the
references lifts Qwen-9B by 20% perplexity while leaving gpt2-xl flat, and
restores the modern family's monotonicity (on raw text 9B fits worse than 2B;
on cased text it fits better) — so the *inversion* is a formatting effect even
though the ranking is not. And fp16 is exonerated as a cause anywhere: against
fp32 the worst per-token disagreement is 0.011 nats.

Caveats. The slice is 3/8 of validation, not the full 20k: gpt2 and gpt2-xl
read 94.60 and 95.24 under the original scoring against #51's full-val 94.45
and 94.88, so levels run a few tenths high and the deltas here are slice
deltas. `MARGINAL_CTXS` is eight hand-written neutral prefixes — crude, and a
corpus-sampled estimator would likely do better still, which if anything means
the modern rungs are *under*-corrected. And an ops note that half-inverts
#51's: the fused search is launch-bound only while the model is small; scoring
every live state in one forward makes it 2.4x faster and genuinely
compute-bound by 9B, where one pass takes 8 hours on MPS against gpt2-xl's 30
minutes for identical work.

## Training-free decoding: the LLM as the lexicon

```bash
python scripts/eval_llm_trace.py --limit 500 --context oracle   # rung 1
python scripts/eval_llm_beam.py --limit 200 --context oracle \
    --gate-letters 10 --topk-geom 60 --beam 64                  # rung 3
```

Everything above trains an encoder on gesture data. This section asks the
inverted question: with the corpora used for **evaluation only**, how far do
analytic geometry and a pretrained LM get? No CTC encoder, no trie, no
lexicon, no rescorer — the LM's vocabulary is the candidate space and a
closed-form alignment cost is the acoustic channel.

| decoder | LM | context | top-1 | n-best | swipes/s |
|---|---|---|---|---|---|
| key trace → LM, few-shot generation | Qwen3.5-2B | none | 10.0% | — | 5.7 |
| key trace → LM, few-shot generation | Qwen3.5-2B | oracle | 21.2% | — | 5.7 |
| joint geometric-LM beam | Qwen3.5-2B | none (primed) | 54.7% | 66.0% | 0.18 |
| joint geometric-LM beam | gpt2-xl | none (primed) | 60.7% | 69.3% | ~1.1 |
| joint geometric-LM beam | Qwen3.5-0.8B | oracle | 80.7% | 86.0% | 0.21 |
| joint geometric-LM beam | Qwen3.5-2B | oracle | 80.7% | 86.7% | 0.18 |
| joint geometric-LM beam | Qwen3.5-9B | oracle | 85.3% | 90.0% | 0.10 |
| joint geometric-LM beam | **gpt2-xl** | oracle | **88.7%** | **92.7%** | **1.09** |
| joint beam, cross-corpus (HWS test) | Qwen3.5-2B | none (primed) | 36.7% | 50.7% | — |
| geom + trie + unigram, dwell-weighted | — | none | 78.0% | 91.3% | ~4.0 |
| geom + trie + LLM rescore, dwell-weighted | gpt2-xl | oracle | **89.3%** | 94.7% | ~2 |
| geom + trie, cross-corpus (HWS), dwell-weighted | — | none | 61.3% | 81.3% | ~4.0 |
| geom + trie + unigram, **full 20k val** | — | none | **79.2%** | 91.7% | ~4.0 |

All futo rows on the same 150 validation swipes, disjoint from the 50 used to
set the search widths (n=500 for the generation rows); fp16 on MPS.
Trained-stack anchors on this split: greedy 78.4, trie beam 91.9, full fused
stack ~95, and 80.4 cross-corpus on HWS.

**2019's gpt2-xl beats the entire modern family — and this time it is not an
artifact.** #66's first climb produced the same ordering and died under
scrutiny because the delta form's subtracted prior was mis-estimated for
Qwen. This decoder has no delta form: candidates are ranked by raw
`logP(word | ctx)`, so absolute fit to the eval text is exactly what matters —
and `probe_lm_fit.py` already measured gpt2-xl as the best per-word fit on
these lowercase Common Voice prompts (5.62 NLL/word vs Qwen-9B's 6.50).
The in-search ordering here tracks that NLL table (xl 88.7 > 9B 85.3 > 2B =
0.8B 80.7; xl > 2B at McNemar p=0.008, xl vs 9B +5 discordant-9 n.s. at
p=0.27), where #66's delta-form fused search inverts it. Same checkpoints,
same corpus: whether scale or distribution match wins depends on whether the
scoring form cancels the prior. As a bonus the winner is also 6-10x faster:
full attention runs at 1.1 swipes/s on MPS where Qwen3.5's linear-attention
torch fallback costs ~11ms *per hypothesis row* regardless of sequence
length, which also caps what context caching can save (13%).

**The trace string is not enough.** `trace.key_trace` reduces a swipe to the
keys under the finger (`crack` → `cftresaszxcvghjk`); the collapsed label is a
strict subsequence of it for 78% of validation swipes, and the misses are
corner cuts (`that` traced with no `h`). Few-shot prompting a base model to
invert that string — examples synthesized from straight-line templates, so no
gesture data enters the prompt — lands at 10–21%: the model emits frequent
words agreeing with the first letter and the context and ignores the
interior. 9B ties 2B; reading character-level geometry out of a prompt is not
a base-LM competence, which is why the geometry has to move inside the search.

**The geometric channel alone is nearly sufficient.** `geomllm.GestureDP`
scores a letter string by explaining *every* resampled gesture point: letters
land on points (Gaussian around key centers, in key half-extent units),
points between landings pay transit around the connecting segment, partial
hypotheses are charged a lower bound for path not yet consumed. One O(N) row
per hypothesis, one recurrence per appended letter, so it rides inside a beam.
Ranked against 2k common words with untuned physics constants it puts the true
word first on 88% of real swipes and in the top-8 on 100% (n=100) — most of
what the trained encoder provides, from the layout alone. What it cannot do is
*enumerate*: candidates have to come from somewhere, and that is the LM's job.

**The joint beam** (`eval_llm_beam.py`): hypotheses are letter strings scored
`lm_weight * logP_LM + logP_geom`; proposals per step are the LM's top tokens
plus per-letter quotas for the geometrically plausible next letters, each
gated letter's bare single-char token always included (they are LM-rare, so no
LM ranking surfaces them, yet they are the only guaranteed path to words the
tokenizer has buried). A parallel LM-only pass recalls contextually likely
words the joint pass loses to path-hugging noise; the pooled words are ranked
by canonical LM score plus geometry.

Four things broke on the way, each worth keeping:

1. **Qwen3.5 has no usable cold start** — conditioned on `<|endoftext|>`
   alone, its next-word distribution is flat multilingual noise (every
   letter-start token ≈ −15 nats, `Ġthe` included); gpt2's BOS distribution
   is sharply English. Same asymmetry that inverted #66's first climb. A
   one-sentence neutral English prime restores a usable prior and is worth
   ~15 points on the no-context row.
2. **Left padding corrupts Qwen3.5 logits** — the linear-attention torch
   fallback ignores the attention mask, so pad tokens leak into the state.
   Everything here right-pads and gathers at true positions, which is safe
   for any causal model.
3. **A word reached letter-by-letter carries the wrong LM score.** Its
   char-token path logprob sinks roughly twice as fast as its canonical
   tokenization (` sled` = −6.6 canonical, ≈ −35 as `s·l·e·d`), starving
   long rare words a step before they finish. Hypotheses are therefore
   re-tokenized canonically at every step — the forward pass then yields the
   exact prefix logprob for free and no separate rescoring pass exists.
4. **Partial-cost optimism breeds degenerates.** Score a prefix by
   `min(row)` and strings like `cffffffff` camp on 10% of the gesture at
   near-zero cost while adding letters free (`crack` lost its beam slot to
   them by 0.5). The unexplained-tail bound (each remaining point at least
   hovers near its nearest key) removes the entire failure class.

**Recall is the bottleneck, and it is exactly the lexicon's job.** At every
setting tried, when the true word survived to completion it won the pool —
the last climb (68 → 84 on the dev slice) came entirely from proposal width
(gate letters 4 → 10, beam 32 → 64), and the residual misses are proper
nouns and rare words whose char paths branch away mid-word
(`androscoggin`, `hanseatic`, `sled` → `dle`). A trie collapses that
branching for free; without one the search pays seconds per swipe against
milliseconds for the trie beam. Engineering recovered some of it — context
KV cache with batch expansion (works on gpt2 and, via manual state cloning,
on Qwen3.5's hybrid cache), one batched softmax/gather per step instead of
per-row reductions (which cost 20x the forward itself in launch overhead),
row dedup across the lockstepped passes, vectorized DP extension — for 1.6x
on Qwen and the architecture note above doing the rest. The honest headline
stands: a pretrained LM plus closed-form geometry recovers beam-level
accuracy with zero gesture training, and the entire remaining price —
compute and the proper-noun tail — is the candidate enumeration the lexicon
was buying.

**The two substitutions, decomposed** (`eval_geom_trie.py`). The design
above replaced both trained stages at once — encoder with geometry, lexicon
with the LM's vocabulary — so its deficit could sit in either. Putting the
trie back while keeping the training-free first stage separates them:
geometry + wf320k trie + unigram prior + gpt2-xl rescore reaches ~89-91
(oracle, same 150 swipes; 89.3/94.7 at the final frozen config, and every
knob variant lands within 2-4 words of it — the surface is flat), a
statistical tie with the LLM-as-lexicon beam at 20x the speed. The trained
beam scores **93.3** on the same 150 (its full-val 91.9 understates it
here), so the true residual is ~4-6 net words: on inspection, one dictionary
hole (`bodychecked` is in FUTO's train vocab but not wordfreq), a couple of
coin-flips (`hes`/`hers`), and 3-4 sloppy gestures where the trained
encoder's learned noise model genuinely outranks the analytic alignment —
the same acoustic residual the HWS transfer prices at scale.

**Dwell weighting buys back a third of that residual.** The alignment was
timing-blind — arclength resampling discards the fact that fingers slow
down on real letters. Weighting transit costs by local dwell time
(``GeomConfig.time_weight``, tuned on a separate 200-swipe slice, exponent
flat over 1.0-1.5) is worth **+5.3 on the LM-free row (72.7 → 78.0) and
+5.3 cross-corpus (56.0 → 61.3, tuned on futo only)** — timing is real
acoustic signal, free for the taking, and it transfers. Two nulls from the
same round: the global touch-offset calibration measures only ~7% of a key
(the canonical space already absorbed the bias — `calibrate_layout.py` is
why), and the delta-form rescore ties raw logP once geometry and the
unigram are both in the score (three channels make the prior subtraction
redundant; #10's gating result is about raw logP *alone*). On the full 20k
validation set the LM-free training-free stack reads **79.2/91.7** against
the trained beam's 91.86/97.05 — level with the trained encoder's greedy
78.4 without having seen a single gesture, with the beam's remaining lead
sitting in candidate ranking on the sloppiest swipes and 1.5% wordfreq OOV. Verdict on the two substitutions:
swapping the *encoder* for analytic geometry is nearly free; swapping the
*lexicon* for the LM's vocabulary bought nothing measurable — only ~0.7% of
validation refs fall outside wf320k and the lexicon-free beam decoded zero
of them — while costing the enumeration recall that produced the entire
deficit. Making the LM the enumerator does not remove LM bias; it relocates
it from scoring (where bias misranks a candidate you have) to proposal
(where bias makes the word unreachable). Two footnotes worth keeping: the
unigram prior is worth 68 → 98 on the dev slice (geometry alone against
320k confusables is not enough, the prior is doing SHARK²'s job), and a
*cold* raw-LM rescore is no better than the unigram (70.7 vs 72.7) — #10's
"raw logP hurts, only the gated delta form works" reappearing in a
training-free costume.

**Transfer strips the disguise off that claim.** On How We Swipe (different
apparatus, users, year; no sentence context exists there) the joint beam
falls to 36.7/50.7 while the trained stack holds 80.4. The decomposition is
clean: the analytic channel alone still ranks the true word first on 70% of
HWS swipes (top-8 96%) against 2k distractors — apparatus noise costs it ~18
points from futo's 88 — but with no context the LM prior cannot break ties
among path-compatible common words (`plane` → `one`, `target` → `great`),
and enumeration-without-context collapses. The trie-equipped version proves
the point by construction: geometry + trie + unigram holds 56.0/77.3 on the
same corpus with no context at all — +19 over the LM-as-lexicon — and the
remaining gap to the trained stack's 80.4 prices the one thing the trained
encoder still owns: robustness to apparatus noise (the analytic channel's
88 → 70 standalone drop), which is acoustic value, not implicit-LM value.

## The geometry channel: fusing the classical scorer back in

```bash
python scripts/gen_geom_proposals.py --bundle fused_base_hws.pkl \
    --data data/canonical/how_we_swipe/test --out geom_props_hws.pkl
python scripts/eval_geom_fusion.py --gamma 0.5 --proposals geom_props_hws.pkl \
    --baseline runs/hyps_base_hws.npz --buckets
```

The cell that became #73 set out to test something else: a *proposal rung*
above #66's authority ladder — when the list looks wrong, let another
enumerator (the geometric trie beam, or gpt2-xl's vocabulary via #68's
search) inject candidates the AR beam pruned, vetoed by a scorer. The
proposal half works: the trie re-search surfaces the true word on 27% of
hws coverage misses. The veto half exposed a trap worth keeping: scoring
proposals with the AR decoder's own teacher-forced logP is **null end to
end (+0.03, p=0.27)** — 340 truths entered the candidate lists and 8 won —
because proposals are precisely the words the AR model prunes, so the
pruner's score buries them again (#41's 8-nat losses met from the
constructive side; #70's bias relocation running in reverse: LM-as-enumerator
moves bias into proposal, AR-as-veto moves it into the veto).

The measurement underneath redirected the design. Within a swipe's
candidate list, the AR score and the GestureDP alignment cost correlate at
**−0.17** (calibration R² 0.04): the trained and analytic channels rank by
nearly independent criteria, which by #10/#11's rule means the analytic
channel holds evidence the trained one lacks — so it belongs *in the score*,
not behind it as a gate. With one term added to the fused acoustic for every
candidate, `ar + β·len + α·uni − γ·geom` (γ=0.5 dev-picked, decode otherwise
parity-identical to the frozen harness), the #61 hws baseline moves **80.90
→ 83.04 eval (+2.14; 595 fixed / 250 broken, McNemar p=3e-33)** — the
largest hws top-1 gain any decode-time change has bought, training-free. The
structure is the tail-thread's dream shape: unseen **+9.4**, rare (count
1–5) **+7.1**, head **+0.7** — the first tail gain with no head tax
(contrast #60–62's every-gain-taxed ledger). Proposals then add a real
sliver on top (+0.16; 35/9, p=1e-4); the gpt2-xl proposer surfaces *less*
than the trie (20% vs 27% of gated misses) — #36's diagnosis again, the hws
tail is in-lexicon, an open vocabulary adds nothing here.

Caveats, honestly held: cross-corpus is the geometry channel's best case —
the trained channel is off-domain there while geometry is corpus-agnostic —
and the futo-val cell confirms it: γ=0.5 carried over unchanged is a wash
in-domain (94.44 → 94.35 eval, p=0.54), the tail gain persisting (unseen
+2.6, rare +2.8) but now paying a head tax (−0.5) the off-domain setting
never charged; the acoustic-only γ optimum moves to ~0.1 in-domain, so the
channel weight tracks how much the trained model can be trusted, exactly as
an evidence-weighting story predicts (`runs/geom_fusion_val.log`). Fused at
γ=0.1 the in-domain channel is a small net positive with the head tax gone —
dev +0.21 (40/20, p=0.013), eval +0.09 (n.s.), unseen +0.7 / rare +1.0 /
head −0.03 (`runs/geom_fusion_val_g01.log`). α/β/μ were
never re-tuned jointly with γ; the damage ledger is short near-ties plus
wf320k's own typo entries (`plese`, `destory`) — lexicon hygiene would claw
some back; and the run used `time_weight=0`, so the dwell lever measured
above (+5.3 standalone) has not yet been composed with the fusion. Test
stays untouched.

## Second-pass acoustic rescoring

```bash
python scripts/dump_nbest.py --split futo/train --limit 150000
python scripts/dump_nbest.py --split futo/validation --limit 20000
python scripts/train_rescorer.py --epochs 8
python scripts/eval_full_stack.py --rescorer runs/rescorer/rescorer.pt
```

Since ~73% of the headroom is acoustic, a 436k-param rescorer scores whole
(gesture, candidate) pairs: each candidate letter carries its key position and
cross-attends to the full trajectory, modelling the frame dependencies CTC gives
up. It learns a **residual** — the logit is `scale · first_pass + model(...)`
with `scale` initialised to 1 — so it starts from the existing ranking and only
has to fix what the first pass gets wrong.

### The two second-pass signals stack

futo/validation, n = 20,000, beam 64 top-8:

| | lm=0.0 | lm=0.8 |
|---|---|---|
| rescorer=0.0 | 0.9186 | 0.9343 |
| rescorer=1.0 | 0.9240 | **0.9381** |

Alone: rescorer +0.54, LM +1.57. Together **+1.95 — 37% of the available
headroom**, close to additive, confirming they fix different errors. With
*decoded* rather than oracle context the streaming figure is ≈0.9355.

### Two negative results worth keeping

**The rescorer is not capacity-limited.** Going 96→160 d_model and 8→18 epochs
gives exactly the same validation accuracy (0.9245) while training loss falls
0.164→0.125 — it overfits rather than underfits.

**And the obvious fix for that made things worse.** Training n-best lists come
from the encoder's own training data, where first-pass top-1 is 0.9398 against
0.9186 on validation, so the rescorer learns to correct a first pass stronger
than the one it meets at test time. Retraining on swipe-2/3/4 — 111k swipes the
encoder has genuinely never seen — scored *worse*: 0.9211 (4.6% of headroom)
versus 0.9245 (11.3%). Those batches are much harder (first-pass 0.7945), and
that distribution shift outweighs the overconfidence it was meant to remove. A
proper fix needs held-out folds of the *same* distribution, not a different
corpus slice.

### A zero-probability bug the rescorer exposed

The first training run was all-NaN and reported "0.0% of headroom recovered" — a
plausible-looking negative result that was pure artifact. The cause was a real
bug in `beam_search`: extending a beam by a *repeated* letter draws on
`p_blank`, which stays `-inf` for a prefix that has never ended in a blank, so
both probability fields can be `-inf`. Those candidates are impossible rather
than unlikely, and were being emitted anyway.

It is invisible during decoding — such candidates lose regardless — and appeared
in only 33 of 1.2M slots, always on low-ranked repeats like `"ayyy"`, never on
the true word. It only surfaced once a downstream model consumed the scores as
features, where one `-inf` silently NaNs an entire run. Now filtered at source,
with a regression test and a loud assert in the loader.

### A trap: judge corpora by swipes, not by unique sentences

Sampling unique sentence strings suggested How We Swipe was 93% random word
lists ("prior simpson clearing tencent") with no context to exploit. That is
true of unique strings and badly wrong in effect — the 6% natural sentences
("im on a plane", "is that ok") repeat far more often and cover **half its
swipes**, so bigram coverage is 40.3%, and it gains *more* from reranking than
FUTO does.

## The cross-corpus gap

How We Swipe scores ~12 points below futo/validation. Two rounds of
investigation went into this — the second one overturned parts of the first, so
both the evidence and the corrections are recorded.

```bash
python scripts/diagnose_transfer.py --checkpoint runs/full/encoder.pt
python scripts/finetune_transfer.py --epochs 3                      # fine-tune
python scripts/finetune_transfer.py --from-scratch --train-on hws \
    --futo-limit 0 --epochs 25 --lr 3e-3                            # matched-size
```

### What it is not

| hypothesis | measurement | verdict |
|---|---|---|
| participant population | English level 0.024 spread, finger 0.028, hand 0.026 | no |
| keyboard geometry | affine sweep on eval coords peaks at identity (0.6345 vs best 0.6347) | no |
| long tail of bad users | 2 of 275 users below 0.50; median 0.815 | mild |
| sampling rate | 61 Hz vs 57 Hz; 63 vs 62 points per gesture | no |
| model input distributions | key affinity ratio 1.01, kinematics 0.85–1.01 | no |
| sloppier gestures | 62.5% of gestures cover every key of their label, vs FUTO's 60.0% | no — *better* |
| dwell-driven sample starvation | time-uniform resampling yields 42.5 vs 42.4 effective points | no |
| inter-user heterogeneity | cross-user shape distance ratio 1.09 (arc-length, timing removed) | mild |

Two corrections from the first round, worth recording because both mistakes are
easy to repeat:

- The original "timing" refutation (globally rescaling `t`) was a **null test**:
  time-uniform resampling is invariant to affine time rescaling, so it only
  perturbed velocity magnitudes. The real timing channel — dwells eating the 64
  time-uniform samples and starving transit segments — was untested. Measured
  properly, it is also innocent: both corpora allocate ~42.5 effective points to
  the path.
- "The model cannot fit this data" (fine-tune loss plateauing at 0.365) was an
  artifact of the fine-tuning schedule. Trained from scratch, the model fits How
  We Swipe to a loss of 0.15 without difficulty. The fit was never the problem —
  generalization to *new users of that corpus* is.

### What it is: the matched-size experiment

Train the same architecture from scratch on ~60k swipes of each corpus and
evaluate in-domain on held-out users (greedy):

| trained on | users in train | in-domain accuracy | other corpus |
|---|---|---|---|
| FUTO 60k | 655 | **67.3%** (futo/val) | 51.9% |
| How We Swipe 60k | 926 | **61.8%** (held-out users) | 54.9% |

In-domain versus in-domain, at matched sample count, with *more* users on its
side, How We Swipe is **5.5 points harder** — *as first read on a 5k eval
subset. #81 re-reads the same checkpoints on the full held-out sets (3.6)
and with #80's label filter (**1.8**); most of this number was measurement
and label quality, not gesture difficulty.* That is intrinsic corpus
difficulty, established by construction rather than inference — consistent with
its collection protocol (a timed typing test, versus FUTO's relaxed donation
task). Notably, our full FUTO model already beats the 60k in-domain model on How
We Swipe zero-shot (64.4 vs 61.8).

### What is recoverable

| lever | held-out HWS users, beam top-1 | n-best hit |
|---|---|---|
| zero-shot, beam 32 / prune −9 | 80.0% | 90.7% |
| + wider search (beam 64 / prune −13) | 80.5% | 92.7% |
| + fine-tune on 59.7k user-disjoint HWS swipes | **82.6%** | **93.2%** |

Two practical findings:

- **A third of the "encoder never surfaces the word" bucket was beam pruning.**
  Widening to beam 64 / prune −13 lifts the n-best hit from 90.4% to 92.6% (beam
  128: 93.7%) at ~7 ms/word single-threaded — still realtime. This raises the
  reranker ceiling on How We Swipe accordingly.
- Fine-tuning with FUTO mixed in adds +2.1 beam points on held-out users while
  costing only 0.5 on futo/validation.

**Bottom line:** of the ~12-point gap, roughly 2.6 points are recoverable with
in-domain data plus wider search, ~5.5 points are intrinsic by direct
measurement, and the remainder is the two corpora scaling differently. Read How
We Swipe as a pessimistic stress test; the reranking headroom (top-1 82.6 vs
n-best 93.2) remains the largest lever there, as everywhere.

## The encoder's implicit LM

```bash
python scripts/nbest_freq_buckets.py runs/mmi/nbest/futo_validation.npz
python scripts/train_encoder.py --d-model 128 --dilations 1,2,4,8,1,2,4,8 \
    --epochs 13 --permute-prob 0.25 --out runs/perm25e13
python scripts/nbest_freq_buckets.py data/nbest/how_we_swipe_test.npz \
    runs/perm25e13/nbest/how_we_swipe_test.npz
```

CTC emissions favour letter sequences that were common in training — the
encoder carries an implicit LM. Bucketing every n-best miss by the target
word's training count makes it visible: truth-in-beam climbs monotonically
from **64% on words the encoder never trained on to 99.8% on >1k-count
words** (futo/validation), and unseen words are 13.9× overrepresented among
ceiling misses (4.7× on How We Swipe, from a 57% in-beam floor).

Whether that is *actionable* splits sharply by corpus. A missed word outside
the lexicon can never be emitted by the trie, so no encoder change reaches
it — and futo's unseen tail is mostly out-of-lexicon proper nouns
(`androscoggin`, `pellam`): its actionable slice (in-lexicon, ≤5 training
examples) is 0.5% of swipes. How We Swipe's tail is the opposite — everyday
conversational vocabulary FUTO's transcription prompts underrepresent
(`tomorrow`, `sitter`, `ours`), in-lexicon and 3.9% of swipes. The implicit
LM is not just a frequency prior; it is a *corpus-specific* one, and it
fights every word a lexicon update would add.

### Relocating it with permuted gestures

The encoder only sees per-letter key affinity, so relabelling a real gesture
under a random letter permutation of the layout — a column permutation of
the affinity block plus the inverse permutation of the target — yields a
perfectly realistic gesture for a different letter sequence: real
kinematics, near-uniform label statistics, zero synthesis. This is the
cheapest possible synthetic data, and it is the permutation-invariance the
qwertz result (#3) said the architecture already supports.
`--permute-prob 0.25` relabels a quarter of training samples; 13 epochs
matches the canonical run's real-sample count.

Measured through the beam against the canonical encoder (beam 64, 320k
lexicon, 20k swipes; greedy drops 1–2 points by design — the same
weak-greedy/strong-search trade as MMI):

| | futo/validation | how_we_swipe |
|---|---|---|
| beam top-1 | 91.86 → 91.75 (0.6 SE — wash) | 80.82 → **81.07** |
| truth-in-beam@8 | 97.05 → 96.97 (noise) | 90.75 → **91.01** |
| unseen-word in-beam | 61.9 → 62.9 (n=814, noise) | 56.2 → **60.3** (+4.1, ~4 SE) |
| in-lex ≤5-count in-beam | 92.7 → 92.7 | 76.4 → **79.0** (+2.6, ~4 SE) |
| head (>1k) in-beam | 99.87 → 99.81 | 98.95 → 98.95 |

The head does not move, the tail moves only where it is in-lexicon, and the
aggregate follows the size of the actionable slice — a wash in-domain,
positive cross-corpus. (At 10 epochs futo showed −0.32; matched real compute
resolves that as training-budget tax, not a cost of the change.)

Two structural signatures confirm the prior *relocated* rather than
vanished. The α/β surface, flat to under a point on the canonical encoder
(#15, #23) because the emissions already carried the unigram, now spans
**4.9 points**, and removing α outright costs 3.5 — the prior sits in the
tunable knob instead of the frozen weights. Its β optimum moves to 2.4,
worth +0.24 on futo, which puts the permutation encoder at parity with
canonical after re-tuning.

### Layout transfer was the buried payoff

Random-permutation training is layout-agnosticism as a training objective,
so #22's held-out-layout eval is its natural exam (beam 100, β=1.2 for
comparability; baseline is the stronger 20-epoch encoder):

| layout | grid | 20-epoch | perm25e13 | hit@8 |
|---|---|---|---|---|
| qwerty (swipe-5) | 3×10 | 84.3% | 83.9% | 90.0% |
| qwertz | 3×10 | 86.1% | 87.5% | 95.4% |
| clearflow | 5×6 | 83.4% | **94.3%** | 97.3% |
| kasroz | 5×6 | 84.0% | **95.7%** | 97.7% |
| dvorak | 4×10 | 79.4% | **94.6%** | 97.4% |

**Every unseen grid gains 11–15 points**, from an encoder that saw 25% fewer
real swipes than its baseline. The dvorak deficit — the one thing #22 left
standing — is erased; all three unseen-geometry layouts now score *above*
the same-corpus qwerty baseline, and their n-best hit rates sit at ~97.5%.
Greedy tells the mechanism: it *rises* 15–25 points on unseen grids
(clearflow 49.0 → 75.6, dvorak 40.4 → 65.1) while falling 3 on qwerty — the
implicit LM was a qwerty-conditional sequence prior, actively fighting
letter-position pairings it had never seen. For calibration, the FUTO
paper's 96.84/97.68 on clearflow/kasroz used a wordlist extended with the
evaluation targets; these numbers use the honest 320k lexicon.

Two follow-ups sharpened the picture (#39–40). The overconfidence problem
(#16) is *not* the implicit LM: the permutation encoder's posteriors are
identically one-hot (mean max-posterior 0.9902 vs 0.9899, 81% of frames
above 0.999 in both), posterior averaging is still dead, and the 4-view
beam union still sits under the noise bar — CTC peakiness, not
prior-driven certainty. And MMI (#26) *composes* with the relocation:
one epoch over the permutation encoder's own beam-128 lists buys the same
+0.36 first pass it bought canonical (91.75 → 92.11, ceiling 97.28), with
the gain concentrated in the rare-word tail, α still load-bearing after
(the prior is not re-absorbed), and the best How We Swipe numbers of any
encoder yet (81.09 top-1, 91.31 in-beam, unseen-word in-beam +5.5 over
canonical). Layout wins survive the fine-tune on 3- and 5-row grids
(clearflow 95.4, kasroz 96.6 at β=1.2; the encoder's own β=2.4 adds
another ~0.7 pre-MMI) — dvorak alone gives back 2.0 of its 15.2 (~4 SE),
still +13 over canonical.

**Not promoted** — the composed encoder sits 0.20 (~1 SE) under the frozen
first pass on futo/validation, and re-freezing would spend a test read on
a change that is at best neutral there (#18's logic, again). But the
recommendation now has two tiers: `runs/full` + MMI remains canonical for
the frozen qwerty benchmark, and `--permute-prob 0.25` plus one MMI epoch
(`runs/perm25mmi`) is the right training recipe for anything that ships to
more than one layout or accepts user-added vocabulary — which is to say,
for a keyboard.

## Pricing absolute position: the invariance ablation

```bash
python scripts/train_encoder.py --d-model 128 --dilations 1,2,4,8,1,2,4,8 \
    --epochs 10 --shape-only --out runs/shape10
python scripts/probe_shape_collisions.py
python scripts/train_ar_decoder.py --shape-only --d-model 128 \
    --dilations 1,2,4,8,1,2,4,8 --epochs 10 --out runs/ar_shape10
python scripts/eval_ar_decoder.py --checkpoint runs/ar_shape10/ar_decoder.pt
```

How much of decoding rests on knowing *where on the keyboard* a gesture
happened? `--shape-only` answers by making the input translation- and
scale-invariant: per-gesture normalization (bbox center out, isotropic
long-side scale out), all 8 feature channels recomputed on the normalized
trajectory, no affinity block. SHARK's shape channel, in this pipeline.

| futo/validation, 20k | canonical CTC | shape CTC | shape AR |
|---|---|---|---|
| greedy, no lexicon | 78.5 | 62.5 | **78.4** |
| truth among beam survivors | ~97.9 | 93.8 | **97.8** |
| truth-in-beam@8 | 97.05 | 92.8 | 96.1 |
| beam top-1 | **91.86** | 81.8 | 84.4 |

Three separable findings:

**The ambiguity is structural (#43).** Qwerty is dense with congruences —
`is`≅`od`, `was`≅`esd`, `he`≅`gw` are literal translates on the grid, and a
tap matches *every* word once scale is gone (a dot is any template scaled to
zero). Ideal-template analysis (`probe_shape_collisions.py`): at quarter-key
alignment tolerance 69% of validation swipes have a collider and a perfect
shape matcher with a unigram tie-break caps at **90.7%** — below the anchored
encoder's measured first pass. Anchored templates lose 2.3% at the same
tolerance, nearly all of it the known doubled-letter class. Neither partial
invariance is affordable either (translation-only loses 16%, scale-only 11%
at half-key), and cross-corpus the shape features transfer *worse* (hws beam
80.8 → 58.8): gesture-relative velocity re-inherits every apparatus
difference key units were built to cancel. Context is the one real lever —
gpt2-xl deciding among the collision sets with oracle two-sided context
rescues two-thirds of the lost mass (`probe_lm_rescue.py`) — but it lifts the
anchored stack too, needs the candidates surfaced first, and evaporates at
position zero, where a keyboard spends much of its life.

**CTC is the wrong loss for invariant input (#44).** Under invariance the
letter posterior is *coupled*-multimodal — "i goes with s, o goes with d" —
and CTC's conditionally independent frames can only emit marginals.
`probe_ctc_coupling.py` shows the signature on real `is` swipes: the
incoherent cross-term `os` **ties with the truth** (43% wins) while the
coherent twin `od` loses by 7.9 nats of implicit-LM prior; on `on` swipes the
more frequent near-twin `in` outscores the truth 65% of the time, because
training pairs identical inputs with contradictory labels and the
loss-minimizing emissions are the frequency marginal. On anchored input none
of this binds — 19–36 nats of position evidence separates everything, which
is why CTC was the right choice for the canonical encoder.

**An autoregressive decoder refunds the modeling tax — and only that (#45).**
`model/ar.py` keeps the TCN trunk and adds a 2-layer causal transformer head
(1.73M total) emitting letters left to right; the trie constraint survives as
a flattened-array beam (`FlatTrie`). Same data, same 10 epochs. Greedy jumps
16 points to canonical parity, truth-among-survivors is restored to 97.8, and
the probe confirms the mechanism: `os` goes from a coin flip to −9.5 nats,
0% wins. But `is`/`of` still ties and `on`→`in` is still lost — the genuine
collisions are untouched, so beam top-1 recovers only 2.5 of the 10-point
gap. The residual is pure ranking among correctly surfaced congruent twins:
oracle@2 is 91.9, meaning the truth sits at rank 1 or 2 for 92% of swipes
with nothing but a coin to choose between them. That is what absolute
position was buying.

The affinity representation was already the right call: it is invariant to
*co*-transformations of gesture and layout together — which augmentation
exploits and layout swap requires — while staying anchored to the keys, so it
pays no lexical-ambiguity tax. Restoring the anchor to the shape encoder
takes only three numbers (bbox center + log scale, an invertible
factorization), which is why no training or LM effort can substitute for
them: the bits are cheap to keep and impossible to reconstruct.

The price with *everything* applied is now measured rather than estimated
(#50): through the identical fused search — deep lists, delta-form GPT-2,
joint commitment — shape-only reaches 90.94 against canonical's 94.62. The
AR decoder recovered the modeling tax, deep lists recovered the truncation,
and context recovered 6.5 of the remaining 8 beam points; the last 3.7 are
the collisions decoded context cannot reach — context-free positions and
contests where language is indifferent — exactly the classes the theory
said were irreducible.

## Experiment log

Every experiment in one place, including the failures — most of the negative
results cost real effort to establish and would cost it again to rediscover.
Detailed writeups live in the linked sections, the script docstrings, and the
commit messages.

| # | experiment | outcome | detail |
|---|---|---|---|
| 1 | canonical space + cross-corpus alignment | corpora agree to 0.124 key half-widths | `validate_alignment.py` |
| 2 | CTC encoder, 10 epochs | 78.4% greedy / CER 0.070 (val) | `train_encoder.py` |
| 3 | layout transfer | permutation ~free (qwertz −1.6); unseen 5-row grids cost 20-30 pts *at greedy* — but see #22 | `eval_layout_transfer.py` |
| 4 | error analysis | 85% of greedy errors are non-words → decode, not encoder | `error_analysis.py` |
| 5 | trie beam search | +13.2 over greedy | `eval_decoder.py` |
| 6 | lexicon 65k → 320k | OOV errors 41% → 15%; big lexicons don't hurt precision | `eval_decoder.py --lexicon` |
| 7 | oracle n-best curve | sized the reranking ceiling; most headroom at k≤4 | `eval_decoder.py --top-k` |
| 8 | cross-corpus gap, 8 hypotheses | 7 refuted; ~5.5 pts intrinsic by matched-size training — **revised to 1.8 by #81** (5k eval subset + hws label noise) | `diagnose_transfer.py`, `finetune_transfer.py` |
| 9 | beam 32/−9 → 64/−13 | n-best hit +2.2 on hws; a third of the "encoder" bucket was search | `eval_decoder.py` |
| 10 | bigram context reranking | +0.4/+0.6 — **only** the gated delta form works; raw logP hurts at every weight | `eval_reranker.py` |
| 11 | neural LM ceiling (distilgpt2, gpt2) | LM path saturates at +1.3 of 5.0; ~73% of headroom is acoustic | `eval_neural_rerank.py` |
| 12 | acoustic rescorer | +0.5 alone; stacks with LM to +1.95 (37% of headroom) | `train_rescorer.py`, `eval_full_stack.py` |
| 13 | rescorer capacity 96→160d | **null** — identical accuracy, pure overfit | commit `42a6168` |
| 14 | rescorer on encoder-unseen data | **worse** (4.6% vs 11.3% of headroom) — distribution shift beats overconfidence fix | commit `42a6168` |
| 15 | α/β re-tune for new config | **null** — optimum didn't move; sweep now ~1s via `beam_candidates()` | `tune_beam_weights.py` |
| 16 | test-time augmentation | **null** — posteriors are 0.9998 one-hot; nothing to average | `eval_tta.py` |
| 17 | encoder 10 → 20 epochs | +1.7 greedy, but lexicon absorbs ~10× → +0.18 beam (noise on futo) | commit `da6b142` |
| 18 | rescorer behind 20-epoch encoder | first-pass gains absorbed **twice**; compound only where encoder is the bottleneck (hws +0.88) | commit `3098170` |
| 19 | held-out test split, frozen config | **93.92%** top-1; every stage within 1.2 SE of validation — no tuning overfit | commit `a3909bb` |
| 20 | beam 64 → 128 | @8 ceiling 97.05 → 97.29 (val); truth is in the 128 surviving beams 97.9% of the time, so ~0.5 pt is top-8 *truncation*, not search | `eval_decoder.py --beam-widths 128` |
| 21 | full stack on beam-128 lists | **wash** — 93.85 vs 93.81 val; the stack converts ~17% of the extra ceiling and absorbs the rest, same pattern as #17/#18. Config stays beam 64 | `eval_full_stack.py` |
| 22 | layout transfer with the trie | the 5-row "geometry gap" was a decode-mode artifact: clearflow 49.0 → 83.4, within 0.9 of same-corpus qwerty; mixed-row-count training (#3's proposed fix) mostly moot | `eval_layout_beam.py` |
| 23 | length-normalized beam scoring | **null** — `a/len^γ` never beats the tuned `a + α·lp + β·len` at any k; the FUTO paper's better n-best ordering is list quality, not scoring family. α/β at beam 128 re-tune to within noise, still flat | scratch sweep over `beam_candidates()` |
| 24 | top-8 → top-16 lists, rescorer retrained to match | ceiling 97.29 → 97.68, stack converts 33% of headroom *exactly as at k8* — 93.95 vs 93.85, under 1 SE. The absorption ratio is invariant to list depth; decode side closed | `runs/rescorer128k16/` |
| 25 | head-to-head vs the released FUTO Swipe Model encoder (`honorable_sturgeon`, 635k, ExecuTorch) | same 20k swipes, same lexicon, same beam, α/β tuned per encoder: theirs 90.1 top-1 / 95.6 truth-in-beam vs ours 92.1 / 97.9. Their published 92.94 is their *decode stack* (γ/λ scoring their emissions are shaped for — greedy collapses to 48.2 — plus the eval-target-extended wordlist), not an encoder edge. Their efficiency stands: half the params, 844 swipes/s on one CPU thread | scratch, weights not redistributed |
| 26 | MMI fine-tune — train the emissions for the search (#25's mechanism, ported) | **+0.29 through the full stack** (93.81 → 94.10 val), from one epoch over the beam's own top-16 lists at LR 5e-5. First pass 92.0 → 92.3, @8 ceiling 97.05 → 97.41, hws zero-shot 80.5 → 80.7, greedy 80.1 → 79.0 *by design* — the same weak-greedy/strong-search signature the FUTO encoder shows. Zero inference cost. Rerun with a different dropout seed reproduces within ±0.05 everywhere | `train_mmi.py`, `runs/mmi/` |
| 27 | MMI round 2, on-policy (fresh lists from the fine-tuned encoder) | **null** — beam 64 92.4 vs 92.3, identical oracle curve; one round captures the gain. Consistent with the round-2 lists containing 40% fewer truth-missing swipes: the teachable discrimination was taught | `runs/mmi2/` |
| 28 | second test freeze: MMI config, measured once | **94.27%** top-1 (greedy 79.30, beam 92.40, +rescorer 92.73, ceiling 97.50); hws full 85k 80.4. Every stage within ~1 SE of validation, positive, same as #19 | headline table |
| 29 | error budget + confidence refresh on the frozen encoder | MMI absorbed a quarter of the doubled-letter gap (48.8 → 55.2 vs 81.7 without) — its only structural movement. OOV unchanged (30.3%), residual dominated by adjacent-key function words (`had`↔`has`). Posterior averaging still dead, but augmented views now match clean and a 4-view beam union trends positive under the noise bar — calibration is open, not a lever. Everything evening-sized is measured; what remains is project-sized | `error_analysis.py`, `eval_tta.py` |
| 30 | deferred commitment — draft/verify over the sentence word-lattice, speculative-decoding style | **+0.46 (lookahead-1) to +0.74 (joint) over streaming commit through the full stack** (20k val, matched rescorer: 93.49 → 93.95 / 94.23). Every prior LM number committed each word before the next swipe existed, so *right* context was untouched headroom — #11's "73% acoustic" was a left-context-only bound. Last words (no right context) gain ~0, confirming the mechanism, and decoded-context joint *exceeds* the oracle-context streaming stack on the same lists (93.81): right context more than replaces perfect left context. One word of lookahead keeps ~60% of the gain at a 1.8% revision rate; the LM accepts the first-pass draft 96% of the time; deferring only margin<2 words (8%) keeps ~70%. lm weight optimum stays 0.8, beam 8 ≈ 16 — flat surfaces again | `eval_deferred_commit.py` |
| 31 | truecase the decoded context (sentence-initial caps + "I") before LM scoring | **null** — 93.15/93.68/94.01 vs 93.14/93.67/93.99 at 20k, ≤0.02 at every stage. The decoded↔oracle gap is not recoverable surface casing; it is error propagation plus proper-noun casing, and #30 shows right context is the remedy that actually works | `eval_deferred_commit.py --truecase` |
| 32 | rescorer provenance — the current n-best dumps predate MMI (draft 91.86, ceiling 97.05: the original encoder's exact fingerprints), and #30's first sweep had stacked the *MMI* rescorer on them | the mismatched rescorer contributed ≈0 (streaming 93.14 vs 93.19 without); the matched one restores every point — draft +0.54, streaming +0.30, joint +0.28 (93.95 → 94.23). The rescorer's value is real but *pair-specific*: it corrects the first pass it trained against, same lesson as #14 from the other side. Freeze-grade numbers for #30 need MMI-encoder dumps scored by the MMI rescorer | `eval_deferred_commit.py --rescorer runs/rescorer/rescorer.pt` |
| 33 | deferred commitment on the frozen stack — MMI lists (`runs/mmi/nbest/`) + MMI rescorer, 20k val | **streaming 93.67 → lookahead-1 94.12 / joint 94.54** (+0.45/+0.87; ceiling 97.41 confirms the pairing). Decoded-context joint exceeds the frozen stack's *oracle*-context validation number (94.10) by +0.44 — the deferred stack needs no oracle anywhere. Margin-gated: deferring 8% of words keeps 94.28, 18% keeps 94.45. Candidate third freeze: beam 8, lm 0.8, rescorer 1.0, joint (or lookahead-1 where display latency must be bounded); test untouched | `eval_deferred_commit.py --nbest runs/mmi/nbest/futo_validation.npz --rescorer runs/mmi/rescorer.pt` |
| 34 | LM-scale ladder over #11's frozen lists — GPT-2 family to 1.5B locally, Qwen3.5 base 0.8B–9B on Modal A10Gs (~$1) | **flat: +0.6 to +1.4 across 124M → 9B, no model converts >29% of headroom.** Ladder's best is 2019's gpt2-xl (93.72); Qwen3.5-9B, 70× gpt2's parameters, ties gpt2-medium — modern curated pretraining is *worse per parameter* on lowercase informal text, needing ~9B to match WebText at 124M. Matches ASR (rescoring converts 3–11% on competitive baselines; BERT-base ≈ large, 70M ≈ Llama2-7B, T5-3B → PaLM-540B ≈ +0.5) and Gboard's own n-gram → neural swap (≤1.19% WMR). No LM-size ablation previously published for a gesture keyboard. Scale cannot buy what left context does not contain; the open lever was right context all along (#30, #33) | `eval_neural_rerank.py --lm --nbest-cache`, `export_rerank_bundle.py`, `modal_rerank.py` |
| 35 | third test freeze: deferred commitment, measured once on the full 48,711 | **94.62 joint / 94.34 lookahead-1 / 93.83 streaming** — draft 92.73 and ceiling 97.50 reproduce #28's stages exactly, so the lists are the frozen ones. Prediction posted before the read (93.7–93.8 / ~94.2 / ~94.6) held; every stage landed on the positive side of validation (+0.16 / +0.22 / +0.08), same as freezes one and two. The headline is now oracle-free: 94.27 conditioned the LM on the true prefix, 94.62 conditions only on the stack's own decoded words. Revision rates 1.9% / 2.5%; margin<2 gating defers 8.4% of words for 94.36 | headline table, `eval_deferred_commit.py` |
| 36 | frequency-bucket diagnosis of n-best misses | the encoder carries an implicit LM: truth-in-beam climbs 64% → 99.8% with training count; unseen words are 13.9×/4.7× overrepresented among futo/hws ceiling misses. futo's tail is out-of-lexicon proper nouns (unfixable, 0.5% actionable); hws's is in-lexicon everyday vocabulary, 3.9% of swipes | `nbest_freq_buckets.py` |
| 37 | permutation-mixture training — relabel 25% of samples under random letter permutations, matched real compute (13 epochs) | futo **wash** (−0.11 top-1, 0.6 SE; ceiling −0.09); hws +0.25 with unseen-word in-beam +4.1 (~4 SE) and the head untouched. The α surface steepens from <1 pt (#15) to 4.9 pts, α=0 now costs 3.5 — the frequency prior relocated from frozen emissions into the tunable knob. β re-optimum 2.4 (+0.24, parity with canonical). Greedy −1 to −2 by design | `train_encoder.py --permute-prob`, `runs/perm25e13/` |
| 38 | layout transfer with the permutation encoder | **+11 to +15 on every unseen grid** (clearflow 83.4 → 94.3, kasroz 84.0 → 95.7, dvorak 79.4 → 94.6 at beam 100, honest lexicon); qwerty flat; hit@8 ≈ 97.5% everywhere. #22's residual dvorak deficit erased; greedy on unseen grids +15–25 while qwerty greedy falls — the implicit LM was qwerty-conditional. Layout-agnosticism now holds at the geometry level, from 25% less real data | `eval_layout_beam.py --checkpoint runs/perm25e13/encoder.pt` |
| 39 | is the overconfidence (#16) the implicit LM? — TTA + posterior-sharpness probe on the permutation encoder | **no.** Posteriors identically one-hot (mean max 0.9902 vs 0.9899, 81% of frames >0.999 on both encoders); posterior averaging still dead (−0.12), 4-view beam union still under the noise bar (+0.13/+0.17). Overconfidence is CTC peakiness, not prior-driven certainty — calibration stays open, with sharper attribution | `eval_tta.py --checkpoint runs/perm25e13/encoder.pt` |
| 40 | do the two encoder levers compose? — MMI (#26 recipe, beam-128/top-16 lists, one epoch) on the permutation encoder | **yes.** First pass 91.75 → 92.11 (+0.36, same size as on canonical), ceiling 97.28; the gain lands in the rare-word tail (count 1–5 top-1 +2.6, target slice in-beam 92.7 → 94.6) with the head flat. α stays load-bearing (−3.6 at α=0) — MMI does not re-absorb the prior — and the β optimum returns to 1.2. Best hws numbers of any encoder (81.09 / in-beam 91.31); layout wins survive except dvorak −2.0 (~4 SE, still +13 vs canonical). Composed config lands 0.20 (~1 SE) under the frozen futo first pass with everything else better | `runs/perm25mmi/` |
| 41 | is overconfidence costing the search? — post-hoc softening sweep + full-CTC autopsy of never-surfaced words | **no.** Temperature/floor softening of cached emissions moves truth-survival by at most +0.08 (noise) while top-1 falls up to −1.2 — sharpness is load-bearing for ranking. Of the 268 in-lexicon misses at 20k, the truth loses to the beam winner by a median 7.9 nats (≈2700×) under a full-alignment score, and guaranteeing every letter ≥0.1% at every frame flips only 12% of the contests: write-off damage ≈0.16% of swipes ≈0.05 pts through the stack. Retraining for humility is priced and dead; the open confidence work is post-hoc — calibrated commit gating (#33) and lexicon escape | `probe_peakiness.py` |
| 42 | lexicon escape — greedy-vs-beam full-CTC score gap as an out-of-lexicon detector, escape-to-greedy as policy | **detector excellent, replacement dead.** AUC 0.974/0.924 (futo/hws) for "truth is not in the lexicon" — but the policy ceiling is a product of two small numbers: OOL truths are 1.2%/1.7% of swipes and greedy spells them exactly right only 22%/14% (the perm+MMI recipe lifts hws to 18%, futo unchanged), so net top-1 peaks at +0.02/−0.00 and false escapes outnumber recoveries at any useful sensitivity. The signal's real consumers are a secondary raw-spelling suggestion and the add-to-dictionary prompt, where a false alarm costs a candidate slot, not a correct word; transcription corpora also understate real OOL prevalence (names, slang) | `eval_lexicon_escape.py` |
| 43 | translation+scale-invariant gestures (`--shape-only`): per-gesture normalized shape features replace the affinity block, matched budget | **−16 greedy (62.5), −10 beam (81.8), −4.2 ceiling in-domain; −22 beam cross-corpus.** Structural, not a training deficit: qwerty congruences (`is`≅`od`, taps ≅ everything) cap a *perfect* shape matcher + unigram at 90.7% (quarter-key tolerance) — under the anchored first pass. Partial invariance doesn't pay either (translation-only −16%, scale-only −11% at half-key), and gpt2-xl with oracle two-sided context rescues ⅔ of the lost mass but lifts the anchored stack equally, needs the candidates surfaced, and dies at position zero. Anchoring is recoverable from 3 numbers (bbox center + log scale); nothing else substitutes | `probe_shape_collisions.py`, `probe_lm_rescue.py`, `runs/shape10/` |
| 44 | is CTC the right loss for invariant input? — full-CTC scores of congruent twins and their cross-terms on real swipes | **no.** The coupled posterior ("i goes with s, o goes with d") is inexpressible in factorized emissions: the incoherent cross-term `os` ties with truth (43% wins) while the coherent twin `od` loses by 7.9 nats of implicit-LM prior, and `on` swipes read as the more frequent near-twin `in` 65% of the time — identical inputs, contradictory labels, marginal emissions. On anchored input 19–36 nats of position evidence separates everything, so CTC was right for the canonical encoder and wrong only here | `probe_ctc_coupling.py` |
| 45 | autoregressive letter decoder on shape-only input (TCN trunk + 2-layer causal transformer head, trie-constrained AR beam) | **refunds the modeling tax exactly, and only that**: greedy +16 to 78.4 (canonical parity), truth-among-survivors 93.8 → 97.8 (canonical level), `os` cross-term 0 → −9.5 nats; but `is`/`of` and `on`/`in` stay lost, so beam top-1 recovers just 2.5 of the 10-point gap (84.4). Residual is pure ranking among surfaced congruent twins — oracle@2 = 91.9. hws greedy 35.4 → 53.4. Caveats: +400k decoder params, and CE-on-words bakes the unigram into the head (α optimum 0.8 → 0.2) | `model/ar.py`, `train_ar_decoder.py`, `eval_ar_decoder.py`, `runs/ar_shape10/` |
| 46 | the control the ablation earned: the same AR head on *canonical* (affinity) input, matched budget | **beam top-1 92.40 — ties the MMI-fine-tuned frozen first pass (92.31 val) with no MMI**, from a plain 10-epoch run. Greedy 86.7/72.8 (+8.2/+9.7 over CTC), @8 ceiling 97.8 (vs 97.05 CTC, 97.41 MMI), truth-among-survivors 98.4. The greedy gain absorbs on the way up exactly per #17/#18: +0.54 beam in-domain, wash on hws beam (80.9 vs 80.8) with the @8 ceiling +0.7. Doubled letters +2.0 top-1 / +2.5 in-beam with singles flat — the repeat-collapse pathology CTC cannot express, priced; the trie had been compensating most of it. OOV rises to 16% of residual errors (list quality up). Caveats: +400k params, sequential decode, unigram partly internalized (α optimum 0.4) | `runs/ar_full/`, `dump_ar_nbest.py` |
| 47 | full stack on the AR lists — pair-matched rescorer (#32's lesson) + deferred commitment, val 20k | **every stage clears the frozen stack: streaming 93.85 (+0.18), lookahead-1 94.42 (+0.30), joint 94.66 (+0.12), ceiling 97.80 (+0.39).** The rescorer converts only 4.3% of its headroom (+0.23) vs the usual ~10 — AR scores are already sequence-coherent, so the second pass has less to add; the first-pass +0.54 absorbs to +0.12 at joint, the absorption law yet again. Margin-gated: defer 8.7% of words for 94.39. **Not promoted yet** — joint's edge is under 1 SE and a freeze spends a test read (#18/#40's logic); the untried lever is the MMI-analogue (sequence-discriminative fine-tune over the AR beam's own lists, #26's +0.36 recipe), which should be measured first. If the composed gain holds, that is freeze four | `train_rescorer.py --nbest data/nbest_ar`, `eval_deferred_commit.py --nbest data/nbest_ar/...`, `runs/rescorer_ar/` |
| 48 | MMI on the AR decoder — softmax over each swipe's rival set (truth always injectable, unlike CTC's trie), one epoch at 5e-5 over the beam's own 150k lists | **+0.15 first pass (92.40 → 92.55), half of what CTC's MMI bought** — the AR head trains sequence-discriminatively from birth, so there is less left to teach; same weak-greedy/strong-search trade (greedy −1.4). Composed stack on the MMI lists: **streaming 93.92 / lookahead-1 94.55 / joint 94.73** — the best validation numbers of the project, +0.19 over the frozen stack. Still ~1.2 SE; the freeze stays unspent | `finetune_ar_mmi.py`, `runs/ar_mmi/`, `runs/rescorer_ar_mmi/` |
| 49 | is there a cleaner join than n-best → rescorer → lattice → deferred layer? — fused sentence-beam (one search, LM in the loop, commitment = pruning lag) × prior-algebra grid on Modal A10Gs, 45 configs, same MMI deep lists (64 cands/swipe, ceiling 98.33 vs 97.85@8) | **yes, to within the rescorer's +0.11: fused + delta-form LM = 94.62 joint vs the three-pass 94.73, with no rescorer, no separate deferred pass, one score formula.** The grid is a controlled demonstration of prior algebra: raw LM + unigram degrades monotonically with LM weight (93.89 → 92.65 — #10's lesson reproduced at scale); subtracting a prior *once* fixes it, either the decoder's internal LM (HAT-style, zero-memory ablation, 94.53, flat ridge λ 0.3–0.4 · μ 0.8) or the LM's own unigram (delta form, 94.62); subtracting *both* collapses to 93.59 — the subtractions are alternatives, not complements. Deep lists pay +0.2 only under correct algebra (~0 under raw); zero-ablation beats mean everywhere. Commitment lags come free from the same beam: lag 0 = streaming, 1 = lookahead, ∞ = joint | `export_fused_bundle.py`, `modal_fused_search.py`, `probe_ilm_fusion.py`, `runs/fused_modal*.log` |
| 50 | the invariance thread's closing number: shape-only AR lists through the identical fused stack | **90.94 joint vs canonical's 94.62 — the fully-realistic price of translation+scale invariance is 3.7 points**, down from 8.0 at the beam. Deep lists lift the shape ceiling to 97.56 (from 96.1@8 — coverage was never the problem), context converts 6.5 of the 8 beam points (vs 2.2 for canonical: the LM does 3× the work when acoustics can't separate congruent twins), and the streaming→joint spread triples (3.1 vs 1.0 — right context matters most exactly where evidence is weakest). μ optimum stays 0.8 even here — more LM *work*, not more LM *weight*. ILM subtraction loses to delta by 0.9 on shape (vs 0.1 canonical): the 25% of tokens without sentence context have nothing but the prior, and shape has no acoustics to catch them. The residual 3.7 is the theory's irreducible set — context-free positions and context-indifferent contests | `runs/fused_shape_delta.log`, `runs/fused_shape_ilm.log` |
| 51 | can LM scale close the invariance gap? — gpt2 → gpt2-xl 2×2 over both bundles, fused joint, matched config | **no: both lift +0.4, the gap does not move (3.73 → 3.75).** Scale's gains land on context-*soluble* errors, which both encoders share; the invariance residual is context-free or context-indifferent by construction and no reader size touches it. **The control cell, however, revises #34's scope: scale is not flat in-search.** gpt2→xl buys +0.43 at fused joint on canonical (vs ~+0.2 for the same swap as a second-pass rescorer), concentrated where the LM has decoding authority (streaming +0.12, joint +0.43) — and canonical fused + xl reaches **94.88, the first validation number above the 2-SE freeze bar.** #34's conclusion stands for second-pass rescoring and falls for in-search fusion; the ladder above xl is reopened. Ops lesson, twice-paid: the fused search is kernel-launch-bound at batch ≤ 8, so single cells run *faster locally* (11 min/pass on MPS fp16) than on cloud GPUs — Modal buys config-parallelism, not single-cell speed | `run_fused_local.py`, `runs/*_xl8_local.log` |
| 52 | one decoder, two conditioning streams — #49's ladder rung 3, the "horizon" design: a single AR decoder over the sentence's letters (self-attention over the cross-word token stream is the LM; per-word cross-attention into gesture memory is the acoustics), same trunk/head/params as #46, matched 10-epoch budget, then the monolith as the *only* scorer through the identical fused beam (same deep MMI lists, M=8, no α/β/μ, no external LM) | **loses 3.3–3.8 joint — 91.10 vs 94.45 (fused+gpt2) / 94.88 (xl) — and the streaming→joint spread collapses to zero (91.13/91.10 vs 93.59/94.45).** The corpus-internal LM (~5 MB of prompt text) converts +0.7 of left context and *nothing* from the right — too weak to overturn its own acoustic stream, so deferred commitment, the stack's best lever, dies inside the monolith. Sharing the weights also taxes the acoustics: the no-context control scores 90.42 where the word-level AR's raw score ranks the same lists at 91.65; sentence context only refunds that deficit. Ensembling the word model back in reaches 91.99, spread still zero — the deficit is the internal LM, not the search. What context it does learn is #36–38's baked-in prior, now measured directly: +1.1 on the 7% of val sentences whose text occurs in train, and hws greedy −2.7 vs word-level. Greedy parity in-domain (86.2 vs 86.8): the architecture is fine; the design is wrong at this scale. Rung 3 was a known trade (#34/#36–38 argued it); it is now a price | `model/sentence_ar.py`, `train_sentence_ar.py`, `eval_sentence_fused.py`, `runs/sent_ar/`, `runs/sent_fused*.log` |
| 53 | **fourth test freeze: the fused AR decoder, measured once on the full 48,711** | **95.20 joint / 94.72 lookahead-1 / 93.85 streaming, deep-list ceiling 98.22** — every stage on validation's positive side (+0.05/+0.09/+0.08), prediction (95.1–95.3) posted before the read, truth-in-list fingerprint 98.39 matched the dump. Config: AR-MMI encoder (#46/#48) → trie-constrained AR beam, 24 candidates/swipe → fused sentence-beam, delta-form gpt2-xl at μ=0.8, beam 8, joint commitment (#49/#51). The headline moves 94.62 → 95.20; lookahead-1 (94.72) now exceeds the old joint headline with display latency bounded to one word. First freeze to change the first-pass architecture and the decode topology at once; the four-for-four positive-side record holds | headline table, `run_fused_local.py --bundle fused_bundle_test.pkl` |
| 54 | is there an optimal *partial* invariance? — gesture-only translation jitter (σ keys) during AR training, the layout untouched: σ=0 asserts an exact anchor, σ→∞ marginalizes it (shape-only) | **no — monotone destruction from the first step, and fastest exactly where the hypothesis predicted gains.** val beam 92.40 → 91.90 → 90.99 → 89.83 and hws 80.92 → 79.68 → 76.68 → 73.31 across σ ∈ {0, 0.25, 0.5, 1.0}; cross-corpus falls *faster* than in-domain (−1.2 vs −0.5 at σ=0.25). The anchor-noise story fails because #1 already aligned both corpora to 0.124 key half-widths — calibration was solved upstream in the canonical space, so jitter models noise that does not exist and only destroys evidence transfer uses. Re-closes #8 from a new angle (miscalibration joins the seven refuted transfer hypotheses) and completes the invariance curve end to end: the optimum is exactly σ=0. Caveat: both corpora are lab-normalized; raw field data without a calibration pass could still want σ>0 — a claim about dirtier data, not this pipeline | `train_ar_decoder.py --anchor-jitter`, `runs/ar_aj*/`, `runs/anchor_jitter_sweep.log` |
| 55 | #37's permutation mixture ported to the AR architecture — 25% of samples relabelled under letter permutations, 13 epochs, then the swipe-5 layout exam the fused-era first pass had never taken | **the AR head is a far bigger qwerty-memorizer than CTC's emissions ever were — and the same 25% dilution buys it back.** Baseline ar_full transfers terribly: dvorak 37.6 beam / clearflow 77.5 / kasroz 70.8 (the CTC canonical encoder managed 79.4/83.4 on the same exam, #38 — the decoder head's letter-LM binds letter statistics to qwerty geometry, exactly the mechanism #52 measured from the other side). Perm-25 restores dvorak to **86.4 (+48.8)**, clearflow 90.3, kasroz 91.1, qwertz 87.3, at a cost of −0.4 beam on swipe-5 qwerty, −0.16 val beam (92.24 vs 92.40, inside noise), −1.4 greedy in-domain, hws beam flat (80.7 vs 80.9) — #37's shape exactly: the implicit LM pays for the transfer, the external stack absorbs the greedy tax. Coupling probe intact (truth wins every congruence contest). Residual vs perm-CTC's 94.3/94.6 (#38) suggests 25% under-dilutes the AR head specifically — rate, or head-only permutation, is the open lever | `train_ar_decoder.py --permute-prob`, `eval_ar_layout.py`, `runs/ar_perm25*` |
| 56 | can synthetic gestures train the decoder? — WordGesture-GAN (CHI'23) read for the recipe, but its *baseline* built first: the minimum-jerk generator (via points at key centers + fitted aiming noise, CLC duration law m·L^n fit by grid+closed-form on train, both trajectory profiles: rest-to-rest straight segments and the global via-point quintic spline), 917k synthetic swipes mirroring futo/train's exact word multiset, decoder trained on nothing else | **the paper's headline utility result does not survive contact with a learned decoder — and realism ranks backwards.** Their SHARK² gap for synthetic-only training was 0.8 WER; ours is 26 points at the beam (66.3 vs 92.40 val top-1, greedy 38.2 vs 86.7). The profile ablation is the finding: the *segments* profile (straight lines, dead stop at every letter — mean-jerk 3.4× below real, exaggerated dwell) trains a usable-if-weak decoder (38.2/66.3, truth-in-list 88.4), while the *spline* profile — closer to real on every aggregate stat (speed p50 2.48 vs segments 2.20, real 2.88; dwell fraction 0.18 vs real 0.20) — is nearly untrainable: greedy 8.5, beam 35.1, and real-val CER *diverges* past 1.0 after epoch 0. Same inversion under the frozen decoder (segments 78.0 readable, spline 70.7, real 85.9). Synthetic training data lives or dies on the per-letter dwell cue, not distributional realism — the spline glides through interior letters and marks nothing. Sets the bar a learned generator must clear: reproduce dwell *texture*, not kinematic statistics | `minjerk.py`, `probe_minjerk.py`, `gen_minjerk_corpus.py`, `runs/ar_minjerk*`, `runs/minjerk_probe.log` |
| 57 | #56's failure read as sim2real domain shift: don't chase fidelity, *randomize* — per-gesture profile coin-flip, dwell pauses at 60% of interior vias (lognormal ~80ms), per-segment timing jitter, correlated tremor calibrated so synthetic mean-jerk matches real (0.20 key half-widths → 4376 vs 4139), same 917k-swipe protocol, decoder still trained on zero real gestures | **synthetic-only greedy 73.4 / beam 85.65 / truth-in-list 97.66 on real val — the sim2real gap collapses from 26 beam points to 6.75.** Against the pure profiles (segments 38.2/66.3, spline 8.5/35.1) randomization is worth +35 greedy over the best of them, and the real-val curve now improves monotonically through all 13 epochs where spline diverged after one. The mechanism is the standard DR argument made concrete: when every temporal texture varies per draw, the only invariant left to fit is where the trajectory passes — the decoder is forced onto the letter-position signal precisely because nothing else is stable. Readability under the real-trained frozen decoder *fell* to 70.9 (vs segments' 78.0) while training utility soared — the two metrics measure different domains' cue expectations, and utility is the one that matters. α optimum rises to 0.8 (weaker internalized statistics from texture-random data). hws: 56.2 greedy / 71.4 beam | `minjerk.py` dwell/tremor/seg_jitter knobs, `runs/ar_minjerk_rand*`, `runs/minjerk_rand_probe.log` |
| 58 | the mixing curve — what synthetic data is actually *for*: real thinned to 1% (9,168) / 10% (91,683), each alone, concat-mixed with the #57 corpus, and the sim2real classic (pretrain on 917k synthetic → fine-tune on the real slice); all 13-60 epochs to convergence, identical eval | **synthetic pretraining is worth ≥10× real data at the low-resource margin: fine-tune-on-1% (89.39 beam, 80.8 greedy) beats real-only-10% (89.23, 77.4).** Full curve (val beam): real-only 81.02 → 89.23 → 92.40 across 1%/10%/100%; +synthetic lifts the 1% point by +8.4 (concat 87.90, fine-tune 89.39) and the 10% point by +1.6 (ft 90.82, −1.6 from full-real with 10× less data). Fine-tune ≥ concat at both fractions (the 100:1 dilution wastes real signal exactly as predicted), synthetic-only (85.65) outranks real-only-1% (81.02) — 917k randomized min-jerk gestures beat 9k real ones — and fine-tuned models recover α=0.4 (the internalized LM returns with real texture). The paper's Table 7 story survives in this form only: synthetic can't replace real (#56) but multiplies scarce real data — the regime every new-layout / new-language deployment actually lives in. hws follows the same order (ft-10 78.4 vs real-10 76.8) | `train_ar_decoder.py --train-path a:N,b --init`, `runs/ar_{real,mix,ft}{1,10}*` |
| 59 | layout onboarding with zero real gestures: #57's randomized generator run on clearflow/dvorak/kasroz/qwertz geometry (200k each — the duration law and aiming noise are functions of the target layout, so they transfer by construction), ar_full fine-tuned on them + 200k real-qwerty replay; then the stacking cell, same recipe from ar_perm25 with permutation kept on | **synthetic geometry beats unlearning qwerty on 3 of 4 layouts — clearflow 94.1, kasroz 94.4 (vs perm-only's 90.3/91.1, reaching #38's perm-CTC gold 94.3/94.6 with no real target-layout data) — but dvorak refuses: 81.1, five under perm-only's 86.4, and the combo cell does not rescue it (82.9).** The dvorak anomaly is a ranking failure, not a coverage one: synthetic-tuned greedy 72.1 far above perm's 63.4 and hit@8 even (91.8 vs 92.7), yet beam top-1 inverts — the model reads dvorak gestures better and *ranks* their candidates worse. Combo elsewhere: clearflow 94.6, kasroz 94.7 (small stacks), qwertz flat, qwerty replay holds regression to −0.8 (swipe-5) / −1.1 (val beam 91.31). Deployment recipe as of now: synthetic onboarding for a new layout, permutation training only when the layout is unknown at training time — and dvorak stays an open case | `gen_minjerk_corpus.py --layout`, `train_ar_decoder.py` @layout specs, `runs/ar_layout_ft*` |
| 60 | #36's actionable slice attacked directly: every wf320k word with ≤2 real training gestures — 269,089 words, 93% of the usable lexicon — given 2 randomized synthetic gestures each (uniform, not frequency-weighted: the trie supplies identity, gestures only teach geometry), ar_full fine-tuned on 538k tail + 200k replay, before/after n-best dumps bucketed by training count | **the intervention lands exactly on its target and pays a measurable head tax: hws unseen-word top-1 +9.8 (33.0 → 42.9), the in-lexicon count≤5 slice +7.5 (51.8 → 59.3, ~27 SE), in-beam +13.1 on unseen — while every bucket above count 5 loses 0.9–4.1 and the aggregate slips −0.6 (hws) / −0.9 (val).** Uniform tail coverage is implicit-LM dilution by other means (#37's mechanism through the data instead of the labels): probability mass moves from head to tail, and the head loss is exactly the kind the fused stack's external LM exists to absorb, while the tail gain — acoustic coverage — is the kind nothing downstream can substitute. Whether the trade nets positive end-to-end is a fused-stack question (dumps are ready); val shows the same shape smaller (its tail is out-of-lexicon proper nouns, #36's original diagnosis). First measured knob that moves the unseen-word bucket at all | `gen_minjerk_tail.py`, `runs/ar_tail_ft*`, `runs/tail_buckets_{val,hws}.log`, `data/nbest_{base,tailft}/` |
| 61 | does the fused stack refund #60's head tax? — ar_full vs ar_tail_ft deep lists (M=24) through the identical fused sentence beam (gpt2, delta-form, μ=0.8), val + hws, per-word hyps bucketed; α ∈ {0.4, 0.8} both arms after spotting that the harness's hardcoded α=0.4 shortchanges a dilution-trained first pass | **no — the refund hypothesis is refuted at best-config-per-arm: val 94.53 → 94.18 (−0.35), hws 81.12 → 80.08 (−1.04).** The α interaction is real and diagnostic (0.8 lifts tail-ft +0.3, costs base −0.25/−0.7: the diluted model leans on the external unigram exactly as its α*=0.8 first-pass optimum said) but it halves the tax rather than erasing it. The slice structure survives fusion untouched — hws unseen **+7.3**, rare +3.0, head −1.6 at matched α — so the trade is now measured end-to-end: tail synthesis buys its coverage at a real, stack-surviving head price, and gpt2-scale context cannot overturn confident-wrong head rankings (#51's authority ladder suggests xl would refund more; untested). Breakeven arithmetic from the measured deltas: net ≈ 0 at ~23% tail mass — hws sits at 18.6% (−0.3 at matched α), futo val at 10% (clear loss). The mechanism works; the dose was wrong — 269k uniform tail words flatten the prior far past any real usage distribution. The dose-response (tail restricted to plausible words, 1 sample/word, lighter mix) is the open lever if tail coverage is ever wanted in the deployed stack | `run_fused_local.py --alpha --lags --save-hyps`, `runs/fused_{base,tailft}_{val,hws}*.log`, `runs/hyps_*.npz` |
| 62 | the two cells that kept the synthetic thread open: (a) mix100 — full real corpus + the 917k randomized synthetic, from scratch, does synthetic help where real data is already abundant?; (b) #61's dose-response prediction — tail restricted to plausible words (top-50k wordfreq ∩ train-count≤2 = 29,200), one gesture each, 7% of a 400k-replay fine-tune instead of #60's 73% | **(a) no: val greedy jumps +1.3 (86.7 → 88.0, the biggest greedy in the log) and the beam absorbs every point of it (92.46 vs 92.40), hws beam slips −0.7 — the absorption law (#17/#18) one more time; synthetic's value does not extend to the full-data margin. (b) yes: the dose was the problem, exactly as #61's breakeven arithmetic said.** First pass: hws aggregate flips sign to **+0.65** (unseen +3.1, rare +2.2, head buckets flat), val −0.08 (noise). Through the fused stack: **+0.23 end-to-end** (81.12 → 81.35) with unseen +1.3, rare +1.8, head −0.07 — net positive, structurally clean, list ceiling +0.55. The dosed-tail recipe (plausible words only, 1 sample/word, light mix) earns a standing place; the uniform-lexicon dose stays refuted. Where the thread actually stands: synthetic replaces nothing, multiplies scarce real data (#58), onboards known geometry (#59), and covers plausible tails for ~free (#62b); still open — the DR-config sweep (85.65 is a floor, dwell/tremor were one guessed point), xl's refund authority, the learned generator | `runs/ar_mix100*`, `runs/ar_tail50_ft*`, `runs/tail50_buckets_*.log`, `runs/fused_tail50_hws.log` |
| 63 | can a *learned* generator beat the analytic one? — WordGesture-GAN (CHI'23) read for the recipe, then five architectures against it, all judged synthetic-only on real val: v1 free-running regression, v2 Graves-style MDN steps, v3 prototype + monotone time warp, v4 v3 with offsets in a low-frequency basis, and a trajectory diffusion model; plus the published method implemented to spec | **yes, eventually — diffusion ties min-jerk at 85.43 vs 85.65 beam, after three architectures that failed on geometry rather than texture.** The failure axis is accumulation: v1's mean-seeking steps compound into a path *shorter than the polyline through its own letters* (0.86x, so it cuts corners and strands the gesture before the last key — caught by eye before any metric flagged it), v2's sampled steps compound into a random walk (path 5-15x, off the keyboard) and score *below* v1 despite better texture. v3 removes integration entirely (curve = prototype + offsets, sampled at a monotone warp normalized to end at 1, so reaching the last letter is structurally guaranteed): +32 beam. Diffusion denoises all 64 points jointly — same immunity, no mode-averaging — and lands nearest real on every geometry statistic (path 1.10 vs 1.10, end-err 0.060 vs 0.067, C2ST 0.61 against a 0.51 real-vs-real floor, where min-jerk sits at 0.77). Published WGG to spec: proxy 0.347, below every arm here, though its critic was still winning at cutoff — undertrained, not refuted | `model/gesturegen.py`, `model/gesturediff.py`, `model/wgg.py`, `runs/ar_gen*`, `runs/gesturediff*` |
| 64 | the controls that make #63 a claim rather than an anecdote: (a) a 3-minute quality oracle — geometry/texture/shape stats, a real-vs-synthetic classifier, within-word diversity, and a *proxy decoder* (96-dim, 3 epochs, 120k gestures, greedy on real val) — calibrated against six corpora with known full-scale numbers; (b) the reconstruction ceiling: encode real gestures, decode the posterior mean, train on that | **(a) the proxy tracks full-scale beam at Spearman 0.94 (0.77 vs greedy) for 1/30th the compute, and predicted diffusion's unseen 85.43 from 0.572 before it was run; (b) the warp family is capped by its parameterization, not its prior — reconstructions of *real* gestures score 0.531 proxy against the family's own sampled 0.521 and real data's 0.662, with dwell 0.159 vs real 0.229 even when copying a specific gesture.** So no better prior, GAN, or sampler could have rescued v3/v4, and diffusion — which has no such bottleneck — already scores past the ceiling. Diversity rules out posterior collapse everywhere (0.40-0.45 vs real 0.36). Third realism/utility inversion, this one costly: v4 fixed the zig-zag a human observer flagged in v3 (turn 0.11 vs 0.18, zigzag 0.14 vs 0.23, closer to real on both) and utility *fell* 0.521 -> 0.454 | `gen_quality.py --calibrate`, `gen_reconstruct_corpus.py`, `runs/gen_quality_calib.log` |
| 65 | if realism and randomization are different goods, are they complementary? — 50/50 mixture of the diffusion corpus and the domain-randomized min-jerk corpus, 917k total, otherwise the identical synthetic-only protocol | **yes, and the mixture is the first synthetic corpus to clear 88: val beam 88.54 / greedy 79.9 (+2.9 over min-jerk's 85.65 and +3.1 over diffusion's 85.43), hws 75.5, truth-in-list 97.8 — 3.9 short of real data's 92.40 from gestures no human produced.** The parents are doing different jobs: diffusion supplies realistic geometry and timing (C2ST 0.61), min-jerk supplies variation so wide it leaves the keyboard (path 1.76x, turn 0.92 vs real 0.27) and prevents the decoder leaning on any one regularity. Neither substitutes for the other, which the sampling sweep confirms from the other side: raising diffusion's own stochasticity (eta 0 -> 1) moved the proxy not at all (0.572 -> 0.572), because within-distribution noise is not the out-of-distribution randomization that does the work; 200 denoising steps bought +0.016, a fidelity gain. Recipe for synthetic gesture data as of now: learn one generator, hand-build a deliberately unrealistic one, mix them | `runs/ar_gen_mix*`, `runs/quality_gmix.json` |
| 66 | **#51's reopened ladder, climbed — and the first climb was wrong.** Seven LMs (gpt2 124M→1.5B, Qwen3.5 0.8B→9B) in-search on the byte-identical fused bundle, one config, fixed 3/8 val slice, every comparison paired (McNemar over discordant words) | **scale pays in-search where it was flat as a rescorer, and the payment is authority: gpt2→xl is +0.24 at streaming (p=0.11), +0.42 at lookahead-1, +0.56 at joint (p=1.2e-04)**, converting 42–60% of headroom against 25–29% for the same models as a second pass. **Above gpt2-xl the ladder keeps climbing — the GPT-2 family saturates at 774M (large = xl = 95.22) and Qwen3.5-9B passes it at 95.54 (+0.32, p=0.037), 2B ties at 95.38, 0.8B draws level at 95.04 on half xl's parameters.** That reverses this cell's own first pass, which had every Qwen rung 1–2.7 points low: `delta` = logP(w|ctx) − logP(w) estimated logP(w) as logP(w|start token), a proxy correlating 0.92 with the corpus unigram for gpt2 (real BOS) and 0.77 for Qwen (none, gets <|endoftext|>), subtracted from every candidate. Estimating the prior over neutral contexts instead moves gpt2-xl by −0.01 (p=1.0) and Qwen-0.8B/2B/4B/9B by **+2.65/+1.22/+1.74/+1.15** (p≤5.5e-14). Four first-pass 'findings' were that one asymmetry: the modern-family deficit, a *sharp* family-specific μ surface in a project of flat ones (fixed, all eight want μ=0.8), 0.8B below the no-LM floor, and 4B's dip below its own smaller sibling (−0.64 → −0.12, n.s.). Fit is not usefulness: per-word NLL ranks gpt2-xl best (5.62) and Qwen-9B worst (6.50), exactly inverting the in-search order — delta removes the prior by construction, so absolute calibration is beside the point and the prior term is the one place it leaks back in. **#34 uses the same convention on the same checkpoints and wants re-running** (done: #72) | `run_fused_local.py --uncond marginal`, `compare_hyps.py`, `probe_lm_fit.py`, `runs/{ladder,uncond38,musweep}_*.log` |
| 67 | training-free floor: swipe → nearest-key trace string → base LM inverts it few-shot, prompt built entirely from straight-line templates (corpora eval-only). Plus the two channel diagnostics that shaped rung 3 | **10.0% / 21.2% top-1 (none / oracle context, n=500, Qwen3.5-2B; 9B ties 2B)** — the LM emits frequent words agreeing with first letter and context and ignores the interior; character-level trace reading is not a prompting competence. The diagnostics cut the other way: the collapsed label is a strict subsequence of the trace for 78% of real swipes (misses are corner cuts — `that` with no `h`), and the analytic alignment cost alone ranks the true word first among 2k common words on **88%** of swipes, top-8 100% (n=100, untuned constants) — geometry is nearly sufficient, enumeration is the hard part | `trace.py`, `eval_llm_trace.py` |
| 68 | training-free joint decoder: LM token beam with the analytic alignment cost in-search — the LM's vocabulary as the lexicon. Letter-string hypotheses, canonical re-tokenization every step, per-letter proposal quotas with forced single-char tokens, LM-only companion pass, unexplained-tail bound on partials. One knob (lm_weight, flat 0.5–1.5), widths set on 50 val swipes, measured on the disjoint 150 | **oracle context 81.3 top-1 / 87.3 n-best; cold start 54.7 / 66.0 (2B, ~9 s/swipe MPS)** — beam-level accuracy (trained trie beam: 91.9) with zero gesture training and no lexicon. En route: Qwen3.5's `<|endoftext|>` distribution is flat noise (#66's asymmetry; a neutral prime is worth ~15 pts cold), left padding corrupts its linear-attention fallback, char-token paths under-score words ~2x vs canonical tokenization, and `min(row)` partials breed `cffff…` degenerates until the tail bound kills the class. Recall is the whole bottleneck: whenever the true word finished it won the pool, every gain came from proposal width, and the residual misses are proper nouns (`androscoggin`, `hanseatic`) — compute and the proper-noun tail are exactly the candidate enumeration a lexicon buys | `geomllm.py`, `eval_llm_beam.py` |
| 69 | #68's three open levers, pulled: (a) speed — context KV cache with batch expansion (manual state cloning for Qwen3.5's hybrid DynamicLayer + LinearAttentionLayer cache, parity-probed), batched softmax/gather replacing per-row reductions, row dedup across lockstepped passes, vectorized DP extension; (b) the LM ladder in-search (gpt2-xl, Qwen3.5 0.8B/2B/9B, oracle context, disjoint-150); (c) cross-corpus on HWS test | **(b) 2019's gpt2-xl tops the ladder at 88.7/92.7 over 9B's 85.3/90.0 (xl > 2B p=0.008; xl vs 9B +5 n.s. p=0.27; 0.8B = 2B = 80.7) — #66's refuted first-pass ordering, resurrected legitimately: this decoder ranks by raw logP(word|ctx), no delta form, so absolute fit decides, and the ordering tracks probe_lm_fit's NLL table exactly (xl 5.62 best, 9B 6.50 worst) where the delta-form fused search inverts it. Whether scale or distribution match wins is a property of the scoring form, not the models.** (a) 1.6x on Qwen and a hard ceiling found: the linear-attention torch fallback costs ~11ms/row regardless of sequence length (batch-bound, cache saves 13%); gpt2-xl's full attention runs 1.09 swipes/s — the most accurate rung is also 6-10x the fastest. (c) HWS cold: 36.7/50.7 vs trained stack's 80.4. Decomposition: geometry alone still 70% top-1 / 96% top-8 there (apparatus noise −18 from futo's 88), so the collapse is enumeration-without-context — misses are common words (`plane`→`one`), not exotica. LM-as-lexicon does not survive cold on a foreign corpus; a lexicon would inherit the 70% for free | `eval_llm_beam.py --offset`, `geomllm.ContextCache`, `runs from scratchpad dumps` |
| 70 | the cell that arbitrates #68's deficit — training-free geometry + wf320k trie + unigram prior + optional gpt2-xl rescore, i.e. keep the bias-free first stage, put the lexicon back. Motivating hypothesis under test: training-free removes stage-1 implicit-LM bias, lexicon-free removes stage-2 OOV | **stage-1 substitution is nearly free, stage-2 substitution caused the whole deficit: 90.0/94.0 oracle on the same 150 (statistical tie with the LLM-as-lexicon beam, 5v3 discordants p=0.73, at 20x the speed; trained trie beam 91.9), 72.7/90.0 with no LM and no context at 4 swipes/s, 56.0/77.3 cross-corpus cold (+19 over LM-as-lexicon; trained stack 80.4).** The lexicon-free premise dissolves on contact: only ~0.7% of refs sit outside wf320k (#6 already said the tail was small) and the lexicon-free beam decoded zero of them — LM-as-enumerator relocates LM bias from scoring, where it misranks, to proposal, where it makes words unreachable. Unigram prior worth 68 → 98 on the dev slice (geometry cannot fight 320k confusables alone; the prior is SHARK²'s job); cold raw-LM rescore ≤ unigram (70.7 vs 72.7) — #10's gating result in a training-free costume. Remaining misses are the true proper-noun tail (`androscoggin`, `swanland`, `batam`) plus HWS apparatus noise (88 → 70 standalone), the residual that is genuinely acoustic | `eval_geom_trie.py` |
| 71 | the residual attacked training-free: dwell weighting (transit costs scaled by local finger speed — arclength resampling had discarded timing), label-free touch-offset calibration, and the three-channel ranking formula, all tuned on a fresh 200-swipe slice, then frozen reads on the untouched 150 + HWS + full 20k val | **timing is the one that pays: +5.3 LM-free (72.7 → 78.0) and +5.3 cross-corpus (56.0 → 61.3, tuned on futo only — it transfers); full 20k val 79.2/91.7 with no LM, no context, no training, level with the trained encoder's greedy 78.4.** Calibration is a null — the measured bias is ~7% of a key because the canonical space already absorbed it (#1's alignment work paying out again). Delta-form rescore ties raw logP once geometry + unigram are both in the score; the knob surface is flat (every oracle-rescore variant within 2-4 words, final frozen 89.3/94.7 vs the trained beam's 93.3 on the same 150). Residual after all of it: candidate ranking on the sloppiest swipes (the trained noise model's last genuine edge), 1.5% wordfreq OOV, and coin-flips | `eval_geom_trie.py --time-weight --calibrate --rescore-unigram` |
| 72 | #66's standing obligation discharged: the #34 second-pass ladder re-run with the marginal prior, byte-identical frozen lists, local MPS | **every rung rises — Qwen 0.8B/2B/4B/9B +0.68/+0.72/+0.82/+0.62 decoded, and the gpt2-xl "control" +0.40 (0.9372 → 0.9412) where the fused search had measured marginal a GPT-2 no-op: as a rescorer the delta term is the only LM signal, so even the 0.92-correlation proxy leaks.** Every optimum snaps to weight 0.8 delta (bos had scattered them 0.3–0.5, family-specific — #66's fabricated surface again); peak conversion is now 40% of headroom (9B decoded 0.9424, +1.96 of +4.96), so #34's "no model converts >29%" falls and #11's "73% acoustic" softens to ~60%. The scale claim survives the fix: 31–40% across the ladder, 9B over xl by 6 words in 5,000 (<1 SE) — flat as a second pass, and in-search authority (#66) stays the only regime where scale pays. The 4B rescoring dip persists but shrinks to noise | `eval_neural_rerank.py --uncond marginal`, `runs/lm_ladder/ladder_*_marginal.log` |
| 73 | the proposal rung above #66's ladder — out-of-list candidates (geometric trie re-search; gpt2-xl's vocabulary via #68's beam) injected into the fused search, on #61's hws base arm (81.12), harness parity-verified word-for-word against the saved baseline | **the rung as designed is null, and the null bought a bigger lever.** AR-as-veto: truth injected on 340 of 1,432 coverage misses, 8 win, +0.03 (p=0.27; the LLM arm −0.16) — scoring proposals with the model that pruned them is #70's bias relocation in reverse, #41's 8-nat losses met constructively. Underneath: within-list AR score and GestureDP cost correlate at **−0.17** — near-independent channels — so geometry goes *in the score*, not behind it. `acoustic = ar + β·len + α·uni − γ·geom` (γ=0.5 dev-picked) for every candidate: **eval 80.90 → 83.04 (+2.14, 595/250, p=3e-33), unseen +9.4 / rare +7.1 / head +0.7 — the first hws tail gain with no head tax** (contrast #60–62); proposals +0.16 on top (35/9, p=1e-4); the trie out-surfaces the LLM 27% vs 20% (#36's in-lexicon tail again). The futo-val cell confirms cross-corpus is geometry's best case: γ=0.5 carried over unchanged is a **wash in-domain** (94.44 → 94.35 eval, 81/90, p=0.54) with the same structure underneath — unseen +2.6, rare +2.8, head −0.5 — the tail gain persists but now pays #60's head tax; the acoustic-only γ surface peaks at 0.1 in-domain (+0.27) vs 0.5 on hws — fused at γ=0.1: dev +0.21 (p=0.013), eval +0.09 (n.s.), head flat — i.e. the right weight tracks how far off-domain the trained channel is. Remaining caveats: α/β/μ not re-tuned with γ; dwell weighting (#71's +5.3) not yet composed; some damage is wf320k typo entries (`plese`, `destory`). Test untouched | `gen_geom_proposals.py`, `eval_geom_fusion.py`, `runs/geom_fusion_hws.log`, `runs/proposal_arveto_hws.log`, `runs/hyps_geom_fusion_hws.npz` |
| 74 | the control the whole log was missing: training-seed variance. `ar_full` retrained twice with `--seed 1`/`--seed 2` (new flag: seeds init, dropout, data order, and offsets the per-epoch augmentation seed — runs before the flag were unseeded), identical config and budget, identical beam eval | **beam noise floor ≈0.02 in-domain, <0.2 cross-corpus — below the 20k slice's own sampling error (0.12 SE).** val beam top-1 92.40 / 92.38 / 92.38, @8 ceiling 97.8 all three, truth-among-survivors 98.41/98.45/98.41; hws beam 80.92 / 81.08 / 81.01. Greedy is the noisy readout: val 86.7 / 86.4 / 86.9, hws 72.8 / 73.0 / 73.1 (spread ~0.5) — the lexicon absorbs seed noise exactly as it absorbs encoder gains (#17/#18). Consequence: every sub-point *beam* claim in this log clears the training-noise floor — AR over CTC +0.54 (#46), AR-MMI +0.15 (#48), the permutation wash −0.11 (#37), 20 epochs +0.18 (#17) — while greedy-only deltas under ~0.5 should be read as noise. MMI's ±0.05 seed replication (#26) was the only prior check | `runs/ar_full_s1/`, `runs/ar_full_s2/`, `train_ar_decoder.py --seed` |
| 75 | the AR capacity sweep, run in August and never logged (d384's eval had died after loading the lexicon; re-run): d_model 96 / 128 / 192 / 256 / 384 = 0.98M / 1.74M / 3.88M / 6.88M / 15.4M params, 10 epochs each, same data, same beam eval | **16× parameters buys +0.34 at the beam and +3.8 greedy; the ceiling does not move.** val beam top-1 92.21 / 92.40 / 92.46 / 92.50 / 92.55, greedy 85.4 / 86.7 / 87.9 / 88.5 / 89.2, @8 ceiling 97.7 / 97.8 / 97.8 / 97.8 / 97.9, truth-among-survivors 98.39–98.48 flat. hws beam 80.26 / 80.92 / 81.27 / 81.32 / 81.14 — peaks at d256 and falls at d384, the first cross-corpus overfit. Against #74's 0.02 floor the in-domain climb is real but it is #17/#18's absorption curve in the capacity axis: the lexicon converts ~9% of the greedy gain. Beam time scales 241s → 2637s. The acoustic side is not capacity-limited, and with #17 (epochs) it is not budget-limited either; the residual is out-of-lexicon words, misranked in-list candidates, and never-surfaced tail words — none of which more capacity touches. d128 stays the config | `runs/ar_d96/` … `runs/ar_d384/`, `runs/ar_d*_eval.log` |
| 76 | what *is* the residual? — the 20k-val fused-joint errors (gpt2, 94.53) split three ways: out-of-lexicon / in-lexicon-never-surfaced / surfaced-but-misranked, then the misranked bucket cut by the AR score's own preference, by spelling relation, by the truth's training count, and finally a direct separability test: dedicated two-word classifiers (logistic, MLP, 1-NN on positions) trained on FUTO-train gestures of each frequent confusion pair, judged on every val gesture of the pair | **The residual is a training-coverage tail, not a context deficit and not head ambiguity the encoder could learn.** Budget: OOV 1.25 pts, in-lexicon never surfaced 0.40, misranked 3.83 (766 words). The misranked bucket is a long tail — 689 distinct pairs, `had/has` the largest at 15 — and 72% of it is *acoustically* misranked (the AR score alone preferred the rival); only 28% is the prior/LM overriding correct acoustics. By spelling relation: adjacent-key substitution 19%, edit-distance ≥2 to an unrelated word 46% (`batam→bayan`, `ugadi→uday`, `wahroonga→washings`), interior/suffix ±1 letter 16%, doubled letter 7%, spelling variants 3%. By training count the picture is stark: error rate is 1.0% for words with >500 training gestures, 2.8% for 51–500, 7.7% for 6–50, 17% for 1–5 and 49% for 0 — **words with ≤5 training gestures are 10% of val and 55% of errors** (36% of errors are zero-count words, of which 249 are OOV and 148 are in wf320k but never swiped in train). The frequent-pair probe closes the head: on `had/has`, `in/on`, `this/thus`, `a/as`, `country/county` and 20 more pairs, dedicated classifiers separate the *population* at 0.80–0.99 balanced accuracy but never beat the stack's own accuracy on the same instances, and on the specific misranked gestures they recover 32–38% — *below chance* — i.e. those gestures genuinely resemble the rival (or the donor swiped the rival). So: on every recurring head pair tested (77 of the head's ~240 misranked words) the error is irreducible from the gesture alone — the rest of the head is inferred, not measured — ~1.25 is vocabulary, and ~3.0 is rare in-lexicon words the encoder has ≤50 gestures for — acoustic, data-limited *per word*, the #36 tail again, now sized as the whole remaining lever. #60/#62 are the only attacks on it so far (tail +2.6–2.8 in-domain, head −0.5, net wash); the head-tax mechanism is the open problem, not the target | `probe_pair_separability.py`, `runs/probe_pair_*.log`, `runs/probe_error_budget.log` |
| 77 | can the tail fine-tune's head tax be refunded from outside? — #76 said the head is irreducible, so #60's tax could only be a flattened implicit prior; if so an external prior should cancel it. First-pass sweep of `ar + β·len + α·uni − λ·ilm` over the saved deep lists of base vs `ar_tail_ft` (40 cells, val + hws, bucketed by the truth's real training count), flip analysis of the words the fine-tune loses, then two new arms — the same fine-tune with the *full* 917k real corpus as replay (`ar_tail_ft_fullreplay`), and the control #60 never had: six more epochs on real FUTO alone at the same restarted LR (`ar_full_cont6`) — and finally a count-gated two-expert decode over the union of both models' lists | **No — and the "head tax" was misnamed. At matched best prior the 500+ bucket is flat (−0.02 val / −0.4 hws); the loss lives in words with 6–500 real gestures (val −2.1/−1.2, hws −2.8/−2.7), and no (α, λ) cell moves it.** Flip analysis: on the mid-bucket words base gets right and tail-ft loses, the *truth's own AR score* fell by a median 2.4 nats in 99% of cases; only a third of winning rivals are synthetic-only words. That is forgetting, and the recipe explains it: `futo/train:200000` replays the *first* 200k cache rows — 25% of donors, median 3 replay gestures per mid-frequency word, 860 with none. Full replay refunds two-thirds of it in-domain (val mid buckets −0.74/−0.25, overall −0.17 vs −0.60) where the continued-training control is a perfect null (−0.01, every bucket within 0.3) — so the residual −0.7 at 6–50 is the genuine price of 269k new competitors. Cross-corpus the accounting differs: **continuing to train on FUTO at all costs hws −0.42 (51–500 −1.2)** even with no synthetic data, full replay adds another −0.5 on the mid buckets, and it dilutes the tail gain from +9.6 to +2.7 on zero-count words (synthetic share 73% → 37%). No fine-tuned arm beats base in aggregate on either corpus. The gated two-expert decode (base scores words with ≥K real gestures, tail model the rest, δ and K on even sentences, read on odd) is the tax-free ceiling: **hws +0.27 (95/68, p=0.04), val +0.02 (n.s.)** — the zero-count bucket takes +3.9 of the +10 available; the rest is lost to the two models' score-scale mismatch. Where the thread lands: synthetic tail coverage is worth ~+0.25 cross-corpus however it is harvested (#62b's dosed fine-tune +0.23, this +0.27) and ~0 in-domain, because FUTO's zero-count words are out-of-lexicon (#36) and its in-lexicon tail is already 82% right. The lever is real, small, and now bounded from above. Side result: λ=0.25 ilm-subtraction lifts the *base* first pass +0.9 on hws (80.7 → 81.6) and +0.1 val — the prior-algebra term was only ever tuned in-search on val (#49); worth carrying into the hws fused config | `sweep_prior_algebra.py`, `gated_tail_expert.py`, `runs/sweep_prior_*.log`, `runs/gated_tail_*.log`, `runs/ar_tail_ft_fullreplay/`, `runs/ar_full_cont6/` |
| 78 | #77's side result taken through the full stack: internal-LM subtraction (`−λ·ilm[mean]`, #49's prior-algebra term, tuned only in-search on val and left at λ=0 in every deployed config) combined with a heavier external unigram, fused joint decode (gpt2, delta, marginal, μ=0.8, M=24), paired McNemar on the same deep lists. Cells picked from the first-pass sweep on the 20k hws slice, then read on a *disjoint* 20k hws slice (words 20,000–39,998 of the test split, straddling sentence dropped) and on val | **+0.63 hws joint on the tuning slice (81.25 → 81.88, 313/187, p=2e-8) and +0.61 on the held-out slice (80.66 → 81.27, 305/183, p=4e-8) — replicated to the second decimal; val −0.05 (86/96, p=0.5).** The stack kept ~70% of the first pass's +0.9 (vs the usual third), and the gain is broad: held-out buckets 0 / 1–5 / 6–50 / 51–500 / 500+ move +2.2 / +1.9 / 0.0 / +0.9 / −0.2 — the gain is the tail and mid buckets, the head untouched. λ alone at α=0.4 is +0.19 (p=0.14): the two terms are one operation — subtract the encoder's memorized P_train, then lean harder on the corpus unigram — and neither half pays alone (α=0.6 by itself is −0.3 on the first pass). Reading: this is the cheapest measurement yet of how much of the cross-corpus gap is the encoder's implicit prior being wrong for the new corpus — ~0.6 of it removable at inference for free, consistent with #37's permutation result (+0.25, partial removal at training cost) and #77's zero-count sweep (λ recovers half of what synthetic gestures bought unseen words). In-domain it is a null because the memorized prior *is* the test prior. The `run_fused_local.py` prior now composes both terms (before this, λ>0 *replaced* the unigram; no frozen number used λ>0). Not yet done: the full-85k hws read, which is what the headline cross-corpus row quotes; the test freeze is untouched | `runs/fused_lam_hws*.log`, `runs/hyps_lam_*.npz`, `fused_base_hws_heldout.pkl` |
| 79 | data quality, prompted by #76's below-chance head confusions: hygiene stats on futo train/val and hws (duration, points, start/end-key distance in key radii, monotonic time, coordinate range, duplicates, per-session accuracy), error rate vs gesture quality, and a label-free check — the training-free GestureDP cost (#71, dwell-weighted) of the *label* vs the *decoded word* on every val error, calibrated against correct decodes' runner-ups; 16 confusion gestures plotted over the keyboard | **The gestures are clean; the labels are not quite.** Hygiene: 0 flagged rows survive, 0 non-monotonic timestamps, 0 exact duplicates in futo (39 in 917k train), 90% of val swipes start within 1 key radius of the first letter and 94% end within 2 of the last; sloppy gestures (start or end >2 radii off) are 6.7% of swipes and 17.6% of errors at a 14% error rate vs 4.7% for the rest — a real but minor factor. No bad-donor effect: among 162 sessions with ≥30 swipes the worst is 86.7% and sessions under 80% hold 2.7% of errors. Fast swipes are the *most* accurate (200–400 ms: 1.5% error; >4 s: 13.5%). Label noise: on misranked errors the decoded word fits the raw geometry better than the label in 42% (59% when the label is a head word) vs 10% for correct decodes' runner-ups; by a margin >5 cost units, 16% vs 2%. Over all 1,072 val errors the excess over baseline is ~123 words (>5) / ~37 (>15) — **0.2–0.6 pts of the 5.4-pt error rate is the donor swiping a different word than the recorded label**, and the plots show it plainly: `had` gestures that end on *s*, an `a` that drags to *s*, `in`/`on` starting on the i/o seam. The strongest cases are alignment slips, not near-misses — `a→the` (gap 485), `a→part`, `a→day`, `of→fred`, `i→india`, `wal→walmart`, `economic→econo`: the FUTO record's `word` is the prompt's target and the gesture sequence is matched to it in order, so a skipped or merged word shifts the labels of the rest of the sentence. HWS: 245 exact-duplicate groups (279 extra copies) in the 20k slice, all same-session and 195 consecutive — the logger double-wrote gestures; harmless for accuracy, but paired statistics on hws count those words twice. Consequence for #76's budget: the 1.2-pt "irreducible head" bucket is at least a third label noise and the rest genuine ambiguity, and the *clean-label* ceiling of the stack is ~0.2–0.6 higher than any number in this log; no method can recover it, and no test-set number should be read to better than that | `runs/data_quality_*.log`, `runs/data_quality_confusions.png` |
| 80 | #79's label check run on How We Swipe (20k slice, same fused baseline, same GestureDP label-vs-decoded criterion, same runner-up calibration), plus the classes it exposed: labels outside the English lexicon, labels that wordfreq scores as es/de/fr/pt/it/nl, aborted gestures (path under half a key per letter transition), and the loader's retry mechanism tested directly | **A quarter to a third of the cross-corpus gap is HWS label quality, not gesture difficulty.** Decoded-fits-better-than-label: 23% of hws errors by >5 (15% by >15) vs 1.9% / 0.7% baseline — **~2.7–3.9 pts of the 18.75-pt error rate**, five to seven times FUTO's 0.2–0.6. Identifiable garbage is 2.8% of swipes and 13% of errors: 1.65% of labels are not English words at all (100% error), 0.9% are German/Spanish/French (`sollte`, `unterkunft`, `desarrollador`, `sinopsis`; 61% error), 0.6% are aborted gestures — a single tap labelled `acquisitions` (100% error). Two sessions are >30% non-English and one is 63% taps. Dropping the identifiable classes lifts hws 81.25 → 83.22 (FUTO's equivalent filter: 94.64 → 95.92, its non-English 1.2% being the OOV proper nouns). Beyond those, another 430 errors (2.15 pts) by the strict margin are gestures that trace a *different English word* than the prompt — `convenience→convience`, `definitely→definatley`, `advertisements→adverts`, `training→traini`, `holiday→hi`: users in a timed test swiping misspellings, truncations and near-misses, recorded under the prompt. Not retries: only 1.1% of swipes are non-final attempts and they score *higher* (85.3 vs 81.2) — the release's per-attempt flags already removed the failures. Re-accounting the ~13.4-pt val→hws gap: ~2.0 identifiable garbage, ~2–3.5 further label mismatch, ~0.6 the encoder's FUTO prior (#78), ~2.6 recoverable with in-domain data (#8), leaving ~4–5 of genuine apparatus/user difficulty — and #8's "5.5 intrinsic by matched-size training" was measured against these labels, so it absorbed the mismatch. The 279 exact-duplicate gestures are logger double-writes (all same-session, 195 consecutive). Standing: hws is a pessimistic stress test in two senses now — harder users *and* noisier labels — and any hws number is bounded ~3 pts below the stack's clean-label accuracy there | `runs/data_quality_hws_*.log` |
| 81 | #8 re-read after #80: the two matched-size checkpoints (`runs/scratch_hws`, `runs/scratch_futo`, 60k swipes each, from scratch) re-scored greedy on the *full* held-out sets (#8 read 20 batches ≈ 5k) with and without a decoder-independent label filter applied identically to both corpora — label outside the English lexicon, wordfreq-foreign, aborted gesture, or label geometry cost/letter > 6 | **#8's "5.5 pts intrinsic" is 1.8.** Full held-out sets: HWS-in-domain 63.71 (n=25,700; #8's 61.8 was the 5k subset) vs FUTO-in-domain 67.28 — gap 3.6, not 5.5; a third of the quoted number was eval-subset noise. The filter drops 5.9% of HWS held-out swipes (410 non-English, 145 foreign, 55 aborted, 898 untraced) vs 2.9% of FUTO val (249 OOV names, 67, 0, 268) — same rule, twice the yield, which is #80 measured from the label side alone. Clean: HWS 66.51 vs FUTO 68.30 — **gap 1.8**. So of the 5.5 the log quoted as corpus difficulty, ~1.9 was measurement, ~1.8 was label quality, ~1.8 remains — and that remainder is the same order as the user-heterogeneity and bad-user "mild" findings #8 left standing. Cross-corpus (scratch_futo on HWS) 54.14 → 56.83 clean. The label filter is in `rerun_matched_size_clean.py`; the same rule can produce a clean-label HWS read for any model | `runs/rerun_matched_size_clean.log` |
| 82 | two training-data cells on the same seed as `ar_full_s1`, identical recipe: (a) **clean** — `futo_clean/train`, FUTO train minus #81's decoder-independent label filter (1.59% dropped: 11,698 untraced, 2,909 foreign, 3 other); (b) **mixed** — clean FUTO + `hws_clean/train`, the 70%-user half of How We Swipe filtered the same way (56,272 swipes, 5.8% dropped). Val bucketed and McNemar-paired at a common cell; HWS read on the 25,708 swipes of the 30% held-out users that no arm trained on (`hws_heldout/test`, new cache) | **Label cleaning is a null in-domain and +1.0 cross-corpus, the largest training-side cross-corpus gain in the log; 56k real HWS gestures add +2.9 on top for HWS and nothing for FUTO's tail.** Val: s1 92.38 / clean 92.39 / mixed 92.39 beam; paired at α=0.6 λ=0.25 no bucket differs (clean −0.12 overall p=0.32; mixed vs clean on the 6–50 bucket HWS covers, +0.65 p=0.13) — the per-arm-best table's tail→head shift was cell choice, not model. HWS 20k slice, clean vs s1 at the same cell: **+1.01 (623/420, p=4e-10)**, buckets 0 / 1–5 / 6–50 / 51–500 / 500+ = +2.9 / +2.7 / +0.2 / +1.4 / −0.05; +1.27 at the default cell; stacks on #78's λ. Held-out users: s1 80.69 → clean 81.54 (+0.85) → mixed **84.39** (+2.85 more; truth-in-list 95.2 → 96.0). Reading: the 1.6% of training gestures that do not trace their label teach associations FUTO val shares (its labels carry the same noise — hence the in-domain null) and HWS does not; removing them is free cross-corpus robustness. The filter's yield (1.6%) exceeds #79's mislabel estimate (0.2–0.6%), so it also drops the sloppiest correctly-labelled gestures; which of the two carries the gain is untested (threshold sweep). Mixed answers the real-gesture version of #77's existence-vs-discrimination question for FUTO: 63% of val's 6–50-bucket swipes gained ≥1 HWS gesture and the bucket moved +0.65 n.s. — a handful of off-apparatus real examples does not sharpen words the encoder already knows, only teaches unseen ones. For HWS itself, in-domain data is worth +2.9 over the clean FUTO model (#8 measured +2.1 from a CTC fine-tune), and #78's λ and this cleaning both carry over. **Seed-2 replication holds**: clean vs base at seed 2 is val 92.41 vs 92.38, hws 20k 81.66 vs 81.01 (+0.65), held-out users 81.40 vs 80.51 (+0.89) — against seed 1's +1.00 / +0.85. Two seeds, two hws slices, four reads, all +0.65 to +1.0 with in-domain flat; the between-seed spread of the *effect* (~0.2) is the number to quote as its error | `build_clean_caches.py`, `runs/ar_clean_s1/`, `runs/ar_mixed_s1/`, `runs/sweep_clean_*_paired.log`, `runs/*/beam_eval_hws_heldout.log`, `data/canonical/{futo_clean,hws_clean,hws_heldout}/` |

Standing conclusions the log supports:

- Top-1 is bounded by the lexicon and the n-best ceiling, not the encoder
  (#4, #17, #18).
- Second-pass signals must add *evidence the first pass lacks*, never re-apply
  a prior it already has (#10, #11). Scaling the LM does not create such
  evidence *as a second pass*: #11's ceiling holds through 9B and two model
  generations (#34) — what pays is new context (#30, #33), not a bigger
  reader of the old context. But the scope is the pass, not the model: moved
  *into* the search, where it steers which hypotheses survive, the same
  gpt2→xl swap buys +0.43 (#51), and the invariance gap it cannot touch
  (#50–51) marks the boundary between context-soluble and context-free
  errors. The full ladder in that role (#66) says what the scale variable
  actually is: **authority**. gpt2→xl is worth +0.24 (n.s.) when the LM may
  only rescore the current word, +0.42 with one word of lookahead and +0.56
  when it may prune the whole sentence — the same swap more than doubling as
  the LM gains the power to decide, and only resolvable once it can revise.
  It *is* a route past gpt2-xl, contrary to what this log said at first:
  Qwen3.5-9B leads it by 0.32 (p=0.037) and the GPT-2 family saturates at
  774M. The correction that produced that reversal is the more portable
  finding — `delta`'s subtracted prior was estimated per model by a proxy
  that suits GPT-2 and not Qwen, which cost the modern rungs 1–2.7 points
  each and fabricated three plausible conclusions (#66). #34's re-run with
  the fix (#72) lifts every rung — peak conversion 40% of headroom, the
  modern-family deficit dissolves, and even GPT-2 gains in the rescoring
  role — while leaving the scale ordering flat as a second pass, so the
  authority reading stands with cleaner numbers. The scope has a far edge, now
  measured too: moved all the way
  *into the decoder's weights* (#52), the LM degenerates to what the swipe
  corpus's text can teach — +0.7 of left context, zero right context, a
  memorized prompt prior — while taxing the acoustic stream it shares
  parameters with. The LM belongs in the search but not in the model;
  separation is a measured optimum, not a compromise.
- The tuned surfaces are flat, which is why a dozen fits on one validation
  slice did not overfit (#15, #19).
- The encoder is too overconfident to ensemble (#16), but the
  overconfidence is CTC peakiness, not the implicit LM (#39), and it is
  behaviorally harmless to the search: softening the emissions buys no
  ceiling and costs top-1, and the words the search never surfaces lose on
  merit, by ~8 nats, not to write-offs (#41). The open problem is
  decision-level confidence — a post-hoc calibrated "is the top-1 right?"
  signal feeding commit gating (#33) and lexicon escape — not emissions
  calibration. The escape detector is built and validated (#42, AUC 0.97):
  it pays as a raw-spelling suggestion and add-to-dictionary trigger, not
  as silent replacement.
- As of #20–24 the decode side is closed: width, pruning, scoring family,
  lexicon size, list depth, and both second-pass weights are all measured,
  and every one either sits at a flat optimum or is absorbed on the way up.
  The second pass converts ~33% of whatever headroom it is given, regardless
  of where the headroom comes from (#12, #21, #24) — so feeding it more list
  is not a lever, and the only remaining one is making the *first pass*
  right more often: the encoder, whose known defects (OOV words, doubled
  letters, overconfidence) are structural to CTC+trie rather than tunable.
- The first encoder lever tried confirmed this reading: MMI fine-tuning over
  the beam's own n-best lists (#26) is the one change all session that moved
  every stage at once — first pass, ceiling, and full stack — because it
  improves what the encoder says rather than how the pipeline reads it. It
  also replicates #25's diagnosis in our own weights: emissions trained for
  the search stop being greedy-readable, and that trade is worth points.
- The OOV defect is the encoder's implicit LM, now measured (#36) and
  relocatable (#37): permutation-mixture training moves the frequency prior
  into the lexicon at no in-domain cost, and the same change closes the
  layout-geometry gap (#38) — the prior was qwerty-conditional, not just
  frequency-conditional. Priors belong in the swappable components; the
  encoder should be as close to a pure acoustic scorer as the explicit
  stack can compensate for.

## Layout

```
src/swipe_typing/
  layout.py      canonical grid, KeyboardLayout, FUTO layout-JSON loader
  schema.py      Swipe record + Arrow schema + plausibility filter
  features.py    resampling, 8-channel encoder, key affinity, kinematics
  augment.py     co-augmentation of trajectory and layout
  cache.py       canonical Parquet read/write
  sources/       per-corpus loaders (futo, how_we_swipe)
  model/
    encoder.py   dilated TCN + CTC loss + input normalization
    data.py      torch dataset over the Parquet cache
    decode.py    greedy CTC decode, CER / word accuracy, alignment ops
    lexicon.py   prefix trie + unigram prior
    beam.py      trie-constrained CTC prefix beam search
scripts/
  fetch_how_we_swipe.py   download from OSF
  build_cache.py          normalize every source into data/canonical
  calibrate_layout.py     recover keyboard geometry from touch data
  validate_alignment.py   check the corpora share one coordinate space
  train_encoder.py        train the encoder
  eval_layout_transfer.py score it on layouts it never saw (greedy, no lexicon)
  eval_layout_beam.py     same layouts through the trie beam — the deployed view
  train_mmi.py            discriminative fine-tune over the beam's own n-best
  error_analysis.py       where the errors are, and what would fix them
  eval_decoder.py         greedy vs trie-constrained beam search
  eval_deferred_commit.py hold n-best open across swipes; does right context pay?
  nbest_freq_buckets.py   bucket n-best misses by the target's training count
  run_fused_local.py      one fused sentence-beam config, LM scoring in-search
  compare_hyps.py         paired (McNemar) comparison of two runs' hypotheses
  probe_pair_separability.py  are misranked in-list errors separable from the gesture?
  rerun_matched_size_clean.py  #8's matched-size models re-read on label-filtered eval sets
  build_clean_caches.py   write futo_clean/train and hws_clean/train (the #81 label filter)
  sweep_prior_algebra.py  first-pass α/λ prior sweep over saved deep lists, bucketed by training count
  gated_tail_expert.py    count-gated two-expert decode: base for known words, tail model for rare
  probe_lm_fit.py         does the best-ranking LM also model the eval text best?
  diagnose_transfer.py    attribute the cross-corpus gap to subgroups
  finetune_transfer.py    does in-domain data close it? (user-disjoint split)
```

## Notes

- `data/` is gitignored; nothing is redistributed here.
- Filters drop ~2.3% of FUTO validation and ~6% of How We Swipe (implausible
  coordinates, zero duration, unsegmentable gestures).
- FUTO's *model weights* are under a restrictive license even though the data is
  MIT. Train your own.
