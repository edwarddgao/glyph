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
| **+ deferred commitment, joint (full stack)** | **94.62%** | — |
| n-best@8 ceiling | 97.50% | — |

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

The test split has now been read three times, once per frozen configuration:
the original stack (93.92, #19), the MMI fine-tune (94.27, #28), and deferred
commitment (94.62, #35). Every time, every stage landed within ~1–2 SE of its
validation estimate, on the positive side.

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

**Distribution match beats capability.** The Qwen column is the sharper
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
side, How We Swipe is **5.5 points harder**. That is intrinsic corpus
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
| 8 | cross-corpus gap, 8 hypotheses | 7 refuted; ~5.5 pts intrinsic by matched-size training | `diagnose_transfer.py`, `finetune_transfer.py` |
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

Standing conclusions the log supports:

- Top-1 is bounded by the lexicon and the n-best ceiling, not the encoder
  (#4, #17, #18).
- Second-pass signals must add *evidence the first pass lacks*, never re-apply
  a prior it already has (#10, #11). Scaling the LM does not create such
  evidence: #11's ceiling holds through 9B and two model generations (#34) —
  what pays is new context (#30, #33), not a bigger reader of the old context.
- The tuned surfaces are flat, which is why a dozen fits on one validation
  slice did not overfit (#15, #19).
- The encoder is too overconfident to ensemble (#16) — calibration, not
  capacity, is its open problem.
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
  diagnose_transfer.py    attribute the cross-corpus gap to subgroups
  finetune_transfer.py    does in-domain data close it? (user-disjoint split)
```

## Notes

- `data/` is gitignored; nothing is redistributed here.
- Filters drop ~2.3% of FUTO validation and ~6% of How We Swipe (implausible
  coordinates, zero duration, unsegmentable gestures).
- FUTO's *model weights* are under a restrictive license even though the data is
  MIT. Train your own.
