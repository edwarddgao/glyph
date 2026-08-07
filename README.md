# swipe-typing

Normalized loaders, keyboard geometry, and features for the public swipe /
gesture-typing corpora. Every source is mapped into one coordinate space and one
record type, so a model trained on one corpus can be evaluated on another.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

python scripts/fetch_how_we_swipe.py          # ~70MB from OSF, expands to ~920MB
python scripts/build_cache.py                 # normalize everything -> Parquet
python scripts/validate_alignment.py          # confirm the corpora agree
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
the geometry case. Closing the geometry gap most likely means training on mixed
row counts — augmentation currently varies scale and shear but never the number
of rows, so the model has no reason to have learned that invariance.

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

The lexicon is the 65,192-word FUTO training vocabulary with observed counts as
the prior. That imposes a hard ceiling: an evaluation word outside it cannot be
produced, so the ceiling is reported next to every number.

### Results (n = 20,000 per split)

**futo/validation** — lexicon ceiling 95.9%

| decoder | CER | top-1 | top-4 |
|---|---|---|---|
| greedy, no lexicon | 0.070 | 78.4% | — |
| beam 16 | 0.053 | 89.8% | 93.6% |
| beam 32 | 0.049 | **90.2%** | **94.1%** |

**how_we_swipe** — lexicon ceiling 89.7%

| decoder | CER | top-1 | top-4 |
|---|---|---|---|
| greedy, no lexicon | 0.126 | 64.4% | — |
| beam 16 | 0.121 | 75.2% | 82.8% |
| beam 32 | 0.117 | **75.6%** | **83.6%** |

The lexicon is worth **+11.8 points** on futo/validation and **+11.2** on How We
Swipe, confirming the error analysis. Beam 32 reaches 94% of the achievable
ceiling on futo/validation and 84% on How We Swipe. Decoding runs at roughly
1,100 swipes/s single-threaded at beam 32 — the trie constraint prunes hard
enough that this was never the bottleneck I expected.

`alpha`/`beta` were tuned on 2,000 validation swipes, but the entire grid
`alpha ∈ [0, 1.6] × beta ∈ [0, 2.0]` spans under one point of accuracy, so
neither value is load-bearing.

### What is left

Of the ~1,969 remaining errors on futo/validation:

- **41.3% are out-of-vocabulary** — the true word is not in the lexicon, so no
  decoder configuration can reach it. Growing the vocabulary or handling OOV
  spelling is the only route.
- **58.7% are real-word confusions** — `val`→`call`, `has`→`had`, `im`→`in`.
  The gesture genuinely resembles both. A lexicon cannot separate these; only a
  context language model can, which is what FUTO's ContextLM component is for.

Both remaining buckets sit outside the encoder. Further encoder work would buy
little without first adding vocabulary coverage and a context model.

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
  eval_layout_transfer.py score it on layouts it never saw
  error_analysis.py       where the errors are, and what would fix them
  eval_decoder.py         greedy vs trie-constrained beam search
```

## Notes

- `data/` is gitignored; nothing is redistributed here.
- Filters drop ~2.3% of FUTO validation and ~6% of How We Swipe (implausible
  coordinates, zero duration, unsegmentable gestures).
- FUTO's *model weights* are under a restrictive license even though the data is
  MIT. Train your own.
