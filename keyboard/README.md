# Glyph — the app and keyboard extension

The iOS side of the repo: the Glyph keyboard (a custom keyboard extension), the app that hosts it (onboarding, SwipeRacer, the benchmark), and the tools that export models from the research checkpoints and replay recorded gestures onto competing keyboards. The public front page and headline results are in the root README; this file is the engineering detail.


```
keyboard/
  tools/export_ar.py     research checkpoint -> Resources/ (AR models, goldens); export_lm/priors/ilm.py
  Resources/             SwipeAREncoder/SwipeARStep/SwipeLM.mlpackage, lexicon.bin, ilm.bin, priors.bin, goldens
  GlyphCore/             Swift package: Geometry, Features, Trie, AR beam, CTC beam, SentenceSearch, LM, GestureTrace
  Shared/                compiled into app and extension: letter grid, native metrics, DecoderLoader
  Extension/             the keyboard (UIInputViewController + views)
  App/                   onboarding, home screen, practice mode; --bench / --lm-probe developer screens
  UITests/               drives the real extension in the simulator; replay benchmark; race test
  project.yml            xcodegen spec -> Glyph.xcodeproj
  deploy.sh              build + sign + install on a connected iPhone
  release.sh             archive + upload to App Store Connect (TestFlight)
  tools/make_icon.py     renders App/Assets.xcassets (the swipe trail of "glyph")
```

## What ships

The decoder resources (two AR models, the LM, trie and tables, ~180 MB) are
bundled once, in the app; the extension reads them from its containing app's
bundle (`Shared/DecoderLoader.swift`, `SharedResources`), so the app's game
and the keyboard run the identical stack without doubling the install.

- First pass: the **AR swipe decoder** `research/runs/ar_mixed_s1` (#82b:
  clean FUTO + 70% of How We Swipe's users; TCN trunk + 2-layer transformer
  letter decoder, 1.7M params) as two fp32 Core ML models — `SwipeAREncoder`
  (features → 64×128 memory, 5.4 MB) and `SwipeARStep` (memory + K prefixes →
  next-letter log-probs, enumerated prefix lengths 1–25, 2.1 MB) — driving a
  Swift port of the trie-constrained AR beam (`ARFirstPass`, beam 32, exact
  against the Python `ar_beam` on the goldens; score-bounded early exit).
  Ranking: `ar + 0.6·unigram + 1.2·len − 0.25·ilm`, the freeze-five cell,
  with the encoder's internal LM per word precomputed into `ilm.bin`
  (`tools/export_ilm.py`, #78's mean-memory ablation). Chosen on the
  held-out real-iPhone gestures (`research/iphone/data`): first pass 71.8 vs
  the CTC `runs/full`'s 67.8, truth-in-beam 93.6 vs 85.5; with the LM 77.0
  vs 71.6 offline. `tools/export_ar.py` regenerates models and goldens. The
  CTC path (`CTCFirstPass`, `SwipeEncoder.mlpackage`) remains in the package
  behind the `FirstPass` protocol but is no longer bundled.
- Lexicon: `train+wf320k` (FUTO training words blended with wordfreq
  English), 301,508 words, 665,991 trie nodes, 10 MB flattened.
- (CTC beam, when `CTCFirstPass` is used: width 64, prune −13, α 0.8, β 1.2
  — `BeamConfig` defaults from `research/src/swipe_typing/model/beam.py`.)
- Latency: ~8–17 ms/word on an M-series Mac CPU including the Core ML call;
  the iPhone should be comparable.

- Sentence LM: **distilgpt2 (82M) in fp16** (162 MB, memory-mapped, CPU
  compute), driving the research stack's fused sentence beam
  (`SentenceSearch`): delta-form scoring `acoustic + 0.8·(log P(w|ctx) −
  log P(w))`, 8 sentence hypotheses over the first pass's top 8,
  **lookahead-1** commitment (the previous word may be silently revised
  once, the way iOS autocorrect does). The recipe is the one the research
  measured in-search (#66 ladder); distilgpt2 was chosen over gpt2-124M for
  latency (21 vs 36 ms per batch of 16 on the iPhone 17). Measured in-search
  on the ladder's slice (7,539 words): distilgpt2 93.73 / 94.24 / 94.73
  (streaming / lookahead-1 / joint) against gpt2's 93.87 / 94.56 / 94.67 —
  a tie at joint and streaming, but −0.32 at lookahead-1, the policy this
  keyboard ships (paired McNemar p = 0.022; 34% vs 40% of the headroom).
  `tools/export_lm.py` exports the model in a gather form — B=16 sequences ×
  L=32 tokens in, log P of the P=6 requested tokens out — so a call moves a
  few hundred floats, not B·L·50k logits. `tools/export_priors.py`
  precomputes the LM's marginal log P(w) for every lexicon word (the eight
  neutral-context average the delta form subtracts) into `priors.bin`, so
  the phone only scores words in context. GPT-2's byte-level BPE is ported
  in `GPT2Tokenizer`.

  Why fp16 on the CPU and not a quantized model on the Neural Engine: the
  keyboard extension had 150 MB of headroom on the iPhone 17, and a probe in
  the host app (`App/LMProbe.swift`, launch with `--lm-probe`) measured the
  real footprint of each variant per compute unit. With CPU compute Core ML
  leaves the weights memory-mapped (+1 MB for distilgpt2, +7 MB for gpt2);
  the GPU path copies them into the process (+200 MB), which is what
  jetsammed the first attempt (`computeUnits = .all`). Quantization is
  therefore unnecessary: int8/int6 were lossless but int4 tripled the
  per-token error. Measured on the phone: 21 ms per batch of 16 for
  distilgpt2 on CPU, so a swipe's LM work is 20–85 ms.

Not shipped: the geometry channel; MMI on the mixed encoder (measured
09-05: +0.7 first pass on the iPhone set, absorbed by the sentence stage —
research/iphone/README "Pre-ship lever audit"); and
the Neural Engine path (9–12 ms per batch, +29 MB, but a 2–5 s compile on
first load) — worth revisiting if CPU latency shows.

## Fidelity

`GlyphCore` is tested against numbers produced by the research code
(`swift test` in `GlyphCore/`): features match `SwipeDataset.__getitem__`
to float precision on 26 real capture gestures, the beam reproduces
Python's candidate list and scores to 1e-6, the Core ML model matches
PyTorch to 2e-3 in log-prob, and end-to-end top-1 agrees with Python on
26/26 gestures. Re-run `tools/export.py` after changing the checkpoint; it
regenerates the goldens too.

The LM path has the same treatment: `GPT2Tokenizer` matches Hugging Face
on 429 strings; `SentenceSearch` fed the research code's candidate lists
and its (ctx, word) LM table reproduces `fused_rescore.sentence_decode`
word for word on 10 capture sentences at every commitment lag
(`search_goldens.json`); `CoreMLLanguageModel` matches the Python-driven
Core ML model to 1e-6 and the fp32 reference within 0.7 nats
(`lm_goldens.json`, int8 + fp16 noise).

## Build & run

```
brew install xcodegen                       # once
../research/.venv/bin/python tools/export.py  # regenerates Resources/ (~1 min)
(cd GlyphCore && swift test)                  # decoder fidelity, on the Mac
xcodegen generate
./deploy.sh                                   # to a connected iPhone
```

`deploy.sh` needs an Apple ID signed in to Xcode (Settings › Accounts; a
free personal team works, the install then expires after 7 days) and an
iPhone with Developer Mode on, plugged in and trusting the Mac. On the phone
afterwards: Settings › General › Keyboard › Keyboards › Add New Keyboard… ›
Glyph, then hold 🌐 in any app and pick Glyph.

TestFlight: `./release.sh` archives a Release build, exports it for App Store
Connect and uploads it. It needs Xcode signed in to a paid Apple Developer
team (or an App Store Connect API key in `../research/iphone/.secrets/`:
`asc_key.p8`, `asc_key_id`, `asc_issuer_id`), refuses to run without the
upload token, and stamps the build number `yyyymmddHHMM` like `deploy.sh`.
`./release.sh --archive` validates the Release build on a personal team.
App Store Connect record: "Glyph Type" (the bare name was taken),
public TestFlight link https://testflight.apple.com/join/ZAXsVCWz.
The Info.plist declares `ITSAppUsesNonExemptEncryption = false` (HTTPS
only) and the privacy policy the store listing links is served by the
Worker at `/privacy`.

Simulator (no signing needed):

```
xcodebuild -project Glyph.xcodeproj -scheme Glyph \
  -destination 'platform=iOS Simulator,name=iPhone 17' -derivedDataPath build \
  CODE_SIGNING_ALLOWED=NO test -only-testing:GlyphUITests/GlyphUITests
```

The UI test adds the keyboard through the Settings app, switches to it via
the globe menu, taps a letter, drags across keys, picks a suggestion and
backspaces, asserting on the text field after each step.

## Layout: the native keyboard, to the pixel

`Shared/NativeMetrics.swift` holds the system keyboard's geometry as
measured from screenshots of the real thing on an iPhone 17 (402 pt, iOS
26.5): column pitch (W − 2·6.67 + 6)/10, keys (pitch − 6) × 43 on a 54 pt
row pitch, row insets 0 / 0.5 / 1.5 pitches, shift/delete 1.3 pitches,
bottom row 1.25 / 1.25 / 5 / 2.5 pitches, predictive bar with the centre
pill, flat white (#3D3D3D dark) keys on #DEDFE3 (#171717 dark), 25 pt
letters, 19 pt SF Symbol icons. The 123 / #+= layers follow the native
ones too, and the emoji key opens an in-keyboard emoji panel laid out like
the system one (`EmojiPanelView`: 5-row column-major grid on 45.8 × 38.75
pt cells, "Frequently Used" first, category bar with ABC / nine categories /
delete at the native positions). iOS gives third-party keyboards no way to
open the system emoji picker, so the panel is ours; it omits the native
search field, since there is no offline emoji-name data. On iOS versions
that ask the extension for a globe key, that slot is the globe instead.

`tools/measure_layout.py native.png ours.png` finds every key face in two
screenshots, pairs them, and prints the per-key deviation in points plus
the glyph-box deviation; the current build passes at 0.33 pt (one device
pixel) on all 32 keys. The `NativeKeyboardCapture` UI test produces both
screenshots (it switches to the system keyboard, dismisses first-use
sheets, snaps letters and numbers, then switches back).

The canonical grid the decoder sees is unchanged: the 10 column cells of
the top row span x ∈ [0, 1] and the three row cells span y ∈ [0, 1], so a
touch converts to corpus coordinates with one division per axis.

## Benchmarking against QuickPath and Gboard

Results (2026-09-04, simulator replay of byte-identical gestures; paired
statistics in `research/iphone/README.md`): on the user's 542 real-iPhone
words Swipe 77.9% vs QuickPath 74.9% (+3.0, p=0.09; everyday words 88.9 vs
82.0, p=0.001; tail and sentence-initial words level); on 1,337 FUTO
validation words Swipe 93.4% vs QuickPath 90.2% (p<0.001). Gboard and
SwiftKey, replayed on the phone (they cannot run in the simulator), are behind
both: Gboard 68.5% / 88.0%, SwiftKey 69.0% / 85.9% (vs Swipe p<0.001 on each). The earlier CTC build scored 70.3 / 92.1
on the same gestures — the swing is the encoder.
`--keyboard swipe-nolm` replays the first pass alone (LM off).

`tools/replay_bench.py --keyboard quickpath|gboard|swiftkey|swipe|swipe-nolm --source capture|futo`
replays recorded gestures (`Resources/bench_gestures.json`, from
`tools/export_bench_gestures.py`) onto the chosen keyboard through XCTest's
private touch synthesizer (`UITests/TouchSynth.m`) and writes each committed
sentence to `research/iphone/data/bench_*.json`; score with
`research/iphone/benchmark_keyboards.py` (paired McNemar per keyboard pair).
Simulator by default for QuickPath and Swipe; `--device phone` for Gboard,
whose letter grid is measured first with `--measure` (screenshot through the
UI test, `tools/measure_layout.py --grid`) and passed as `--grid`. `--shard
i/n` spreads one source over several simulators; `--tag` keeps a keyboard's
phone run apart from its simulator run in the scorer.
Timing calibration and fidelity checks are described in
`research/iphone/README.md`. The host app's `--bench` screen is the text
field the replay types into; `--lm-probe` is the memory probe.

## Practice (the app's game, formerly SwipeRacer)

`App/RaceGame.swift`, `RaceView.swift`, `SwipePad.swift`, `RaceStore.swift`.
Five prompted sentences per session — 3 everyday + 2 tail, drawn without repeats
per player from `Resources/race_prompts.json`, a pool of real modern text
(tweets, reddit, WildChat) chosen for *word coverage* by
`research/scripts/build_race_prompts.py`: every word in the lexicon, no
blocklisted words, 4–9 words, greedy selection that pays 3× for rare words and
almost nothing for the tenth "the". The everyday half is the unbiased test
set, the tail half buys coverage; each record carries `prompt_id`,
`prompt_source` and the tag. The player swipes each word on the
keyboard's own letter grid embedded in the app (same `NativeMetrics`
geometry, same canonical coordinates as the extension). A word advances when
the finger *traced* it: `GlyphCore/GestureTrace`, a Swift port of the
research `GestureDP.word_cost` (the decoder-independent alignment the
training-label filter #81 uses; exact against Python on 149 goldens), per-letter
cost ≤ 6, plus the "aborted" rule for too-short paths. The prompted word is
known, so no decoder and no language model has a say in whether a swipe
counts; speed pressure cannot degrade the labels. The shipped stack still
decodes every swipe in the background and its reading is recorded
(`decoder_right`), so the data answers "given a gesture this clean, did the
keyboard get it?" without gating the game on it. After two misses the word can
be skipped. Single-letter words ("i", "a") are tapped, as on any keyboard, and
the tap is recorded but never enters the swipe data. Score is typeracer's wpm
(characters/5 per minute) plus first-swipe trace rate. Every attempt is recorded with the prompted word (canonical
touches, trace cost, first-pass list, fused choice, decoder verdict) and one
record per sentence (kind `race`) is queued on disk and posted, with a bearer
token, to the upload endpoint: by default the Cloudflare Worker in
`upload-worker/` (`https://swipe-upload.swipe-edwardgao.workers.dev/save`,
records into the R2 bucket `swipe-races`; `research/iphone/sync_race.py`
pulls them down), or the LAN capture server `research/iphone/server.py`
(a field in the info sheet that only the `--debug` launch argument shows). Pending files retry on the next race. The
Worker checks the token, caps records at 2 MB and 60 per minute per IP, and
serves `/list` and `/obj/<key>` to an admin token; tokens live in
`research/iphone/.secrets/` (gitignored) and reach the app only through
`App/Info.plist`, which xcodegen writes at build time and git ignores.
To rotate: `cd upload-worker && npx wrangler secret put UPLOAD_TOKEN`, put
the same value in `.secrets/upload_token`, then deploy and release again —
every installed build carries the old token and stops uploading.
Transport security allows only HTTPS plus local networking. Anonymous per-install id, no other
text is ever recorded. `research/iphone/race_to_capture.py` converts the
records for the research tools and prints the per-player table. Launch
arguments `--race` (open the game) and `--race-set N` (fixed sentences, in
order) serve the UI test `RaceUITests`.

## Keyboard behaviour

- Swipe on the letter grid → top-1 word + space inserted; the bar shows the
  top 3, tapping one replaces the word. A tap (short, still) types a letter.
- Shift: tap toggles, double-tap locks. Auto-capitalizes at sentence start.
- Double space after a swiped word → ". " (iOS convention). Punctuation
  typed right after a swiped word eats the auto-space.
- Backspace right after a swipe removes the whole word and its space, as
  Gboard and QuickPath do; after that it deletes by character. Held, it
  repeats by character and after about a second steps up to whole words,
  like the system keyboard.
- Row 3 side insets are shift and delete (delete repeats on hold), exactly
  the canonical grid's 0.15 inset, so the letter geometry the decoder sees
  is the corpus geometry.
- 123 / #+= toggle the native tap-only symbol layers. Holding 123 for two
  seconds switches the sentence LM off (the benchmark's `swipe-nolm` row); the
  bar then reads "sentence model off · hold 123 to turn on" until it is held
  again. Before the first swipe the bar says only "swipe a word"; memory and LM
  load state are the label's accessibility value, which the benchmark reads.
- Every key plays the system input click (no Full Access needed; haptics would
  need it, so there are none).
