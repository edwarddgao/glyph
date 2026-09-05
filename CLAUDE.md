# Glyph

One repo: Glyph, an open-source iPhone swipe keyboard (`keyboard/`), and the
swipe-typing research behind it (`research/`) — a decoder stack (encoder +
trie beam + fused LM sentence search) benchmarked on the public FUTO and How
We Swipe corpora and, by gesture replay, against QuickPath, Gboard and
SwiftKey. Root README.md is the public front page; the benchmark tooling
still calls the keyboard "swipe" in file names and `--keyboard` values.

Model choice for the phone is made on cross-domain evidence, never on
in-domain FUTO validation alone: the held-out real-iPhone gestures in
`research/iphone/data` (never trained on) plus the replay benchmark
against QuickPath. The August-2026 "capture study" pilot and its encoder
`runs/full` are superseded; do not cite the 72.7-vs-72.4 tie.

Dropped direction (2026-09-01): the MacBook-trackpad / no-visible-keyboard
target and its `swipepad` web game are abandoned. A translation- and
scale-invariant decoder loses too much accuracy to be worth it — research
README #43/#44 (shape-only: −10 beam, −4 ceiling), #50 (−3.7 through the
full fused stack), #54 (any partial invariance is monotone destruction).
Do not propose trackpad work or invariance experiments; the keyboard must
have a fixed, known geometry.

- `research/` — research. Training pipelines, eval harnesses, model
  checkpoints under `runs/`. README.md is the lab notebook; commits are
  narrative findings. Venv: `uv venv && uv pip install -r
  requirements.lock.txt` (Python 3.12).
- `keyboard/` — the Glyph app + keyboard extension (Swift, Core ML,
  xcodegen; bundle ids `com.edwardgao.glyph[.keyboard]`). Ships the AR
  decoder `research/runs/ar_mixed_s1` (two Core ML models + a Swift
  trie-constrained AR beam), the train+wf320k trie, and the distilgpt2
  fused sentence search, fully on-device; no Full Access, no networking in
  the extension. Resources are not in git: `tools/fetch_models.py` pulls
  them from Hugging Face (`edwarddgao/glyph-models`), `tools/export_*.py`
  regenerate them from checkpoints, `tools/upload_models.py` publishes;
  they ship once in the app bundle and the extension reads them from its
  containing app. `GlyphCore` tests assert agreement with the Python
  featurizer/beams/trace cost. `./deploy.sh` installs to a connected iPhone
  (Apple ID in Xcode; injects the upload token). The app opens with
  onboarding: a three-sentence practice run (the former SwipeRacer game),
  then the enable steps; each run records every labeled attempt (kind `race`, acceptance =
  geometric trace, decoder verdict recorded alongside). Uploads go to the
  Cloudflare Worker in `keyboard/upload-worker` (R2 bucket `swipe-races`;
  tokens in `research/iphone/.secrets/`, gitignored);
  `research/iphone/sync_race.py` pulls them down and `race_to_capture.py`
  converts them. Replay benchmark: `tools/replay_bench.py`, results in
  `research/iphone/README.md`.

Replay benchmarks: before launching simulator runs, check headroom
(`top -l 1 | grep -E 'CPU usage|PhysMem'`, `memory_pressure`) and scale
the fan-out to it — an 18-core/64 GB Mac idles at two sims. Shard one
source over several sims with `tools/replay_bench.py --shard i/n` (one
sim per shard); the scorer merges shards by sentence.

Patch hygiene: heredoc/python patches must assert the old text was found
and the new text is present after writing — silent replace no-ops have
shipped phantom fixes more than once.
