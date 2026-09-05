# Glyph

An open-source swipe keyboard for iPhone that reads your finger better than
the one built into the phone — and runs entirely on it.

**Try it:** [TestFlight](https://testflight.apple.com/join/ZAXsVCWz) (iOS 17+;
listed as "Glyph Swipe Keyboard").

Glyph decodes a swipe with a small transformer (1.7M parameters) trained on
public swipe corpora, a 300k-word trie and a sentence language model
(distilgpt2), all on the phone's CPU, in about 150 ms per word. The keyboard
has no network access and never asks for Full Access.

## How it scores

The same recorded finger paths, replayed byte-for-byte onto each keyboard on an
iPhone 17, scored on the word each keyboard commits (top-1, no corrections),
paired word by word:

| keyboard | real iPhone swipes (542 words) | FUTO corpus (1,337 words) |
|---|---|---|
| **Glyph** | **77.9%** | **93.4%** |
| QuickPath (Apple) | 74.9% | 90.2% |
| SwiftKey (Microsoft) | 69.0% | 85.9% |
| Gboard (Google) | 68.5% | 88.0% |

Against QuickPath: +3.0 on the real swipes (p = 0.09), +6.9 on everyday words
(p = 0.001), level on rare words and on the first word of a sentence; +3.2 on
FUTO (p < 0.001). Against Gboard and SwiftKey: p < 0.001 everywhere. Glyph's
first pass alone, with the language model switched off, is level with
QuickPath and ahead of the other two.

The honest caveat: the real-swipe set is one person's 542 words, fast and
sloppy (twice the speed of the corpus donors). Whether the margin holds for
other people is exactly what the app's game is collecting. Method, statistics
and every intermediate result: [`research/iphone/README.md`](research/iphone/README.md).

## What is in the repo

```
keyboard/    the iOS app and keyboard extension (Swift, Core ML, xcodegen)
             App/           onboarding, practice mode, the home screen
             Extension/     the keyboard (UIInputViewController + views)
             Shared/        letter grid, native metrics, decoder loader (both targets)
             GlyphCore/     Swift package: features, trie, AR beam, CTC beam,
                            fused sentence search, geometric trace test
             UITests/       drives the real keyboard; the replay benchmark
             tools/         export models from research checkpoints, replay
                            benchmark, layout measurement, model fetch
             upload-worker/ Cloudflare Worker + R2 endpoint the game uploads to
research/    training pipelines, evaluation harnesses, the lab notebook
             README.md      every experiment, numbered, with what it showed
             iphone/        the real-iPhone data, replay benchmark scoring,
                            game-record tools
```

The decoder resources (two Core ML models, the language model, trie and
tables, ~180 MB) are not in git. `keyboard/tools/fetch_models.py` downloads
them from Hugging Face into `keyboard/Resources/`; the export scripts
regenerate them from the research checkpoints.

## Build and run

Requirements: Xcode 26, [xcodegen](https://github.com/yonaskolb/XcodeGen),
Python 3.12 with `uv` (for the model fetch and the research tools).

```
cd research && uv venv && uv pip install -r requirements.lock.txt && cd ..
research/.venv/bin/python keyboard/tools/fetch_models.py     # models -> keyboard/Resources
cd keyboard && ./deploy.sh                                    # build, sign, install on a connected iPhone
```

`deploy.sh` needs an Apple ID in Xcode (a free personal team works; builds
then expire after seven days). On the phone: Settings › General › Keyboard ›
Keyboards › Add New Keyboard › Glyph. The app walks through this after a
three-sentence practice run.

Tests: `cd keyboard/GlyphCore && swift test` checks the Swift port against
Python goldens (features, beams, sentence search, trace cost); the UI tests
in `keyboard/UITests` drive the real keyboard in the simulator.

## Practice and the data

The app opens with practice: swipe the words of a sentence, timed. A
word counts when the finger traced it — a geometric test, not the decoder's
opinion, so the recorded labels are never biased toward what the model already
gets right. Every attempt is recorded with the prompted word (finger path,
timing, what the decoder read) under a random per-install id, and uploaded to
an endpoint in `keyboard/upload-worker`. Practicing is consent; the intro screen
says so in plain words. Nothing typed on the keyboard is ever recorded.

Prompts are real modern text (tweets, reddit, WildChat) chosen for word
coverage (`research/scripts/build_race_prompts.py`). The records feed the
per-player benchmark (`research/iphone/race_to_capture.py`) and, in time,
training.

## Research

`research/README.md` is a lab notebook of ~85 numbered experiments: encoder
architectures, data cleaning, the fused language-model search, cross-corpus
generalization, and why the model that ships is the one it is. Start with
"Which encoder generalizes to a real iPhone?" near the end.

## License

MIT. Model weights and data files in this repository and on Hugging Face are
under the same terms. The training corpora (FUTO, How We Swipe) have their own
licenses.
