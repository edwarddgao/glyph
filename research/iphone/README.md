# iPhone evaluation data and tools

*Naming: the keyboard was renamed Glyph on 2026-09-05; the benchmark files,
`--keyboard` values and tables here still say `swipe` / Swipe for it.*

Everything here is about one question: how the stack behaves on a real
iPhone, judged against Apple's QuickPath on identical input. Nothing in
`data/` is ever trained on.

## Data (`data/`)

- `capture_*.json` — **543 real-iPhone swipes** (96 sentences, 8 phrase sets,
  one user, 2026-08-23), raw touches in canonical coordinates. Recorded on a
  drawn web keyboard, so the apparatus differs slightly from the shipped
  keyboard; still the only real-phone gestures in the repo and the held-out
  test set every encoder/LM decision is judged on. These gestures are 2×
  faster and 2× sloppier than FUTO's donors (30% start or end more than two
  key radii off), which is what a real phone user looks like.
- `capture_*_kb.json` — swipes recorded through the shipped keyboard itself
  (`join_kbcapture.py` output), the same shape.
- `native_*.json` — prompted sentences and what a keyboard committed
  (QuickPath, Gboard, or Swipe), from the app's "Record swipes" screen.
- `kbcapture_*.json`, `kbpick_*.json` — one record per swipe / correction from
  the Swipe keyboard's capture mode.
- `race_*.json` — SwipeRacer records from the app's game: one prompted
  sentence per file with every swipe attempted on each word (canonical
  touches, prompted word, attempt number, whether it traced the word — the
  `GestureTrace` cost, #81's label-filter rule, per letter ≤ 6 — and what
  the shipped stack read: first-pass list, fused choice, `decoder_right`),
  under an anonymous per-install id. Prompts come from the coverage-chosen
  pool `keyboard/Resources/race_prompts.json` (`scripts/build_race_prompts.py`:
  real tweets / reddit / WildChat text, in-lexicon, blocklist-clean, 3
  everyday + 2 tail per race). Current pool: 3,000 sentences (1,200
  everyday, 1,800 tail) from 13,658 candidates, 22,072 word tokens, 8,376
  distinct words (947 head / 4,995 mid / 2,434 tail by zipf band), 548 of
  the 646 letter bigrams in the 50k most frequent lexicon words; "the" is
  the most repeated word at 868, i.e. 4% of tokens against 5–6% in running
  English. `prompt_source` = phraseset marks the eight hand-written sets the
  first races used.
  `race_to_capture.py` turns them into `capture_*_race.json` (one gesture per
  word, the same shape as above) and prints per-player first-swipe accuracy.
- `bench_*.json` — replay benchmark results: the same recorded gestures
  replayed onto each keyboard (`keyboard/tools/replay_bench.py`).

## Tools

- `server.py` — LAN upload endpoint (`POST /save` → `data/`). Run it on the Mac
  while recording: `python iphone/server.py`.
- `sync_race.py` — pulls records uploaded to the production endpoint (the
  Cloudflare Worker in `keyboard/upload-worker`, R2 bucket `swipe-races`) into
  `data/` under the same file names server.py writes; `.secrets/` holds the
  Worker URL and tokens (gitignored). Run it before race_to_capture.py.
- `join_kbcapture.py` — pairs keyboard-captured gestures with the app's
  prompts and writes labeled `capture_*_kb.json`.
- `race_to_capture.py` — SwipeRacer records → `capture_*_race.json` (accepted
  attempt per word, or the last one; `--all-attempts` also writes every
  attempt for training) plus a per-player table: words, first-swipe
  acceptance (= the shipped stack's top-1 on that player), swipes per word,
  skipped, wpm.
- `decode_capture.py` — first-pass decode of the capture gestures with a CTC
  encoder; `fused_rescore.py` — the fused sentence beam reference the Swift
  port (`keyboard/SwipeCore`) is tested against word for word.
- `benchmark_keyboards.py` — scores `bench_*.json` (and human `native_*`
  sessions) per keyboard with paired McNemar, tail/everyday and first-word
  splits, and the smallest detectable difference for the data so far.

## Current results (2026-09-04)

Replay benchmark, iPhone 17 simulator, byte-identical gestures to every
keyboard (details and statistics in the main README, "Which encoder
generalizes to a real iPhone?" and "Is the scoring algebra robust across
registers?"):

| source | words | QuickPath | Gboard (phone) | SwiftKey (phone) | Swipe (AR `ar_mixed_s1` + distilgpt2, shipped) | Swipe, no LM (AR first pass) | previous Swipe (CTC `runs/full` + LM / no LM) |
|---|---|---|---|---|---|---|---|
| this user's iPhone swipes | 542 | 74.9 | 68.5 | 69.0 | **77.9** | 73.6 | 70.3 / 67.2 |
| futo/validation sentences | 1,337 | 90.2 | 88.0 | 85.9 | **93.4** | 91.6 | 92.1 / 90.4 |

SwiftKey (Microsoft, 4.3.5, Flow on, default settings; grid measured from
`data/layout/swiftkey_phone.png`: full width, 40.2 pt pitch, 57.5 pt row
pitch) was replayed the same way as Gboard, on the phone at speed 1.1. Paired:
level with Gboard on the iPhone swipes (69.0 vs 68.5, p=0.84) and behind it on
FUTO (85.9 vs 88.0, p=0.02); behind QuickPath on both (−5.9 p=0.002; −4.3
p<0.001); behind Swipe by 8.9 on the iPhone swipes (25 vs 73, p<0.001;
everyday 78.2 vs 88.9, 3 vs 31) and by 7.6 on FUTO (23 vs 124, p<0.001, first
word −8.0 p=0.004). Swipe's first pass alone beats it too (+4.6 p=0.02;
+5.8 p<0.001).

Gboard (2.3.19, glide typing, default settings) cannot be installed in the
simulator, so its column was replayed on the iPhone 17 itself (speed 1.1,
see calibration) with its letter grid measured from a screenshot
(`data/layout/gboard_phone.png`: full-width 40.2 pt pitch, rows at 54 pt,
so its keys sit ~3.7 pt further apart than Apple's — the same canonical
gesture lands on the same letters). Paired: Gboard is behind QuickPath on
both sets (iPhone swipes −6.5, 40 vs 75, p=0.001, almost all of it on the
tail, −10.7; FUTO −2.2, p=0.01) and behind Swipe by more (iPhone swipes
−9.4, 22 vs 73, p<0.001, everyday −8.8 and tail −10.0 both p<0.001; FUTO
−5.5, 26 vs 99, p<0.001; first word −7.3, p=0.007). Even the AR first pass
alone beats Gboard (+5.2, p=0.008 iPhone; +3.7, p<0.001 FUTO). Gboard's
errors on the shared gestures are a superset of Swipe's — the ambiguous
gestures both miss the same way ("said"→"days", "really"→"tally") and
Gboard adds its own on clean ones ("almost"→"Sonos", "was"→"read").

Device check: QuickPath replayed on the phone itself (speed 1.1) against
the simulator column above, same 542 words — 75.8 vs 74.9, 13 vs 8
discordant, p=0.38; everyday 82.4 vs 82.0, tail 69.8 vs 68.3, first word
72.6 vs 69.5 (3 vs 0). 521 of 542 words came out identical, so simulator
numbers stand for phone numbers to within a point, and against the phone's
own QuickPath Swipe is +2.0 (45 vs 34, p=0.26), Gboard −7.4 (p<0.001).

**How clean must a swipe be for the stack to get it?** The same trace cost,
computed for the 542 iPhone words, against what each keyboard committed in the
replay (first pass = AR alone, fused = AR + LM):

| trace cost / letter | n | first pass | fused | QuickPath | LM rescues / breaks |
|---|---|---|---|---|---|
| < 0.5 | 31 | 87 | **94** | 94 | 2 / 0 |
| 0.5–1 | 85 | 84 | **91** | 87 | 6 / 0 |
| 1–1.5 | 89 | 83 | **89** | 87 | 5 / 0 |
| 1.5–2 | 73 | 79 | **81** | 74 | 3 / 2 |
| 2–3 | 103 | 74 | **77** | 72 | 5 / 2 |
| 3–4 | 67 | 66 | **70** | 61 | 5 / 2 |
| 4–6 | 48 | 60 | **62** | 60 | 2 / 1 |
| 6–10 | 28 | 46 | **50** | 32 | 1 / 0 |
| ≥ 10 | 19 | 21 | 26 | 21 | 1 / 0 |

Accuracy falls monotonically with the cost for every keyboard: a swipe that
traces its word to within one key half-extent per letter is read right 9 times
in 10, one at 2–3 three times in 4, one past 6 (the label filter's cut, 9% of
this user's swipes) half the time or less. The LM helps at every level of
cleanliness and never breaks more than it rescues, but its help is small (+2
to +7) next to the geometry's effect (94 → 26). SwipeRacer's acceptance sits
at 6: what the game calls a traced swipe is a swipe the stack should be
getting, and the ones it does not are the data.

Is that one person's curve? The cost and the cut of 6 come from #81, set on
FUTO training gestures before this user's existed. The same table on the 1,337
FUTO validation words (other people): [0,1) 922 words, first pass 92 / fused
93 / QuickPath 86; [1,2) 290, 89 / 92 / 87; [2,3) 71, 83 / 90 / 79; [3,6) 46,
76 / 83 / 76; ≥6 8 words. Same monotone shape, but the distribution differs
— 69% of FUTO's swipes trace under 1 against 21% of this user's, and 0.6%
pass the cut against 8.7%. Within a bucket this user's swipes look harder
(3–6: 67 vs 83 fused), but that is the words, not the gestures: split by the
prompt set, this user's *everyday* words match or beat FUTO in every bucket
(97 / 95 / 84 vs 93 / 91 / 83 for cost <1 / 1–3 / 3–6) and the *tail* words
(names, slang) sit at 86 / 70 / 50. By unigram frequency inside cost 1–6:
common words 92 vs FUTO's 96, rare words (log p < −12) 25 vs 67. Speed is not
it — within any bucket, faster and slower FUTO swipes score the same, and this
user's slower swipes score no better than the fast ones. So gesture quality
(the trace cost) and word rarity together account for the per-person gap, and
both are known for every game swipe: the cost from the path, the rarity from
the prompt. Nothing about the gesture goes unmeasured into the training set.

**Should the LM bonus be clamped?** The practice records showed the fused
search turning a correct first pass into a surname at sentence starts ("hes"
→ "hess", "about" → "scott"): the marginal prior is averaged over mid-sentence
contexts where names are rare, so a name's sentence-start conditional sits
2.5–3 nats above its marginal. `scripts/eval_lm_clamp.py` scores the fix
candidates on the shipped stack (AR first pass, distilgpt2, lookahead 1) over
both replay sets:

| variant | iPhone words | everyday | tail | first word | FUTO | FUTO first word | first-pass-correct words flipped (iPhone / FUTO) |
|---|---|---|---|---|---|---|---|
| current | **78.1** | 87.7 | **69.1** | 67.7 | **94.1** | **92.7** | 8 / 7 |
| clamp bonus at 3 | 76.8 | 86.6 | 67.7 | 68.8 | 93.3 | 92.7 | 5 / 7 |
| clamp at 2 | 76.1 | 85.1 | 67.7 | 66.7 | 93.3 | 92.7 | 5 / 5 |
| clamp at 1 | 75.5 | 85.1 | 66.7 | 67.7 | 93.1 | 92.0 | 3 / 2 |
| clamp at 0 | 73.5 | 82.8 | 64.9 | 65.6 | 93.1 | 92.0 | 4 / 1 |
| no LM on the first word | 77.9 | **88.1** | 68.4 | **70.8** | **94.1** | 92.0 | 8 / 6 |

Not adopted. The flips are 8 of 543 words (1.5%) and every clamp that removes
some of them loses more elsewhere: the same large positive bonuses that
promote "hess" are what rescue rare correct words in context. Switching the LM
off for the first word is a wash overall (+3.1 on the iPhone first word, −0.7
on FUTO's, n.s. both), consistent with the sentence-initial finding in the
main README. The shipped form stays; the surname flips are a known cost.

Replay timing calibration (`testTimingCalibration`: straight 600 ms swipes
at several event spacings, the Swipe extension reporting what it received).
Simulator: events closer than ~33 ms burst, and a 33 ms path arrives ~1.2×
longer — the replay resamples every gesture to 30 Hz and pre-compresses time
by 1.2. Phone: the same path arrives 1.07–1.11× longer (a ~50 ms fixed
overhead rather than a stretch: 300→358, 600→658, 1200→1275 ms), so the
phone replays at speed 1.1. Bursting below 33 ms holds on both.

Paired (McNemar exact, same words): on the iPhone swipes Swipe is +3.0 over
QuickPath overall (48 vs 32 discordant, p=0.09), **+6.9 on everyday words
(88.9 vs 82.0, 24 vs 6, p=0.001)**, level on the tail (67.6 vs 68.3, p=0.89)
and on the first word (68.4 vs 69.5, 6 vs 7, p=1.0). On FUTO +3.2 overall
(71 vs 28, p<0.001), first word 92.0 vs 88.7 (5 vs 0, p=0.06). The AR first
pass alone (LM off) is level with QuickPath on the iPhone set: 73.6 vs 74.9
overall (43 vs 50, p=0.53), 83.1 vs 82.0 everyday, 64.8 vs 68.3 tail
(p=0.22), 70.5 vs 69.5 first word — the CTC first pass was 67.2, −7.7
(p<0.001). The LM stage then adds +4.2 (30 vs 7, p<0.001), all of it on
later words: on the first word it is −2.1 (2 vs 4, n.s.), the sentence-
initial finding of the main README carried onto the phone. On FUTO the
first pass alone is 91.6 vs QuickPath's 90.2 (62 vs 43, p=0.08) and the LM
adds +1.8 (31 vs 7, p<0.001).
The CTC build it replaces lost to QuickPath by 4.6 on the same gestures
(p=0.012); the whole swing is the encoder, since the trie, LM and scoring
did not change. Offline prediction for this build was 77.0; replay gives
77.9, so the simulator replay path costs nothing. What remains behind
QuickPath is nothing significant; what remains *level* is the tail and the
sentence-initial word, both acoustic (the first word has no context to
rescue it — first-pass first-word accuracy below).

Simulator timing for the AR build (two sims busy, not representative of the
phone): first pass 34 ms, fused search 633 ms per word over 1,856 words.

## History

An August 2026 pilot ran here as "the capture study" (a Safari page with a
drawn keyboard, offline decoding, QuickPath typed live). Its headline — the
composed offline stack tying QuickPath at 72.7 vs 72.4 — compared an
offline decode of one apparatus against live typing on another, and its
encoder choice (`runs/full`) was never tested against the later encoders on
these gestures. Both are superseded by the replay benchmark above. The page,
the composed-stack script and the per-user adaptation scripts were removed
on 2026-09-04; the log entries in the main README (#58, capture study) keep
the record of what was measured then.
