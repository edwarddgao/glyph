# iPhone capture study — first incumbent head-to-head

2026-08-23. One user, 12 sentences (6 everyday, 6 tail-heavy: names,
slang, doubled letters), 72 words, captured in Safari on iPhone (iOS 18.7)
via `index.html` + `server.py`. Same sentences swiped in two conditions:
our canonical drawn keyboard (raw trajectories, decoded offline here) and
Apple QuickPath (its full shipped stack, committed text logged, no
corrections allowed). Data in `data/`, one JSON per sentence, latest
upload per sentence wins.

## Results (same user, same 72 words)

| stack | overall | everyday | tail |
|---|---|---|---|
| ours, trie beam only (no context) | 81.9% | 86.5% | 77.1% |
| ours, + gpt2-xl fused joint (`fused_rescore.py`) | 88.9% | 97.3% | 80.0% |
| ours, composed, fused streaming (`composed_stack.py`) | 86.1% | 95% | 77% |
| ours, composed, fused **lookahead-1** | **88.9%** | 95% | 83% |
| ours, composed, fused **joint** | **90.3%** | 95% | 86% |
| ours, composed, **gpt2-124M**, lookahead-1 | **90.3%** | 95% | 86% |
| ours, composed, **gpt2-124M**, joint | **90.3%** | 95% | 86% |
| **QuickPath** (full stack, on-device) | **86.1%** | 97.3% | 74.3% |

The 124M rung loses nothing measurable against gpt2-xl on this data
(joint tied 65/72; lookahead-1 and streaming one word *ahead* — noise at
this n, but #66's ladder said most of the authority gain lands by 124M,
and this corroborates it). Every component of the winning configuration —
1.3M encoder, closed-form GestureDP, 320k trie, 124M LM — is
on-device-sized.

**Commitment-policy honesty.** QuickPath is not a joint decoder: it
commits at each word boundary, revising at most the just-committed word.
The matched-commitment comparison is therefore against our *streaming*
row — 87.5% vs 86.1%, one word apart, a wash at this n. The larger leads
come from deferred commitment, which is a *policy* QuickPath does not
ship, not a like-for-like decoder win: lookahead-1 (90.3%) revises the
previous word ~2% of the time, arguably within the envelope of revision
behavior iOS users already accept from autocorrect; joint may revise any
word in the sentence. The claim this study supports: at matched
commitment we tie the incumbent on 72 words, and the measured advantage
is the deferred-commitment lever itself plus the tail (86% vs 74.3%).

"Composed" = one acoustic formula per candidate,
`ctc_full + 0.8·uni + 1.2·len − 0.5·geom(dwell-weighted, tw=1.25)`, over
trie-beam top-8 ∪ geometry-trie top-8 (#73's proposal channel), then the
delta-form gpt2-xl sentence beam at μ=0.8 (#66/#72 marginal prior).

## What moved, mechanism by mechanism

- Context (joint, +7.0 over first pass): recovered exactly the predicted
  in-top8 confusions — `will→well`, `good→for`, `does→ford`, `im→in`.
  Streaming LM *hurt* the tail (77→69) before geometry: with no right
  context the delta form pushes rare words toward common ones.
- Geometry + proposals (+1.4 joint, coverage 90.3→95.8): surfaced and won
  `ngl`, `omg`, `sus-adjacent`; the off-domain γ=0.5 carried over from #73
  unchanged. This capture is a new apparatus — geometry's best case, as
  predicted.
- perm25mmi encoder: *worse* here than canonical (86.1 vs 90.3 joint,
  n=72). Its tail-in-beam edge did not show at this n; canonical stays the
  pick for this study.
- QuickPath's failure mode is the classic incumbent one: `priya→portia`,
  `divisadero→social`, `ngl→nfl`, `keeps→prod`. Tail 74.3% vs our 86%.

Remaining errors (composed joint): `address→area`, `so→do`, `sus→dys`,
`does→did` (ranking/coin-flips), `gonna/boba/keiko` (never surfaced —
enumeration + sloppy-gesture acoustics, the residual the notebook prices
as the trained noise model's last edge).

## Caveats

n=72, one user, one session: 65 vs 62 words against QuickPath is
directional, not significant. Our stack ran offline on an M-series Mac
with gpt2-xl; QuickPath ran live on the phone. The capture keyboard has
no autocorrect UI pressure, QuickPath does. Needed next: more users, more
sentences (power the McNemar), on-phone latency for the composed stack,
and a swap of gpt2-xl for something on-device-sized (gpt2-124M kept most
of the joint gain in #66's ladder).

## Pooled result (all 8 sets, 543 paired words) — the pilot did not generalize

Same user completed all 8 sets in one evening. Composed stack (canonical
encoder, gpt2-124M, joint): **72.7% vs QuickPath 72.4% — a statistical
dead heat** (discordants 71 vs 69, McNemar p=0.93; tail 42 vs 40,
p=0.91). Lookahead-1 71.3% (p=0.68); streaming 68.0%, *trending behind*
QuickPath (63 vs 87 discordants, p=0.06).

Per-set, joint vs native: set 1 (the pilot) reproduces exactly (90% vs
86%), so the pipeline is stable — the pilot's edge was set-1 luck. Sets
2–8 are harder for both keyboards (ours 59–82%, native 50–79%, winner
alternating by set), median swipe duration flat across sets (~380–500ms,
so not fatigue). Set-composition variance dominates the
between-keyboard difference; first-pass coverage fell 95.8% → 86.7% on
the new sets, and the new error mass is acoustic (`aisha→suga`,
`really→tally`, `makes→maid` — sloppy-gesture MISSes, not ranking).

Honest bottom line as of n=543: a stack trained purely on public
corpora, having never seen this user or device, on a web-capture
keyboard with no haptics, **ties the incumbent's shipped decoder** — and
does not beat it. The pilot's "+4pt" claim is retracted. The untested
levers that could break the tie are per-user adaptation (#58's
recipe now directly testable: fine-tune on ~5 sets of this user's
gestures, eval on held-out sets), personal vocabulary, and commitment
policy UX. The apparatus asymmetry (drawn web keyboard vs the native
keyboard the user has years of practice on) remains unpriced and works
against us.

## Per-user adaptation (#58's lever, cross-validated)

`adapt_user.py`: fine-tune `runs/full` on the user's own captured swipes
(LR 1e-4, 8 epochs, augmentation on; `user_rep` arm mixes 1500 FUTO
replay swipes), two sentence-disjoint folds (train 2,3,5,6,8 → eval
1,4,7 and the reverse), eval through the identical composed stack
(gpt2-124M).

**The adaptation gain is real and replicates**: joint 72.5→76.0 (fold A,
339 train swipes) and 72.9→74.9 (fold B, 204 train swipes) — pooled
held-out joint **72.7% → 75.3%** (+2.6, 395→409/543), with the same sign
in every mode of both folds (streaming +3.9/+2.9, lookahead-1 +1.0/+3.0,
first pass +4.4/+2.3, coverage +1.5 both). Replay mix ≥ user-only.
Twenty minutes of swiping is enough to move the composed stack ~3
points; training is seconds on a laptop — on-device feasible.

**The QuickPath comparison after adaptation: ahead, not significantly.**
Pooled adapted joint vs native: 75.3% vs 72.4%, discordants 78 vs 62,
p≈0.18. Fold A's tail significance (22 vs 9, p=0.029) did not replicate
in fold B (23 vs 27) — a multiple-comparisons lesson recorded here so it
does not get quoted; pooled tail is 45 vs 36, p≈0.32.

**Dose-response (fold A held-out, joint; chronological first-N swipes):**

| train swipes | 0 | 51 | 103 | 204 | 339 |
|---|---|---|---|---|---|
| joint top-1 | 72.5% | 71.6% | 72.5% | 74.0% | 76.0% |

The minimum effective dose is ~200 swipes (~10 minutes of deliberate
swiping, or a day of light use); 50–100 swipes buy nothing. No plateau
by 339 — the curve is still climbing, so the ceiling from real-usage
volumes (thousands of words/week) is unmeasured and above +3.5.
Untested compressions of the dose: the #58 synthetic style multiplier,
and an active-learning onboarding set instead of arbitrary sentences.

**Does the adaptation gain generalize across users? Mostly no
(`hws_peruser_adapt.py`).** The identical recipe (first 200 swipes +
replay, 8 epochs) run independently for the 13 HWS test users with ≥300
swipes: mean delta **+0.5 points** (median +0.7, sd 1.4), 9/13 improved,
4/13 slightly hurt, worst −2.7 — statistically ~nothing (mean is ~1 SE
from zero), versus +2.6 to +3.5 on our capture data. No visible
correlation between base accuracy and gain at this n.

The reconciliation: HWS users already read at ~81% base — the model is
in-regime for them, so 200 swipes have little to teach. Our capture
apparatus read at 41% geometry-decodability, and there adaptation moved
+3.5. **The lever is domain-gap repair, not universal per-user
personalization**: it pays where the input reads poorly (new apparatus,
sloppy styles) and is ~neutral where the base model already works. For a
launched keyboard that is not nothing — every new user's early period on
an unfamiliar keyboard *is* a domain gap — but the "keyboard that learns
you" accuracy claim is unsupported at 200-swipe doses on in-regime
users, and should not be made.

Standing read: base stack ties the incumbent; +20 minutes of user data
puts it numerically ahead everywhere but below the significance bar —
and the per-user study says most of that gain was apparatus adaptation,
not personalization.

## Expanded protocol (post-pilot)

`index.html` now carries **8 phrase sets** (96 sentences, 323 unique
words; set 1 is the pilot set). Each session: pick initials, pick an
unused set, run Block A then Block B on the *same* set (~10 min). All
scoring keys on (session, sentence), so multiple users and sets pool
into one paired analysis. One deliberate OOV probe is included (`istg`,
set 7) — unreachable by our lexicon by construction; it measures the
real-OOV failure mode on both sides (#42's territory), and should be
reported separately, not averaged in silently.

Power targets from the pilot effect sizes (paired McNemar): ~3 sessions
for the tail claim, ~7–8 for the overall claim; the matched-commitment
streaming comparison is expected to stay a wash at any feasible n.

## Files

- `index.html` — capture app (Block A canonical keyboard, Block B native)
- `server.py` — LAN server, POST /save → `data/`
- `decode_capture.py` — first pass + native-condition scoring
- `fused_rescore.py` — delta-form LM sentence beam over beam top-8
- `composed_stack.py` — geometry fusion + proposals + LM, the full cell
