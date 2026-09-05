# Show HN draft

**Title:** Show HN: Glyph – an open-source iPhone swipe keyboard that beats QuickPath, Gboard and SwiftKey on replayed gestures

**Text:**

I built a swipe keyboard for iPhone whose decoder runs entirely on the phone
and, on the same recorded finger paths, commits the right word more often than
Apple's QuickPath, Gboard or SwiftKey. Everything is open: the model, the
training code, the benchmark harness and a lab notebook of ~85 experiments.

How it is measured. Recorded swipes (real ones from an iPhone, and 1,337 words
from the public FUTO corpus) are replayed byte-for-byte onto each keyboard
through XCTest's touch synthesizer on an iPhone 17, and the word each keyboard
commits is scored, paired word by word (McNemar).

| keyboard | real iPhone swipes (542 words) | FUTO (1,337 words) |
|---|---|---|
| Glyph | 77.9% | 93.4% |
| QuickPath | 74.9% | 90.2% |
| SwiftKey | 69.0% | 85.9% |
| Gboard | 68.5% | 88.0% |

The caveat up front: the real-swipe set is one person's 542 words (mine —
fast and sloppy, twice the speed of the corpus donors). The margin over
QuickPath there is +3.0 at p = 0.09; on everyday words it is +6.9 at p = 0.001;
on rare words and sentence-initial words we are level. FUTO is p < 0.001
against all three. Whether this holds for other people is the open question,
and it is why the app opens with a practice run.

What is inside. A 1.7M-parameter TCN + transformer letter decoder trained on
FUTO and How We Swipe with a trie-constrained beam, fused with distilgpt2 as a
sentence model, all Core ML on the CPU, ~150 ms per word. No network access,
no Full Access. The interesting findings along the way are in the notebook:
label cleaning is a null in-domain and +1 cross-corpus; the language model is
worth +4 but hurts on the first word of a sentence; how far a swipe can stray
from its word before any decoder loses it.

Practice. Glyph opens with a timed practice run: swipe the words of a sentence. A word counts when the finger geometrically traced it, not when the
model agrees, so the data never gets biased toward what the model already
handles. Every attempt is uploaded with the prompted word under a random id;
nothing typed on the keyboard ever is. That data is what will settle the
question above, and it is what the next model trains on.

Repo: https://github.com/edwarddgao/glyph — TestFlight: https://testflight.apple.com/join/ZAXsVCWz
