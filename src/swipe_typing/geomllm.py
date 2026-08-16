"""Training-free joint decoder: LLM token beam scored by analytic geometry.

The learned stack is *encoder proposes, LM disposes*. This is the training-free
inversion: the LLM's own next-token distribution replaces the lexicon trie, and
an analytic alignment cost replaces the CTC emissions. Nothing here has ever
seen a gesture: the geometric channel is built from the keyboard layout alone,
the linguistic channel from a pretrained LM, and the single free parameter
(``lm_weight``) is a hyperparameter, not a trained weight.

Geometric channel
-----------------
A hypothesis is a letter string. Its cost explains *every* resampled gesture
point: each letter lands on one point (Gaussian around the key center), and
points between consecutive landings are transit (Gaussian around the segment
joining the two key centers, down-weighted — transit is less informative than
landing). Leading/trailing points are transit around the first/last key.
Because every hypothesis explains the same N points, costs are directly
comparable across hypotheses of different lengths — no length normalization.

The alignment is a DP over gesture positions, incremental in letters: a
hypothesis carries one O(N) cost row, and extending it by a letter is one
O(N) recurrence. This is what lets the geometry ride along inside a token-level
beam. A letter the finger cut the corner on (the classic ``that`` with no
``h`` under the path) is not impossible, merely costly — the failure mode that
kills strict subsequence decoding.

Linguistic channel
------------------
Beam over LM tokens, restricted to pure-lowercase-letter continuations (the
corpora are lowercase and unpunctuated). Word-start tokens are space-prefixed;
a word can end whenever the LM's next-token mass leaves letter continuations,
scored as log(1 - p(continue)). Candidate tokens are proposed by the *sum* of
LM logprob and a geometric probe of the token's first letter, so a
geometrically obvious word survives a cold context and vice versa.

Distances are measured in key half-extent units (see ``trace.nearest_keys``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import resample
from .layout import KeyboardLayout


# ---------------------------------------------------------------------------
# Geometric channel


@dataclass
class GeomConfig:
    n_points: int = 96          #: gesture resampling (arclength)
    sigma_key: float = 1.0      #: landing tolerance, key half-extents
    sigma_transit: float = 2.0  #: transit tolerance, key half-extents
    w_transit: float = 0.3     #: per-point weight of transit vs landing
    #: exponent on per-point dwell time: transit through a region where the
    #: finger lingered costs more, so hypotheses that put a letter at the
    #: dwell are favored. 0 disables; the alignment is otherwise blind to
    #: timing (arclength resampling discards it).
    time_weight: float = 0.0
    #: (dx, dy) touch bias in canonical units, added to key centers — users
    #: systematically touch below centers (README calibration section).
    offset: tuple[float, float] = (0.0, 0.0)


class GestureDP:
    """Incremental alignment cost between one gesture and letter prefixes.

    ``init_row(c)`` starts a hypothesis with first letter ``c``;
    ``extend(row, prev, c)`` appends a letter; ``final(row, c)`` closes the
    hypothesis. Rows are (N,) float64: ``row[j]`` is the best cost of
    explaining points 0..j with the prefix, its last letter landing on j.
    """

    def __init__(self, points: np.ndarray, t: np.ndarray,
                 kb: KeyboardLayout | None = None,
                 cfg: GeomConfig | None = None) -> None:
        kb = kb or KeyboardLayout.qwerty()
        self.cfg = cfg or GeomConfig()
        self.kb = kb
        # Key units: offsets divided by per-key half-extents.
        self.scale = kb.radii.mean(axis=0).astype(np.float64)
        xy = resample(points, t, n=self.cfg.n_points, mode="arclength")
        self.g = xy.astype(np.float64) / self.scale          # (N, 2)
        centers = kb.centers.astype(np.float64) + np.asarray(self.cfg.offset)
        self.keys = centers / self.scale                     # (K, 2)
        self.n = len(self.g)
        # Per-point dwell weight: time spent near each resampled point,
        # normalized to mean 1. Multiplies transit costs only — landing is
        # location evidence, dwell is where-the-finger-paused evidence.
        if self.cfg.time_weight > 0:
            pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            u = np.concatenate([[0.0], np.cumsum(step)])
            tt = np.asarray(t, dtype=np.float64)
            tt = np.maximum.accumulate(tt)
            grid = np.linspace(0.0, max(u[-1], 1e-9), self.n)
            ti = np.interp(grid, u, tt)
            dt = np.gradient(ti)
            w = dt / max(dt.mean(), 1e-9)
            self._tw = np.clip(w, 0.25, 4.0) ** self.cfg.time_weight
        else:
            self._tw = np.ones(self.n)
        # Cached per-letter landing and transit-to-key cost vectors.
        d2 = ((self.g[:, None, :] - self.keys[None, :, :]) ** 2).sum(-1)
        self._land = d2 / self.cfg.sigma_key ** 2                 # (N, K)
        self._hover = (self.cfg.w_transit * d2 * self._tw[:, None]
                       / self.cfg.sigma_transit ** 2)             # (N, K)
        self._seg_cache: dict[int, np.ndarray] = {}
        # Lower bound on the cost of explaining points after j: each must at
        # least hover near its nearest key. Charged to partial hypotheses so
        # that camping on a path prefix is not free — without it, degenerate
        # strings that explain 10% of the gesture outscore real words that
        # consume all of it.
        best_hover = self._hover.min(axis=1)
        self.tail_bound = np.concatenate(
            [np.cumsum(best_hover[::-1])[-2::-1], [0.0]])

    def _seg_transit(self, a: int, b: int) -> np.ndarray:
        """(N,) transit cost of each point against segment key a -> key b."""
        return self._seg_all(a)[b]

    def _seg_all(self, a: int) -> np.ndarray:
        """(K, N) transit cost against every segment key a -> key b."""
        hit = self._seg_cache.get(a)
        if hit is not None:
            return hit
        p = self.keys[a]
        v = self.keys - p                                   # (K, 2)
        vv = (v * v).sum(-1)                                # (K,)
        safe = np.maximum(vv, 1e-12)
        tt = np.clip((self.g - p) @ v.T / safe, 0.0, 1.0)   # (N, K)
        proj = p + tt[:, :, None] * v[None, :, :]           # (N, K, 2)
        d2 = ((self.g[:, None, :] - proj) ** 2).sum(-1)     # (N, K)
        out = (self.cfg.w_transit * d2 * self._tw[:, None]
               / self.cfg.sigma_transit ** 2).T
        out[vv < 1e-12] = self._hover[:, a]
        self._seg_cache[a] = out
        return out

    def init_row(self, c: str) -> np.ndarray:
        """First letter: leading points are transit around its key."""
        k = self.kb.index(c)
        lead = np.concatenate([[0.0], np.cumsum(self._hover[:-1, k])])
        return lead + self._land[:, k]

    def extend(self, row: np.ndarray, prev: str, c: str) -> np.ndarray:
        """Append letter ``c`` after ``prev``. O(N)."""
        trans = self._seg_all(self.kb.index(prev))[self.kb.index(c)]
        cum = np.cumsum(trans)
        shifted = np.concatenate(
            [[np.inf], np.minimum.accumulate(row - cum)[:-1]])
        between = shifted + np.concatenate([[0.0], cum[:-1]])
        return self._land[:, self.kb.index(c)] + np.minimum(row, between)

    def extend_many(self, row: np.ndarray, prev: str) -> np.ndarray:
        """(K, N) rows for appending every letter after ``prev`` at once.

        new[j] = land[j] + min( row[j],                      i == j
                                min_{i<j} row[i] - cum[i] + cum[j-1] )
        where cum[j] = sum_{l<=j} trans[l] and the i<j branch charges
        transit for the points strictly between the two landings. One
        vectorized recurrence replaces 26 scalar ones — the beam probes
        every letter of every hypothesis every step.
        """
        trans = self._seg_all(self.kb.index(prev))          # (K, N)
        cum = np.cumsum(trans, axis=1)
        base = row[None, :] - cum
        shifted = np.empty_like(base)
        shifted[:, 0] = np.inf
        np.minimum.accumulate(base[:, :-1], axis=1, out=shifted[:, 1:])
        between = shifted + np.concatenate(
            [np.zeros((len(trans), 1)), cum[:, :-1]], axis=1)
        return self._land.T + np.minimum(row[None, :], between)

    def init_all(self) -> np.ndarray:
        """(K, N) first-letter rows for every letter, cached."""
        if not hasattr(self, "_init_all"):
            lead = np.concatenate(
                [np.zeros((1, len(self.keys))), np.cumsum(self._hover[:-1],
                                                          axis=0)])
            self._init_all = (lead + self._land).T
        return self._init_all

    def final(self, row: np.ndarray, c: str) -> float:
        """Close the hypothesis: trailing points transit around ``c``'s key."""
        k = self.kb.index(c)
        tail = np.concatenate([np.cumsum(self._hover[::-1, k])[-2::-1], [0.0]])
        return float(np.min(row + tail))

    def word_cost(self, word: str) -> float:
        """Full alignment cost of a word — the non-incremental entry point."""
        row = self.init_row(word[0])
        for prev, c in zip(word, word[1:]):
            row = self.extend(row, prev, c)
        return self.final(row, word[-1])

    def probe_next(self, row: np.ndarray, prev: str) -> np.ndarray:
        """(K,) optimistic cost of extending by each letter (min over row).

        Used to gate token proposal, not to score.
        """
        return self.extend_many(row, prev).min(axis=1)


# ---------------------------------------------------------------------------
# Linguistic channel


class TokenLetterTable:
    """Letter view of an LM vocabulary.

    ``starts``: token ids that begin a word (space-prefixed, lowercase letters
    only). ``conts``: ids that continue one (letters only, no space). Both map
    to their letter strings; everything else can only end the word.
    """

    def __init__(self, tokenizer) -> None:
        size = max(tokenizer.get_vocab().values()) + 1
        toks = tokenizer.convert_ids_to_tokens(list(range(size)))
        self.letters: list[str | None] = [None] * size
        starts, conts = [], []
        for i, s in enumerate(toks):
            if s is None:
                continue
            if s.startswith("Ġ"):  # GPT-2/Qwen byte-BPE space marker
                body, is_start = s[1:], True
            else:
                body, is_start = s, False
            if body and all("a" <= ch <= "z" for ch in body):
                self.letters[i] = body
                (starts if is_start else conts).append(i)
        self.start_ids = np.asarray(starts, dtype=np.int64)
        self.cont_ids = np.asarray(conts, dtype=np.int64)
        self.start_first = np.asarray(
            [ord(self.letters[i][0]) - 97 for i in self.start_ids])
        self.cont_first = np.asarray(
            [ord(self.letters[i][0]) - 97 for i in self.cont_ids])
        # Positions (into start_ids/cont_ids) grouped by first letter, for
        # per-letter proposal quotas.
        self.start_by_first = [np.flatnonzero(self.start_first == c)
                               for c in range(26)]
        self.cont_by_first = [np.flatnonzero(self.cont_first == c)
                              for c in range(26)]
        # Position of each bare single-letter token. Always proposable for a
        # gated letter: single-letter tokens are LM-rare, so no LM ranking
        # surfaces them, yet they are the only guaranteed char-level path to
        # words whose own tokens are buried (rescore repairs the LM score).
        self.start_single = np.full(26, -1)
        self.cont_single = np.full(26, -1)
        for group, single in ((self.start_ids, self.start_single),
                              (self.cont_ids, self.cont_single)):
            for pos, tid in enumerate(group):
                s = self.letters[tid]
                if len(s) == 1:
                    single[ord(s) - 97] = pos


# ---------------------------------------------------------------------------
# Joint beam


@dataclass
class BeamConfig:
    beam: int = 32              #: live hypotheses kept per step
    topk_lm: int = 32           #: candidate tokens per hypothesis by LM
    topk_geom: int = 24         #: extra candidates via per-letter quotas
    gate_letters: int = 4       #: geometrically plausible next letters kept
    #: search steps; a word found letter-by-letter needs one step per letter
    #: plus one to be seen finishing, so this bounds recallable word length.
    max_tokens: int = 14
    max_letters: int = 24
    lm_weight: float = 0.5      #: in-search weight on LM logprob vs geometry
    rescore_weight: float = 1.0  #: final weight, canonical tokenization
    stop_margin: float = 10.0   #: how far past the argmax proof to search
    finished: int = 32          #: n-best size per pass


class ContextCache:
    """KV/recurrent-state cache of a fixed left context, expandable to any
    batch size.

    The context dominates per-forward compute (a ~20-token context against
    ≤10 word tokens), and it is identical for every hypothesis of a swipe —
    including both search passes. One batch-1 forward here; every subsequent
    step forwards only each hypothesis's word tokens against a batch-expanded
    copy. Works for plain KV layers (gpt2) and Qwen3.5's hybrid cache
    (DynamicLayer + LinearAttentionLayer) by repeating every batch-first
    tensor found in the layer objects. Verify with ``context_cache_ok`` before
    trusting a new architecture.
    """

    def __init__(self, lm, context_ids: list[int]) -> None:
        import torch

        device = next(lm.parameters()).device
        with torch.no_grad():
            out = lm(input_ids=torch.tensor([context_ids], device=device),
                     use_cache=True)
        self.master = out.past_key_values
        self.n_ctx = len(context_ids)
        #: next-token distribution right after the context — the step-0
        #: distribution for every stream, no forward needed.
        self.last_lp = (torch.log_softmax(out.logits[0, -1].float(), -1)
                        .cpu().numpy())

    @staticmethod
    def _clone_layer(layer, n: int):
        import copy as _copy

        import torch

        new = _copy.copy(layer)
        for name, val in vars(layer).items():
            if torch.is_tensor(val) and val.dim() >= 2:
                setattr(new, name, val.repeat(n, *([1] * (val.dim() - 1))))
            elif isinstance(val, dict):
                setattr(new, name, {
                    k: (v.repeat(n, *([1] * (v.dim() - 1)))
                        if torch.is_tensor(v) and v.dim() >= 2 else v)
                    for k, v in val.items()})
        return new

    def batch(self, n: int):
        """A fresh batch-``n`` copy (forwards mutate their cache)."""
        import copy as _copy

        new = _copy.copy(self.master)
        new.layers = [self._clone_layer(la, n) for la in self.master.layers]
        return new


def context_cache_ok(lm, tokenizer, atol: float = 0.05) -> bool:
    """Parity-check ContextCache against a full forward on this model."""
    import torch

    ids = tokenizer.encode("she said that")
    suffix = tokenizer.encode(" the dog")
    device = next(lm.parameters()).device
    try:
        cache = ContextCache(lm, ids)
        pkv = cache.batch(2)
        inp = torch.tensor([suffix, suffix], device=device)
        mask = torch.ones(2, len(ids) + len(suffix), dtype=torch.long,
                          device=device)
        with torch.no_grad():
            a = lm(input_ids=inp, past_key_values=pkv, attention_mask=mask,
                   use_cache=True).logits[1, -1].float()
            b = lm(input_ids=torch.tensor([ids + suffix], device=device)
                   ).logits[0, -1].float()
        a = torch.log_softmax(a, -1)
        b = torch.log_softmax(b, -1)
        top = torch.topk(b, 50).indices
        return bool((a[top] - b[top]).abs().max() < atol)
    except Exception:
        return False


@dataclass
class _Hyp:
    letters: str = ""
    row: np.ndarray | None = None
    #: canonical-prefix LM logprob. Set exactly after each forward; holds the
    #: parent's exact value plus the proposal token's logprob until then.
    lm_lp: float = 0.0

    def score(self, lm_weight: float, tail_bound: np.ndarray | None = None,
              ) -> float:
        if self.row is None:
            geom = 0.0
        elif tail_bound is not None:
            geom = float(np.min(self.row + tail_bound))
        else:
            geom = float(np.min(self.row))
        return lm_weight * self.lm_lp - geom


def _extend_row(dp: GestureDP, hyp: _Hyp, piece: str) -> np.ndarray | None:
    row = hyp.row
    prev = hyp.letters[-1] if hyp.letters else ""
    for c in piece:
        row = dp.init_row(c) if row is None else dp.extend(row, prev, c)
        prev = c
    return row


@dataclass
class _Stream:
    """One beam pass advancing in lockstep with its siblings."""

    cfg: BeamConfig
    beam: list = field(default_factory=lambda: [_Hyp()])
    finished: dict = field(default_factory=dict)
    steps: int = 0
    done: bool = False


def _score_hyps(lm, tokenizer, context_ids: list[int], hyps: list[_Hyp],
                cache: ContextCache | None, max_rows: int = 128,
                ) -> list[np.ndarray]:
    """Set each hypothesis's exact canonical-prefix logprob and return its
    next-token distribution.

    Right padding only: logits at real positions are unaffected by later pads
    in a causal model, whereas left padding corrupts models whose attention
    kernel ignores the mask (Qwen3.5's linear-attention fallback does). With
    a ContextCache, only each hypothesis's word tokens are forwarded and the
    empty hypothesis costs nothing at all.
    """
    import torch

    device = next(lm.parameters()).device
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    nctx = len(context_ids)
    dists: list[np.ndarray | None] = [None] * len(hyps)

    def score_chunk(rows: list[tuple[int, list[int]]], start: int) -> None:
        """One forward; all reductions batched on device, two transfers.

        ``rows`` are (hyp index, token seq); ``start`` is the position of the
        first token whose logprob counts toward the prefix (tokens before it
        are context). Naive per-row softmaxes and per-token Python sums cost
        20x the forward itself in launch overhead and GPU syncs.
        """
        width = max(len(s) for _, s in rows)
        inp = torch.full((len(rows), width), pad, dtype=torch.long)
        nexts = torch.zeros(len(rows), width, dtype=torch.long)
        want = torch.zeros(len(rows), width, dtype=torch.bool)
        last = torch.zeros(len(rows), dtype=torch.long)
        for i, (_, s) in enumerate(rows):
            inp[i, :len(s)] = torch.tensor(s)
            # position j predicts token s[j+1]; count from `start`.
            for j in range(start - 1, len(s) - 1):
                if j >= 0:
                    nexts[i, j] = s[j + 1]
                    want[i, j] = True
            last[i] = len(s) - 1
        with torch.no_grad():
            if cache is not None:
                mask = torch.ones(len(rows), nctx + width, dtype=torch.long)
                logits = lm(input_ids=inp.to(device),
                            past_key_values=cache.batch(len(rows)),
                            attention_mask=mask.to(device),
                            use_cache=True).logits
            else:
                logits = lm(input_ids=inp.to(device)).logits
            # Normalizers per position, fp32, without materializing the
            # full float logits.
            lse = torch.stack([torch.logsumexp(logits[:, w].float(), -1)
                               for w in range(width)], 1)
            nexts_d, want_d = nexts.to(device), want.to(device)
            tok_lp = (logits.gather(2, nexts_d[:, :, None])[:, :, 0].float()
                      - lse)
            prefix = (tok_lp * want_d).sum(1).cpu().numpy()
            last_lg = logits[torch.arange(len(rows), device=device),
                             last.to(device)].float()
            last_lp = (torch.log_softmax(last_lg, -1).cpu().numpy())
        for i, (r, s) in enumerate(rows):
            hyps[r].lm_lp = float(prefix[i])
            if cache is not None:
                hyps[r].lm_lp += float(cache.last_lp[s[0]])
            dists[r] = last_lp[i]

    # The passes run in lockstep over the same context, so identical letter
    # strings appear in several beams — forward each unique string once.
    groups: dict[str, list[int]] = {}
    for r, h in enumerate(hyps):
        groups.setdefault(h.letters, []).append(r)

    if cache is not None:
        need = [(rs[0], tokenizer.encode(" " + w))
                for w, rs in groups.items() if w]
        if "" in groups:
            for r in groups[""]:
                hyps[r].lm_lp = 0.0
                dists[r] = cache.last_lp
        for b in range(0, len(need), max_rows):
            score_chunk(need[b:b + max_rows], start=1)
    else:
        seqs = [(rs[0], context_ids + (tokenizer.encode(" " + w) if w else []))
                for w, rs in groups.items()]
        for b in range(0, len(seqs), max_rows):
            score_chunk(seqs[b:b + max_rows], start=nctx)

    for rs in groups.values():
        for r in rs[1:]:
            hyps[r].lm_lp = hyps[rs[0]].lm_lp
            dists[r] = dists[rs[0]]
    return dists


def _search_streams(lm, tokenizer, table: TokenLetterTable, dp: GestureDP,
                    context_ids: list[int], cfgs: list[BeamConfig],
                    cache: ContextCache | None = None,
                    ) -> list[dict[str, tuple[float, float, float]]]:
    """Run several beam passes in lockstep over one gesture and context.

    Hypotheses are letter strings, re-tokenized *canonically* at every step:
    a word reached letter-by-letter would otherwise carry the log-probability
    of its char-token path, which sinks ~2x faster than the word's own
    tokenization and starves long rare words out of the beam. The forward
    pass over the canonical tokens yields the exact prefix logprob for free,
    and the returned LM lp includes the word-boundary term, so pool ranking
    needs no second rescoring pass. All passes share one batched forward per
    step (and the context cache), which is why they run in lockstep.

    Returns one ``word -> (search score, geometric cost, LM lp)`` dict per
    config.
    """
    streams = [_Stream(cfg=c) for c in cfgs]
    while True:
        active = [s for s in streams if not s.done]
        if not active:
            break
        flat = [h for s in active for h in s.beam]
        dists = _score_hyps(lm, tokenizer, context_ids, flat, cache)
        i = 0
        for s in active:
            _advance(s, dp, table, dists[i:i + len(s.beam)])
            i += len(s.beam)
    return [s.finished for s in streams]


def _advance(st: _Stream, dp: GestureDP, table: TokenLetterTable,
             dists: list[np.ndarray]) -> None:
    """One step of one stream: record finishes, propose, prune."""
    cfg, finished = st.cfg, st.finished

    # Live scores are optimistic bounds on any descendant's final score;
    # stop once even the best trails the best finished word by the margin
    # (zero margin proves the argmax but starves the n-best).
    if finished:
        best_live = max(h.score(cfg.lm_weight, dp.tail_bound)
                        for h in st.beam)
        if best_live < (max(s for s, _, _ in finished.values())
                        - cfg.stop_margin):
            st.done = True
            return

    candidates: list[_Hyp] = []
    for h, lp in zip(st.beam, dists):
            ids = table.cont_ids if h.letters else table.start_ids
            first = table.cont_first if h.letters else table.start_first
            lp_sub = lp[ids]

            # A non-empty hypothesis may finish: the next token leaves the
            # letter-continuation set.
            if h.letters:
                p_cont = float(np.exp(lp[table.cont_ids]).sum())
                lp_end = float(np.log(max(1.0 - p_cont, 1e-9)))
                geom = dp.final(h.row, h.letters[-1])
                lm_full = h.lm_lp + lp_end
                score = cfg.lm_weight * lm_full - geom
                prev = finished.get(h.letters)
                if prev is None or prev[0] < score:
                    finished[h.letters] = (score, geom, lm_full)

            # Candidate tokens: LM's top-k plus per-letter quotas for the
            # geometrically plausible next letters. Geometry pins the next
            # letter to a handful of keys with near-certainty; within each,
            # the LM picks its best tokens — so a rare-but-on-path word is
            # proposed even when it is buried in the global LM ranking.
            take = set(np.argpartition(-lp_sub, min(cfg.topk_lm, len(lp_sub) - 1)
                                       )[:cfg.topk_lm])
            ext = (dp.extend_many(h.row, h.letters[-1]) if h.letters
                   else dp.init_all())
            if cfg.topk_geom > 0:
                probe = ext.min(axis=1)
                gate = lp_sub - probe[first]     # joint proposal score
                take |= set(np.argpartition(-gate, min(cfg.topk_geom,
                                                       len(gate) - 1)
                                            )[:cfg.topk_geom])
                by_first = (table.cont_by_first if h.letters
                            else table.start_by_first)
                single = (table.cont_single if h.letters
                          else table.start_single)
                quota = max(1, cfg.topk_geom // cfg.gate_letters)
                for c in np.argsort(probe)[:cfg.gate_letters]:
                    pos = by_first[c]
                    if len(pos) == 0:
                        continue
                    best = pos[np.argpartition(-lp_sub[pos],
                                               min(quota, len(pos) - 1))[:quota]]
                    take |= set(int(b) for b in best)
                    if single[c] >= 0:
                        take.add(int(single[c]))

            seen = set()
            for k in take:
                tid = int(ids[k])
                piece = table.letters[tid]
                if (piece in seen
                        or len(h.letters) + len(piece) > cfg.max_letters):
                    continue
                seen.add(piece)
                # No English word triples a letter; cheap orthographic guard
                # against degenerate repeat strings.
                joined = h.letters + piece
                if any(joined[m] == joined[m + 1] == joined[m + 2]
                       for m in range(max(0, len(h.letters) - 2),
                                      len(joined) - 2)):
                    continue
                row = ext[dp.kb.index(piece[0])]
                prev = piece[0]
                for ch in piece[1:]:
                    row = dp.extend(row, prev, ch)
                    prev = ch
                candidates.append(_Hyp(joined, row,
                                       h.lm_lp + float(lp[tid])))

        # Different token proposals can produce the same letter string; keep
    # the best-scored copy of each. Prune with per-progress quotas:
    # hypotheses at different points along the gesture are not
    # competitors, and a flat top-k lets whichever progress region scores
    # best this step evict everything ahead of or behind it.
    candidates.sort(key=lambda h: h.score(cfg.lm_weight, dp.tail_bound),
                    reverse=True)
    uniq: dict[str, _Hyp] = {}
    for h in candidates:
        if h.letters not in uniq:
            uniq[h.letters] = h
    per_bucket = max(4, cfg.beam // 4)
    counts = np.zeros(11, dtype=int)
    beam, overflow = [], []
    for h in uniq.values():
        b = int(10 * np.argmin(h.row + dp.tail_bound) / max(1, dp.n - 1))
        if counts[b] < per_bucket:
            counts[b] += 1
            beam.append(h)
        else:
            overflow.append(h)
        if len(beam) >= cfg.beam:
            break
    beam.extend(overflow[:cfg.beam - len(beam)])

    st.beam = beam
    st.steps += 1
    if st.steps >= cfg.max_tokens or not st.beam:
        st.done = True


def lm_proposal_config(cfg: BeamConfig) -> BeamConfig:
    """The LM-only companion pass for ``decode_word``.

    No geometry gate and an LM weight large enough that pruning is purely
    linguistic: the LM walks its own preferred (canonical) tokenizations, so
    this pass recalls contextually likely words the joint pass loses to
    path-hugging noise. Geometry re-enters at rescore time.
    """
    w = 50.0
    return BeamConfig(beam=min(cfg.beam, 24), topk_lm=max(cfg.topk_lm, 48),
                      topk_geom=0, max_tokens=min(cfg.max_tokens, 6),
                      max_letters=cfg.max_letters, lm_weight=w,
                      stop_margin=5.0 * w, finished=cfg.finished)


def decode_word(lm, tokenizer, table: TokenLetterTable, dp: GestureDP,
                context_ids: list[int], cfg: BeamConfig | None = None,
                lm_pass: bool = True, use_cache: bool = True,
                ) -> list[tuple[str, float]]:
    """N-best words for one gesture, best first.

    ``context_ids`` is the tokenized left context (BOS included); the decoded
    word continues it as a space-prefixed token sequence. Runs the joint
    geometric-LM search and the LM-only proposal pass in lockstep (shared
    forwards, shared context cache), and ranks the pooled words by
    ``rescore_weight * lm - geometry``. Both terms are already canonical (the
    search re-tokenizes every step), so the final ranking is one weighted
    sum — search recall and scoring stay decoupled through the pass-specific
    in-search weights.

    ``use_cache`` requires a cache-compatible model — gate on
    ``context_cache_ok`` once per model.
    """
    cfg = cfg or BeamConfig()
    cfgs = [cfg] + ([lm_proposal_config(cfg)] if lm_pass else [])
    cache = ContextCache(lm, context_ids) if use_cache else None
    pools = _search_streams(lm, tokenizer, table, dp, context_ids, cfgs,
                            cache)
    best: dict[str, tuple[float, float]] = {}
    for pool in pools:
        short = sorted(pool.items(), key=lambda kv: kv[1][0],
                       reverse=True)[:cfg.finished]
        for w, (_, g, lm_lp) in short:
            if w not in best or best[w][1] < lm_lp:
                best[w] = (g, lm_lp)
    ranked = [(w, cfg.rescore_weight * lm_lp - g)
              for w, (g, lm_lp) in best.items()]
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked
