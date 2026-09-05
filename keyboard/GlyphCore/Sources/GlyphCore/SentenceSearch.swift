import Foundation

/// Scores words in context: log P(word | left context) under a causal LM, and
/// the LM's own marginal log P(word) estimated over neutral prefixes. A port
/// of `capture/fused_rescore.py`'s `LMScorer`.
public protocol LanguageModel: AnyObject {
    /// log P(word | ctx) for every (ctx, word) pair. `ctx` is the lowercase
    /// left context joined by single spaces ("" at sentence start).
    func score(pairs: [(ctx: String, word: String)]) throws -> [Double]
}

/// The eight neutral prefixes the research code averages over to estimate the
/// LM's marginal (its own prior), which the delta form subtracts.
public let marginalContexts = ["", "i think", "and then", "she said", "it was", "we can", "they will", "he did"]

public struct FusedConfig {
    /// LM weight on the delta `log P(w|ctx) − log P(w)`.
    public var mu = 0.8
    /// Sentence hypotheses kept alive.
    public var beam = 8
    /// First-pass candidates considered per word.
    public var m = 8
    /// Commitment lag: 0 = streaming, 1 = lookahead-1 (the previous word may
    /// be revised once), nil = joint (any word may change until the sentence ends).
    public var lag: Int? = 1
    public init() {}
}

/// One live sentence hypothesis.
public struct Hypothesis: Equatable {
    public var words: [String]
    public var score: Double
}

/// The fused sentence beam: acoustic (first-pass) scores plus the delta-form LM
/// contribution, over word sequences, with a commitment lag. Mirrors
/// `sentence_decode` in `capture/fused_rescore.py` exactly, step by step, so a
/// sentence fed word by word here produces the same hypotheses.
public final class SentenceSearch {
    public let lm: LanguageModel
    public var config: FusedConfig
    public private(set) var states: [Hypothesis] = [Hypothesis(words: [], score: 0)]
    /// Words already committed (fixed) by the lag policy.
    public private(set) var committed = 0
    private var cache: [String: Double] = [:]       // "ctx\u{1}word" -> logp
    private var priorCache: [String: Double] = [:]  // word -> marginal logp
    /// Optional precomputed marginal prior (tools/export_priors.py); nil falls
    /// back to averaging the LM over `marginalContexts` on the fly.
    public var priorLookup: ((String) -> Double?)?
    /// Left context that precedes the swiped words (e.g. tapped letters/words
    /// earlier in the same sentence), lowercase words separated by spaces.
    public var prefix: String = ""

    public init(lm: LanguageModel, config: FusedConfig = FusedConfig()) {
        self.lm = lm
        self.config = config
    }

    public func reset(prefix: String = "") {
        states = [Hypothesis(words: [], score: 0)]
        committed = 0
        self.prefix = prefix
        cache.removeAll(keepingCapacity: true)
    }

    /// Feed the next swipe's first-pass candidates (word, acoustic score),
    /// best first. Returns the new best hypothesis. `words.count` grows by one.
    @discardableResult
    public func step(candidates: [(word: String, acoustic: Double)]) throws -> Hypothesis {
        let cands = Array(candidates.prefix(config.m))
        guard !cands.isEmpty else {
            states = states.map { Hypothesis(words: $0.words + [""], score: $0.score) }
            return states[0]
        }
        let ctxs = states.map { (prefix.isEmpty ? [] : [prefix]) + $0.words }.map { $0.joined(separator: " ") }
        // Fill the LM cache for every (ctx, word) and the priors.
        var need: [(ctx: String, word: String)] = []
        for c in ctxs { for cd in cands where cache[key(c, cd.word)] == nil { need.append((c, cd.word)) } }
        for cd in cands where priorCache[cd.word] == nil {
            if let p = priorLookup?(cd.word) { priorCache[cd.word] = p; continue }
            for c in marginalContexts where cache[key(c, cd.word)] == nil { need.append((c, cd.word)) }
        }
        need = dedupe(need)
        if !need.isEmpty {
            let got = try lm.score(pairs: need)
            for (p, v) in zip(need, got) { cache[key(p.ctx, p.word)] = v }
        }
        var priors: [String: Double] = [:]
        for cd in cands {
            if let p = priorCache[cd.word] { priors[cd.word] = p; continue }
            let p = marginalContexts.map { cache[key($0, cd.word)]! }.reduce(0, +) / Double(marginalContexts.count)
            priorCache[cd.word] = p
            priors[cd.word] = p
        }
        // Expand every state by every candidate; keep the best score per word sequence.
        var expansions: [[String]: Double] = [:]
        var order: [[String]] = []
        for (s, ctx) in zip(states, ctxs) {
            for cd in cands {
                let sc = s.score + cd.acoustic + config.mu * (cache[key(ctx, cd.word)]! - priors[cd.word]!)
                let wt = s.words + [cd.word]
                if let old = expansions[wt] {
                    if sc > old { expansions[wt] = sc }
                } else {
                    expansions[wt] = sc
                    order.append(wt)
                }
            }
        }
        // Python's sorted is stable over insertion order; mirror that for ties.
        var ranked = order.map { Hypothesis(words: $0, score: expansions[$0]!) }
        ranked = ranked.enumerated().sorted { a, b in
            a.element.score != b.element.score ? a.element.score > b.element.score : a.offset < b.offset
        }.map { $0.element }
        states = Array(ranked.prefix(config.beam))

        // Commitment: fix word t - lag to the best hypothesis's choice.
        let t = states[0].words.count - 1
        if let lag = config.lag, t - lag >= 0 {
            let j = t - lag
            let w = states[0].words[j]
            let kept = states.filter { $0.words[j] == w }
            states = kept.isEmpty ? [states[0]] : kept
            committed = j + 1
        }
        return states[0]
    }

    /// The best hypothesis's words.
    public var best: [String] { states[0].words }

    /// Alternatives for the most recent word, ranked by fused score (best first,
    /// deduplicated), for the suggestion bar.
    public func alternativesForLastWord() -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for s in states {
            guard let w = s.words.last, !seen.contains(w) else { continue }
            seen.insert(w); out.append(w)
        }
        return out
    }

    /// Replace the last word with the user's pick and collapse the beam onto it.
    public func forceLastWord(_ word: String) {
        guard !states[0].words.isEmpty else { return }
        if let s = states.first(where: { $0.words.last == word }) {
            states = [s]
        } else {
            var s = states[0]; s.words[s.words.count - 1] = word
            states = [s]
        }
    }

    private func key(_ ctx: String, _ word: String) -> String { ctx + "\u{1}" + word }

    private func dedupe(_ pairs: [(ctx: String, word: String)]) -> [(ctx: String, word: String)] {
        var seen = Set<String>(); var out: [(ctx: String, word: String)] = []
        for p in pairs where !seen.contains(key(p.ctx, p.word)) { seen.insert(key(p.ctx, p.word)); out.append(p) }
        return out
    }
}
