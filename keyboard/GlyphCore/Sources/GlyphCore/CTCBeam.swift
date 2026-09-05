import Foundation

/// `BeamConfig` from `research/src/swipe_typing/model/beam.py`.
public struct BeamConfig {
    public var beamWidth = 64
    public var pruneLogP = -13.0
    public var alpha = 0.8   // unigram prior weight (ranking only)
    public var beta = 1.2    // per-letter bonus (ranking only)
    public init() {}
}

public struct Candidate: Equatable {
    public var word: String
    /// acoustic + alpha * unigram + beta * length
    public var score: Double
    public var acoustic: Double
    public var unigram: Double
    public var length: Int
    public init(word: String, score: Double, acoustic: Double, unigram: Double, length: Int) {
        self.word = word; self.score = score; self.acoustic = acoustic; self.unigram = unigram; self.length = length
    }
}

/// Trie-constrained CTC prefix beam search (Hannun et al. 2014), a direct port
/// of `beam._search`. A prefix is its trie node id.
public enum CTCBeam {
    struct Beam {
        var pBlank: Double
        var pNonBlank: Double
        @inline(__always) var total: Double { logAddExp(pBlank, pNonBlank) }
    }

    @inline(__always)
    static func logAddExp(_ a: Double, _ b: Double) -> Double {
        if a == -.infinity { return b }
        if b == -.infinity { return a }
        return a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))
    }

    /// `logProbs` is row-major (T, 27) with the blank at index 26.
    /// Returns the finished, ranked candidates (best first).
    public static func search(logProbs: [Float], frames: Int, trie: Trie,
                              config: BeamConfig = BeamConfig()) -> [Candidate] {
        let nLabels = Geometry.nKeys
        let stride = nLabels + 1
        precondition(logProbs.count >= frames * stride)

        var beams: [Int32: Beam] = [0: Beam(pBlank: 0.0, pNonBlank: -.infinity)]
        var next: [Int32: Beam] = [:]
        var candidates = [Int](); candidates.reserveCapacity(nLabels)

        for f in 0..<frames {
            let base = f * stride
            candidates.removeAll(keepingCapacity: true)
            for k in 0..<nLabels where Double(logProbs[base + k]) > config.pruneLogP { candidates.append(k) }
            let lpBlank = Double(logProbs[base + nLabels])
            next.removeAll(keepingCapacity: true)

            for (node, beam) in beams {
                let total = beam.total

                // (a) blank: prefix unchanged, now ends in blank
                var entry = next[node] ?? Beam(pBlank: -.infinity, pNonBlank: -.infinity)
                entry.pBlank = logAddExp(entry.pBlank, total + lpBlank)
                // (b) repeat last label without a blank: collapses
                let last = node == 0 ? -1 : trie.letterIndex(node)
                if last >= 0 {
                    entry.pNonBlank = logAddExp(entry.pNonBlank, beam.pNonBlank + Double(logProbs[base + last]))
                }
                next[node] = entry

                // (c) extend where the lexicon allows
                if !trie.hasChildren(node) { continue }
                for k in candidates {
                    let child = trie.child(of: node, letterIndex: k)
                    if child < 0 { continue }
                    let score = Double(logProbs[base + k])
                    var nb = next[child] ?? Beam(pBlank: -.infinity, pNonBlank: -.infinity)
                    if k == last {
                        nb.pNonBlank = logAddExp(nb.pNonBlank, beam.pBlank + score)
                    } else {
                        nb.pNonBlank = logAddExp(nb.pNonBlank, total + score)
                    }
                    next[child] = nb
                }
            }
            if next.isEmpty { break }
            if next.count > config.beamWidth {
                let kept = next.sorted { $0.value.total > $1.value.total }.prefix(config.beamWidth)
                beams = Dictionary(uniqueKeysWithValues: kept.map { ($0.key, $0.value) })
            } else {
                swap(&beams, &next)
            }
        }

        var out = [Candidate]()
        for (node, beam) in beams where node != 0 && trie.isWord(node) {
            let total = beam.total
            if total == -.infinity || total.isNaN { continue }
            let length = trie.depth(node)
            let unigram = trie.logp(node)
            out.append(Candidate(word: trie.word(at: node),
                                 score: total + config.alpha * unigram + config.beta * Double(length),
                                 acoustic: total, unigram: unigram, length: length))
        }
        out.sort { $0.score > $1.score }
        return out
    }
}
