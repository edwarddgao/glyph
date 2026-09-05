#if canImport(CoreML)
import CoreML
import Foundation

/// The autoregressive swipe decoder as a first pass: a Core ML encoder turns
/// the (64, 32) features into a (64, 128) memory, and a Core ML step model
/// scores the next letter for a batch of K prefixes. The trie-constrained
/// beam is a port of `ar_beam` in `research/src/swipe_typing/model/ar.py`.
public final class ARFirstPass: FirstPass {
    public struct Meta: Decodable {
        public let checkpoint: String; public let beam, max_word_len, vocab, bos, eos, d_model: Int
    }
    public struct Weights {
        /// acoustic = ar_logp + alpha·unigram + beta·length − lambda·ilm
        public var alpha = 0.6, beta = 1.2, lambda = 0.0
        public init(alpha: Double = 0.6, beta: Double = 1.2, lambda: Double = 0.0) { self.alpha = alpha; self.beta = beta; self.lambda = lambda }
    }

    public let trie: Trie
    public let meta: Meta
    public var weights: Weights
    /// Optional per-trie-node internal-LM table (the #78 mean-memory ablation), NaN where absent.
    public var ilm: [Float]?
    /// See the early exit in `candidates`; 0 disables it. Longer words gain
    /// beta per letter in the composed score, so the margin covers that too.
    public var earlyExitMargin: Double = 6.0
    let encoder: MLModel, step: MLModel
    let K: Int, frames = Features.nPoints, d: Int
    let memoryIn: MLMultiArray
    let featuresIn: MLMultiArray
    private var tokenArrays: [Int: MLMultiArray] = [:]

    public init(encoderURL: URL, stepURL: URL, meta: Meta, trie: Trie, weights: Weights = Weights(),
                computeUnits: MLComputeUnits = .cpuOnly) throws {
        let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits
        func load(_ url: URL) throws -> MLModel {
            let target = url.pathExtension == "mlpackage" ? try MLModel.compileModel(at: url) : url
            return try MLModel(contentsOf: target, configuration: cfg)
        }
        encoder = try load(encoderURL); step = try load(stepURL)
        self.meta = meta; self.trie = trie; self.weights = weights
        K = meta.beam; d = meta.d_model
        featuresIn = try MLMultiArray(shape: [1, NSNumber(value: frames), NSNumber(value: Features.nInput)], dataType: .float32)
        memoryIn = try MLMultiArray(shape: [NSNumber(value: K), NSNumber(value: frames), NSNumber(value: d)], dataType: .float32)
    }

    /// (64, 128) memory for one gesture, replicated into the K-row batch input.
    func encode(_ features: [Float]) throws {
        let s1 = featuresIn.strides[1].intValue, s2 = featuresIn.strides[2].intValue
        let fp = featuresIn.dataPointer.assumingMemoryBound(to: Float.self)
        for f in 0..<frames { for c in 0..<Features.nInput { fp[f * s1 + c * s2] = features[f * Features.nInput + c] } }
        let out = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: featuresIn)]))
        guard let mem = out.featureValue(for: "memory")?.multiArrayValue else { throw CoreMLAcousticModel.ModelError.missingOutput }
        let m1 = mem.strides[1].intValue, m2 = mem.strides[2].intValue
        let mp = mem.dataPointer.assumingMemoryBound(to: Float.self)
        let o0 = memoryIn.strides[0].intValue, o1 = memoryIn.strides[1].intValue, o2 = memoryIn.strides[2].intValue
        let op = memoryIn.dataPointer.assumingMemoryBound(to: Float.self)
        for f in 0..<frames {
            for c in 0..<d {
                let v = mp[f * m1 + c * m2]
                for k in 0..<K { op[k * o0 + f * o1 + c * o2] = v }
            }
        }
    }

    private func tokens(length L: Int) throws -> MLMultiArray {
        if let a = tokenArrays[L] { return a }
        let a = try MLMultiArray(shape: [NSNumber(value: K), NSNumber(value: L)], dataType: .int32)
        tokenArrays[L] = a
        return a
    }

    /// Port of `ar_beam` for one gesture. Returns candidates ranked by the
    /// composed acoustic score; `acoustic` carries the raw AR log-prob.
    public func candidates(_ samples: [TouchSample]) throws -> [Candidate] {
        let x = Features.encode(samples)
        try encode(x)
        let nKeys = Geometry.nKeys, NEG = -Double.greatestFiniteMagnitude / 4
        var score = [Double](repeating: NEG, count: K); score[0] = 0
        var node = [Int32](repeating: 0, count: K)
        var toks: [[Int32]] = Array(repeating: [Int32(meta.bos)], count: K)
        var finished: [Int32: (ar: Double, len: Int)] = [:]   // by terminal node
        var live = 1

        for stepIdx in 0...meta.max_word_len {
            let L = stepIdx + 1
            let tokIn = try tokens(length: L)
            let t0 = tokIn.strides[0].intValue, t1 = tokIn.strides[1].intValue
            let tp = tokIn.dataPointer.assumingMemoryBound(to: Int32.self)
            for k in 0..<K { for l in 0..<L { tp[k * t0 + l * t1] = toks[k][l] } }
            let out = try step.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "memory": MLFeatureValue(multiArray: memoryIn), "tokens": MLFeatureValue(multiArray: tokIn)]))
            guard let lp = out.featureValue(for: "logp")?.multiArrayValue else { throw CoreMLAcousticModel.ModelError.missingOutput }
            let p0 = lp.strides[0].intValue, p1 = lp.strides[1].intValue
            let lpP = lp.dataPointer.assumingMemoryBound(to: Float.self)

            // close beams on word-terminal nodes
            for k in 0..<live where score[k] > NEG / 2 && node[k] != 0 && trie.isWord(node[k]) {
                let s = score[k] + Double(lpP[k * p0 + meta.eos * p1])
                if let old = finished[node[k]] { if s > old.ar { finished[node[k]] = (s, toks[k].count - 1) } }
                else { finished[node[k]] = (s, toks[k].count - 1) }
            }
            if stepIdx == meta.max_word_len { break }
            // Early exit: scores only fall along a path (log-probs), so once even
            // the best live prefix sits `margin` below the 8th-best finished word's
            // raw AR score, no continuation can enter the top list.
            if finished.count >= 8 {
                let eighth = finished.values.map { $0.ar }.sorted(by: >)[7]
                var bestLive = NEG
                for k in 0..<live where score[k] > bestLive { bestLive = score[k] }
                if bestLive < eighth - earlyExitMargin { break }
            }
            // expand
            var cands: [(score: Double, src: Int, letter: Int, child: Int32)] = []
            cands.reserveCapacity(live * 8)
            for k in 0..<live where score[k] > NEG / 2 {
                if !trie.hasChildren(node[k]) { continue }
                for l in 0..<nKeys {
                    let child = trie.child(of: node[k], letterIndex: l)
                    if child < 0 { continue }
                    cands.append((score[k] + Double(lpP[k * p0 + l * p1]), k, l, child))
                }
            }
            if cands.isEmpty { break }
            if cands.count > K { cands.sort { $0.score > $1.score }; cands.removeLast(cands.count - K) }
            else { cands.sort { $0.score > $1.score } }
            var newScore = [Double](repeating: NEG, count: K), newNode = [Int32](repeating: 0, count: K)
            var newToks: [[Int32]] = Array(repeating: [], count: K)
            for (i, c) in cands.enumerated() {
                newScore[i] = c.score; newNode[i] = c.child; newToks[i] = toks[c.src] + [Int32(c.letter)]
            }
            // dead rows: keep a valid token row (bos + padding) so the batch is well-formed
            for i in cands.count..<K { newToks[i] = toks[0].count == L ? toks[0] : Array(repeating: Int32(meta.bos), count: L) ; newToks[i] = newToks[i] + [Int32(meta.eos)] }
            for i in cands.count..<K { newToks[i] = Array(newToks[i].prefix(L + 1)) }
            score = newScore; node = newNode; toks = newToks; live = cands.count
        }

        var out: [Candidate] = []
        for (n, f) in finished {
            let uni = trie.logp(n)
            var sc = f.ar + weights.alpha * uni + weights.beta * Double(f.len)
            if let ilm, weights.lambda != 0 { let v = ilm[Int(n)]; if !v.isNaN { sc -= weights.lambda * Double(v) } }
            out.append(Candidate(word: trie.word(at: n), score: sc, acoustic: f.ar, unigram: uni, length: f.len))
        }
        out.sort { $0.score > $1.score }
        return out
    }
}
#endif
