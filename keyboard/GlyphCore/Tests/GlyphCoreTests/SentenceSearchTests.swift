import XCTest
@testable import GlyphCore

/// A LanguageModel backed by the (ctx, word) -> logp table the research code
/// produced, so the search port can be checked independently of the model.
final class TableLM: LanguageModel {
    var table: [String: Double] = [:]
    var misses = 0
    func score(pairs: [(ctx: String, word: String)]) throws -> [Double] {
        pairs.map { p in
            if let v = table[p.ctx + "\u{1}" + p.word] { return v }
            misses += 1; return -20
        }
    }
}

final class SentenceSearchTests: XCTestCase {
    struct Golden: Decodable {
        struct Cand: Decodable { let word: String; let acoustic: Double }
        struct Sent: Decodable { let sentence: String; let refs: [String]; let candidates: [[Cand]]; let decoded: [String: [String]] }
        struct Entry: Decodable { let ctx, word: String; let logp: Double }
        let lm: String; let mu: Double; let beam, m: Int; let marginal_ctxs: [String]
        let sentences: [Sent]; let lm_table: [Entry]
    }

    static let golden: Golden = {
        let url = GoldenTests.resources.appendingPathComponent("search_goldens.json")
        return try! JSONDecoder().decode(Golden.self, from: try! Data(contentsOf: url))
    }()

    func testReplaysResearchDecoder() throws {
        let g = Self.golden
        XCTAssertEqual(g.marginal_ctxs, marginalContexts)
        let lm = TableLM()
        for e in g.lm_table { lm.table[e.ctx + "\u{1}" + e.word] = e.logp }
        var cfg = FusedConfig(); cfg.mu = g.mu; cfg.beam = g.beam; cfg.m = g.m
        for (name, lag) in [("streaming", 0), ("lookahead1", 1), ("joint", nil as Int?)] {
            cfg.lag = lag
            var correct = 0, total = 0
            for s in g.sentences {
                let search = SentenceSearch(lm: lm, config: cfg)
                for cands in s.candidates {
                    try search.step(candidates: cands.map { ($0.word, $0.acoustic) })
                }
                XCTAssertEqual(search.best, s.decoded[name]!, "\(name): \(s.sentence)")
                correct += zip(search.best, s.refs).filter { $0 == $1 }.count
                total += s.refs.count
            }
            print("\(name): \(correct)/\(total) correct (python identical)")
        }
        XCTAssertEqual(lm.misses, 0, "every (ctx, word) the search asked for was in the research table")
    }

    #if canImport(CoreML)
    struct LMGolden: Decodable {
        struct Pair: Decodable { let ctx, word: String; let logp, coreml: Double }
        let model, quant: String; let B, L, P: Int; let pairs: [Pair]
    }

    func testCoreMLLanguageModelMatchesHF() throws {
        let res = GoldenTests.resources
        let g = try JSONDecoder().decode(LMGolden.self, from: try Data(contentsOf: res.appendingPathComponent("lm_goldens.json")))
        let meta = try JSONDecoder().decode(CoreMLLanguageModel.Meta.self, from: try Data(contentsOf: res.appendingPathComponent("lm_meta.json")))
        let tok = try GPT2Tokenizer(vocabURL: res.appendingPathComponent("gpt2/vocab.json"), mergesURL: res.appendingPathComponent("gpt2/merges.txt"))
        let lm = try CoreMLLanguageModel(modelURL: res.appendingPathComponent("SwipeLM.mlpackage"), meta: meta, tokenizer: tok)
        let pairs = g.pairs.map { (ctx: $0.ctx, word: $0.word) }
        let got = try lm.score(pairs: pairs)
        var maxHF = 0.0, maxCoreML = 0.0
        for (p, v) in zip(g.pairs, got) {
            maxHF = max(maxHF, abs(v - p.logp))
            maxCoreML = max(maxCoreML, abs(v - p.coreml))
        }
        print(String(format: "LM %@ %@: max |Δ| vs HF fp32 %.3f, vs python-coreml %.3f", g.model, g.quant, maxHF, maxCoreML))
        XCTAssertLessThan(maxCoreML, 0.05, "Swift and Python drive the same Core ML model identically")
        XCTAssertLessThan(maxHF, 1.0, "int8/fp16 error against the fp32 reference")
        // steady-state latency for one full batch
        let batchPairs = Array(pairs.prefix(meta.B))
        _ = try lm.score(pairs: batchPairs)
        let t0 = Date()
        for _ in 0..<5 { _ = try lm.score(pairs: batchPairs) }
        print(String(format: "LM batch of %d: %.0f ms (mac, steady)", meta.B, Date().timeIntervalSince(t0) * 200))
    }
    #endif
}

final class PriorTableTests: XCTestCase {
    /// priors.bin: "SWPR", u32 version, u32 N, float32[N] in lexicon.bin node order.
    func testPriorsAlignWithTrieAndMatchResearchEstimator() throws {
        let res = GoldenTests.resources
        let url = res.appendingPathComponent("priors.bin")
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("priors.bin not exported yet") }
        let data = try Data(contentsOf: url)
        XCTAssertEqual(Array(data[0..<4]), Array("SWPR".utf8))
        let n = Int(data.loadLE(UInt32.self, at: 8))
        let trie = GoldenTests.trie
        XCTAssertEqual(n, trie.count, "one prior per trie node")
        let priors = data.array(Float.self, at: 12, count: n)
        // every word has a finite prior, non-words are NaN
        var words = 0, finite = 0
        for node in 0..<n where trie.isWord(Int32(node)) { words += 1; if priors[node].isFinite { finite += 1 } }
        XCTAssertEqual(words, finite)
        XCTAssertTrue(priors[0].isNaN, "root is not a word")
        // agrees with the research estimator (mean over the 8 neutral contexts) on the LM goldens
        let g = try JSONDecoder().decode(SentenceSearchTests.LMGolden.self, from: try Data(contentsOf: res.appendingPathComponent("lm_goldens.json")))
        var byWord: [String: [Double]] = [:]
        for p in g.pairs where marginalContexts.contains(p.ctx) { byWord[p.word, default: []].append(p.logp) }
        var checked = 0
        for (w, v) in byWord where v.count == marginalContexts.count {
            guard let node = trie.node(for: w), trie.isWord(node) else { continue }
            let ref = v.reduce(0, +) / Double(v.count)
            XCTAssertEqual(Double(priors[Int(node)]), ref, accuracy: 0.15, "prior for \(w) (fp16 vs fp32)")
            checked += 1
        }
        XCTAssertGreaterThan(checked, 3)
        print("priors: \(words) words, \(checked) checked against the fp32 estimator")
    }
}
