import XCTest
@testable import GlyphCore

/// Every number here was produced by the research code (`tools/export.py`):
/// raw capture gestures -> features -> encoder log-probs -> beam candidates.
final class GoldenTests: XCTestCase {
    struct Golden: Decodable {
        struct Item: Decodable {
            struct Cand: Decodable { let word: String; let score, acoustic, unigram: Double; let length: Int }
            let word: String
            let x, y, t: [Double]
            let features: [[Float]]
            let log_probs: [[Float]]
            let candidates: [Cand]
            var samples: [TouchSample] { (0..<x.count).map { TouchSample(x: x[$0], y: y[$0], t: t[$0]) } }
        }
        struct Beam: Decodable { let width: Int; let prune, alpha, beta: Double }
        let alphabet: String
        let beam: Beam
        let items: [Item]
    }

    static let resources = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("Resources")

    static let golden: Golden = {
        let data = try! Data(contentsOf: resources.appendingPathComponent("goldens.json"))
        return try! JSONDecoder().decode(Golden.self, from: data)
    }()
    static let trie: Trie = try! Trie(contentsOf: resources.appendingPathComponent("lexicon.bin"))
    static var config: BeamConfig {
        var c = BeamConfig()
        c.beamWidth = golden.beam.width; c.pruneLogP = golden.beam.prune
        c.alpha = golden.beam.alpha; c.beta = golden.beam.beta
        return c
    }

    func testGeometryMatchesLayoutPy() {
        let a = Geometry.center(of: "a")!
        XCTAssertEqual(a.x, 0.1, accuracy: 1e-9); XCTAssertEqual(a.y, 0.5, accuracy: 1e-9)
        let q = Geometry.center(of: "q")!
        XCTAssertEqual(q.x, 0.05, accuracy: 1e-9); XCTAssertEqual(q.y, 1.0 / 6.0, accuracy: 1e-9)
        let m = Geometry.center(of: "m")!
        XCTAssertEqual(m.x, 0.15 + 0.65, accuracy: 1e-9); XCTAssertEqual(m.y, 5.0 / 6.0, accuracy: 1e-9)
        XCTAssertEqual(Geometry.key(atX: 0.1, y: 0.5), "a")
        XCTAssertEqual(Geometry.key(atX: 0.05, y: 0.9), nil)   // shift zone
        XCTAssertEqual(Geometry.key(atX: 0.2, y: 0.9), "z")
    }

    func testTrieLoadsAndLooksUp() {
        let t = Self.trie
        XCTAssertGreaterThan(t.count, 600_000)
        XCTAssertTrue(t.contains("the"))
        XCTAssertTrue(t.contains("hello"))
        XCTAssertFalse(t.contains("zzzzq"))
        XCTAssertFalse(t.contains("downstai"))  // prefix, not a word
        XCTAssertTrue(t.node(for: "downstai") != nil)
        let n = t.node(for: "hello")!
        XCTAssertEqual(t.word(at: n), "hello")
        XCTAssertEqual(t.depth(n), 5)
        XCTAssertLessThan(t.logp(t.node(for: "hello")!), t.logp(t.node(for: "the")!))
    }

    func testFeaturesMatchPython() {
        for item in Self.golden.items {
            let got = Features.encode(item.samples)
            XCTAssertEqual(got.count, Features.nPoints * Features.nInput)
            for f in 0..<Features.nPoints {
                for c in 0..<Features.nInput {
                    let want = Double(item.features[f][c]), g = Double(got[f * Features.nInput + c])
                    // affinity in [0,1]; kinematics are float32-noisy at large magnitude
                    let tol = c < Geometry.nKeys ? 2e-5 : max(2e-3, 1e-3 * abs(want))
                    XCTAssertEqual(g, want, accuracy: tol,
                                   "word \(item.word) frame \(f) channel \(c)")
                }
            }
        }
    }

    func testBeamMatchesPythonOnPythonLogProbs() {
        for item in Self.golden.items {
            let lp = item.log_probs.flatMap { $0 }
            let got = CTCBeam.search(logProbs: lp, frames: Features.nPoints, trie: Self.trie, config: Self.config)
            let want = item.candidates
            if want.isEmpty { XCTAssertTrue(got.isEmpty, item.word); continue }
            XCTAssertFalse(got.isEmpty, item.word)
            XCTAssertEqual(got.first?.word, want.first?.word, "top-1 for \(item.word)")
            for (g, w) in zip(got.prefix(8), want) {
                XCTAssertEqual(g.word, w.word, "rank order for \(item.word)")
                XCTAssertEqual(g.acoustic, w.acoustic, accuracy: 1e-6, g.word)
                XCTAssertEqual(g.unigram, w.unigram, accuracy: 1e-6, g.word)
                XCTAssertEqual(g.score, w.score, accuracy: 1e-5, g.word)
                XCTAssertEqual(g.length, w.length)
            }
        }
    }

    #if canImport(CoreML)
    func testCoreMLModelMatchesTorch() throws {
        let model = try CoreMLAcousticModel(contentsOf: Self.resources.appendingPathComponent("SwipeEncoder.mlpackage"))
        for item in Self.golden.items {
            let x = item.features.flatMap { $0 }
            let got = try model.logProbs(features: x)
            let want = item.log_probs.flatMap { $0 }
            var maxDiff: Float = 0
            for i in 0..<want.count { maxDiff = max(maxDiff, abs(got[i] - want[i])) }
            XCTAssertLessThan(maxDiff, 2e-3, "log-probs for \(item.word)")
        }
    }

    func testEndToEndTop1MatchesPython() throws {
        let model = try CoreMLAcousticModel(contentsOf: Self.resources.appendingPathComponent("SwipeEncoder.mlpackage"))
        let decoder = SwipeDecoder(model: model, trie: Self.trie, config: Self.config)
        var agree = 0, correct = 0
        for item in Self.golden.items {
            let got = try decoder.decode(item.samples)
            let want = item.candidates.first?.word
            if got.first?.word == want { agree += 1 }
            if got.first?.word == item.word { correct += 1 }
            XCTAssertEqual(got.first?.word, want, "end-to-end top-1 for \(item.word)")
        }
        print("end-to-end: agree with python \(agree)/\(Self.golden.items.count), correct \(correct)")
        let t0 = Date()
        for item in Self.golden.items { _ = try decoder.decode(item.samples) }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(Self.golden.items.count)
        print(String(format: "decode latency %.1f ms/word (cpu, mac)", ms))
    }
    #endif
}
