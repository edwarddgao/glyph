import XCTest
@testable import GlyphCore

#if canImport(CoreML)
final class ARTests: XCTestCase {
    struct Golden: Decodable {
        struct Cand: Decodable { let word: String; let ar, unigram: Double; let length: Int }
        struct Item: Decodable { let word: String; let x, y, t: [Double]; let features: [[Float]]; let memory_checksum: Double; let candidates: [Cand]
            var samples: [TouchSample] { (0..<x.count).map { TouchSample(x: x[$0], y: y[$0], t: t[$0]) } } }
        let beam: Int; let items: [Item]
    }

    func testARBeamMatchesPython() throws {
        let res = GoldenTests.resources
        let g = try JSONDecoder().decode(Golden.self, from: try Data(contentsOf: res.appendingPathComponent("ar_goldens.json")))
        let meta = try JSONDecoder().decode(ARFirstPass.Meta.self, from: try Data(contentsOf: res.appendingPathComponent("ar_meta.json")))
        let ar = try ARFirstPass(encoderURL: res.appendingPathComponent("SwipeAREncoder.mlpackage"),
                                 stepURL: res.appendingPathComponent("SwipeARStep.mlpackage"), meta: meta, trie: GoldenTests.trie)
        var top1Agree = 0, top8Agree = 0, correct = 0, scoreErr = 0.0
        let t0 = Date()
        for item in g.items {
            let got = try ar.candidates(item.samples)
            // compare by raw AR score ordering, the quantity Python's ar_beam returns
            let gotByAR = got.sorted { $0.acoustic > $1.acoustic }
            let want = item.candidates
            XCTAssertFalse(gotByAR.isEmpty, item.word)
            if gotByAR.first?.word == want.first?.word { top1Agree += 1 }
            let g8 = Set(gotByAR.prefix(8).map { $0.word }), w8 = Set(want.prefix(8).map { $0.word })
            top8Agree += g8.intersection(w8).count
            if gotByAR.first?.word == item.word { correct += 1 }
            let wantMap = Dictionary(want.map { ($0.word, $0) }, uniquingKeysWith: { a, _ in a })
            for c in gotByAR.prefix(8) {
                if let w = wantMap[c.word] {
                    scoreErr = max(scoreErr, abs(c.acoustic - w.ar))
                    XCTAssertEqual(c.unigram, w.unigram, accuracy: 1e-6); XCTAssertEqual(c.length, w.length)
                }
            }
        }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(g.items.count)
        print(String(format: "AR beam: top-1 agrees with python %d/%d, top-8 overlap %d/%d, max |Δar| %.4f, correct %d, %.0f ms/word (mac cpu)",
                     top1Agree, g.items.count, top8Agree, 8 * g.items.count, scoreErr, correct, ms))
        XCTAssertGreaterThanOrEqual(top1Agree, g.items.count - 1, "AR top-1 vs python")
        XCTAssertLessThan(scoreErr, 2e-3)
    }
}
#endif
