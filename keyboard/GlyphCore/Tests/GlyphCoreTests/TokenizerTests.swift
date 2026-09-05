import XCTest
@testable import GlyphCore

final class TokenizerTests: XCTestCase {
    struct Case: Decodable { let text: String; let ids: [Int] }

    func testMatchesHuggingFace() throws {
        let res = GoldenTests.resources.appendingPathComponent("gpt2")
        let tok = try GPT2Tokenizer(vocabURL: res.appendingPathComponent("vocab.json"),
                                    mergesURL: res.appendingPathComponent("merges.txt"))
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: res.appendingPathComponent("tokenizer_goldens.json")))
        var bad = 0
        for c in cases where tok.encode(c.text) != c.ids {
            bad += 1
            if bad <= 5 { print("mismatch: \(c.text.debugDescription) got \(tok.encode(c.text)) want \(c.ids)") }
        }
        XCTAssertEqual(bad, 0, "\(bad) of \(cases.count) tokenizations differ from HF")
        XCTAssertEqual(tok.encodeWord("the"), [262])
        let t0 = Date()
        for c in cases { _ = tok.encode(c.text) }
        print(String(format: "tokenizer: %.1f µs/case (cached)", Date().timeIntervalSince(t0) * 1e6 / Double(cases.count)))
    }
}
