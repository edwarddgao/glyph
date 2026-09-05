import XCTest
@testable import GlyphCore

/// The geometric trace cost against `GestureDP.word_cost` (research
/// `geomllm.py`, label-filter config) on race and capture gestures:
/// `Resources/trace_goldens.json`, written by the snippet in the research log.
final class TraceTests: XCTestCase {
    struct Item: Decodable { let word: String; let x, y, t: [Double]; let costs: [String: Double] }
    struct File: Decodable { let scale: [Double]; let centers: [String: [Double]]; let items: [Item] }

    static let file: File = {
        let url = GoldenTests.resources.appendingPathComponent("trace_goldens.json")
        return try! JSONDecoder().decode(File.self, from: Data(contentsOf: url))
    }()

    func testKeyGeometryMatchesPython() {
        let f = Self.file
        XCTAssertEqual(f.scale[0], Geometry.radiusX, accuracy: 1e-6)
        XCTAssertEqual(f.scale[1], Geometry.radiusY, accuracy: 1e-6)
        let tr = GestureTrace(samples: [TouchSample(x: 0.1, y: 0.5, t: 0), TouchSample(x: 0.2, y: 0.5, t: 10)])
        for (ch, c) in f.centers {
            let k = Geometry.alphabet.firstIndex(of: ch.first!)!
            XCTAssertEqual(tr.keys[k].x * Geometry.radiusX, c[0], accuracy: 1e-6, "center x of \(ch)")
            XCTAssertEqual(tr.keys[k].y * Geometry.radiusY, c[1], accuracy: 1e-6, "center y of \(ch)")
        }
    }

    func testWordCostsMatchPython() {
        var n = 0, worst = 0.0
        for it in Self.file.items {
            let samples = zip(zip(it.x, it.y), it.t).map { TouchSample(x: $0.0, y: $0.1, t: $1) }
            let tr = GestureTrace(samples: samples)
            for (w, ref) in it.costs {
                let got = tr.cost(of: w)
                let tol = 1e-3 * max(1, abs(ref))
                worst = max(worst, abs(got - ref) / max(1, abs(ref)))
                XCTAssertEqual(got, ref, accuracy: tol, "cost of \(w) for gesture labeled \(it.word)")
                n += 1
            }
        }
        print(String(format: "trace goldens: %d costs, worst relative error %.2e", n, worst))
        XCTAssertGreaterThan(n, 60)
    }

    func testVerdictsOnRaceGestures() {
        // the race gestures: every prompted word except the swipe of the wrong word
        // ("sus" traced while "hes" was prompted, cost/letter 30) is traced
        var traced = 0, rejected: [String] = []
        for it in Self.file.items.prefix(35) {
            let samples = zip(zip(it.x, it.y), it.t).map { TouchSample(x: $0.0, y: $0.1, t: $1) }
            let tr = GestureTrace(samples: samples)
            if let why = tr.rejection(for: it.word) { rejected.append("\(it.word):\(why)") } else { traced += 1 }
        }
        XCTAssertEqual(rejected.count, 1, "\(rejected)")
        XCTAssertGreaterThanOrEqual(traced, 34)
    }
}
