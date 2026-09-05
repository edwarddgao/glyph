import Foundation

/// Anything that maps a (64, 32) feature tensor to (64, 27) CTC log-probs.
public protocol AcousticModel {
    func logProbs(features: [Float]) throws -> [Float]
}

/// Anything that turns a gesture into ranked word candidates (the first pass).
public protocol FirstPass: AnyObject {
    var trie: Trie { get }
    func candidates(_ samples: [TouchSample]) throws -> [Candidate]
}

/// CTC encoder + trie-constrained CTC prefix beam.
public final class CTCFirstPass: FirstPass {
    public let model: AcousticModel
    public let trie: Trie
    public var config: BeamConfig
    public init(model: AcousticModel, trie: Trie, config: BeamConfig = BeamConfig()) {
        self.model = model; self.trie = trie; self.config = config
    }
    public func candidates(_ samples: [TouchSample]) throws -> [Candidate] {
        let x = Features.encode(samples)
        let lp = try model.logProbs(features: x)
        return CTCBeam.search(logProbs: lp, frames: Features.nPoints, trie: trie, config: config)
    }
}

/// Gesture -> ranked word candidates through whichever first pass is loaded.
public final class SwipeDecoder {
    public let firstPass: FirstPass
    public var trie: Trie { firstPass.trie }

    public init(firstPass: FirstPass) { self.firstPass = firstPass }

    /// CTC convenience (the original constructor).
    public convenience init(model: AcousticModel, trie: Trie, config: BeamConfig = BeamConfig()) {
        self.init(firstPass: CTCFirstPass(model: model, trie: trie, config: config))
    }

    public func decode(_ samples: [TouchSample]) throws -> [Candidate] { try firstPass.candidates(samples) }

    /// Greedy CTC collapse of the argmax path — the lexicon-free fallback.
    public static func greedy(logProbs: [Float], frames: Int) -> String {
        let stride = Geometry.nKeys + 1
        var prev = -1
        var out = [Character]()
        for f in 0..<frames {
            var best = 0
            var bestV = -Float.infinity
            for k in 0..<stride where logProbs[f * stride + k] > bestV { bestV = logProbs[f * stride + k]; best = k }
            if best != prev && best != Geometry.nKeys { out.append(Geometry.alphabet[best]) }
            prev = best
        }
        return String(out)
    }
}
