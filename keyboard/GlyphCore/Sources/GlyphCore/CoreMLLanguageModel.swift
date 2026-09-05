#if canImport(CoreML)
import CoreML
import Foundation

/// The exported `SwipeLM` Core ML model (see `tools/export_lm.py`) as a
/// `LanguageModel`. Fixed shapes: B sequences of L tokens, P read positions.
///
/// Scoring follows `capture/fused_rescore.py`: the sequence is BOS + context
/// tokens + the word's tokens, the word carrying GPT-2's leading space only
/// when there is a context; log P(word | ctx) is the sum over the word's tokens.
public final class CoreMLLanguageModel: LanguageModel {
    public let tokenizer: GPT2Tokenizer
    let model: MLModel
    public let batch: Int, length: Int, positions: Int
    let bos: Int, eos: Int
    let ids: MLMultiArray, pos: MLMultiArray, tgt: MLMultiArray
    private var wordTokens: [String: [Int]] = [:]

    public struct Meta: Decodable { public let B, L, P, eos, bos: Int; public let model, quant: String }

    public init(modelURL: URL, meta: Meta, tokenizer: GPT2Tokenizer,
                computeUnits: MLComputeUnits = .all) throws {
        var target = modelURL
        if modelURL.pathExtension == "mlpackage" { target = try MLModel.compileModel(at: modelURL) }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        model = try MLModel(contentsOf: target, configuration: cfg)
        self.tokenizer = tokenizer
        batch = meta.B; length = meta.L; positions = meta.P; bos = meta.bos; eos = meta.eos
        ids = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: length)], dataType: .int32)
        pos = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: positions)], dataType: .int32)
        tgt = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: positions)], dataType: .int32)
    }

    private func tokens(forWord w: String, leadingSpace: Bool) -> [Int] {
        let key = (leadingSpace ? " " : "") + w
        if let t = wordTokens[key] { return t }
        let t = tokenizer.encode(key)
        wordTokens[key] = t
        return t
    }

    public func score(pairs: [(ctx: String, word: String)]) throws -> [Double] {
        var out = [Double](repeating: 0, count: pairs.count)
        var start = 0
        let idsP = ids.dataPointer.assumingMemoryBound(to: Int32.self)
        let posP = pos.dataPointer.assumingMemoryBound(to: Int32.self)
        let tgtP = tgt.dataPointer.assumingMemoryBound(to: Int32.self)
        let idsS = ids.strides[0].intValue, idsS1 = ids.strides[1].intValue
        let posS = pos.strides[0].intValue, posS1 = pos.strides[1].intValue
        while start < pairs.count {
            let chunk = Array(pairs[start..<min(start + batch, pairs.count)])
            // fill
            for b in 0..<batch {
                for l in 0..<length { idsP[b * idsS + l * idsS1] = Int32(eos) }
                for p in 0..<positions { posP[b * posS + p * posS1] = 0; tgtP[b * posS + p * posS1] = 0 }
            }
            var counts = [Int](repeating: 0, count: chunk.count)
            for (b, pair) in chunk.enumerated() {
                var ctx = [bos] + (pair.ctx.isEmpty ? [] : tokenizer.encode(pair.ctx))
                let cont = tokens(forWord: pair.word, leadingSpace: !pair.ctx.isEmpty)
                if ctx.count + cont.count > length { ctx = Array(ctx.suffix(length - cont.count)) }
                let seq = ctx + cont
                for (l, t) in seq.enumerated() { idsP[b * idsS + l * idsS1] = Int32(t) }
                let n = min(cont.count, positions)
                for j in 0..<n {
                    posP[b * posS + j * posS1] = Int32(ctx.count - 1 + j)
                    tgtP[b * posS + j * posS1] = Int32(cont[j])
                }
                counts[b] = n
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "ids": MLFeatureValue(multiArray: ids), "pos": MLFeatureValue(multiArray: pos), "tgt": MLFeatureValue(multiArray: tgt)])
            let result = try model.prediction(from: provider)
            guard let lp = result.featureValue(for: "logp")?.multiArrayValue else { throw CoreMLAcousticModel.ModelError.missingOutput }
            let s0 = lp.strides[0].intValue, s1 = lp.strides[1].intValue
            let lpP = lp.dataPointer.assumingMemoryBound(to: Float.self)
            for b in 0..<chunk.count {
                var total = 0.0
                for j in 0..<counts[b] { total += Double(lpP[b * s0 + j * s1]) }
                out[start + b] = total
            }
            start += batch
        }
        return out
    }
}
#endif
