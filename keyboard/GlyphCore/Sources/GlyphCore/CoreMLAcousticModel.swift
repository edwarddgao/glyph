#if canImport(CoreML)
import CoreML
import Foundation

/// The exported `SwipeEncoder` Core ML model as an `AcousticModel`.
public final class CoreMLAcousticModel: AcousticModel {
    let model: MLModel
    let input: MLMultiArray

    public enum ModelError: Error { case missingOutput }

    /// `url` points at a compiled `.mlmodelc` (what Xcode puts in the bundle)
    /// or an `.mlpackage`, which is compiled on the fly (tests on macOS).
    public init(contentsOf url: URL, computeUnits: MLComputeUnits = .cpuOnly) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        var target = url
        if url.pathExtension == "mlpackage" || url.pathExtension == "mlmodel" {
            target = try MLModel.compileModel(at: url)
        }
        model = try MLModel(contentsOf: target, configuration: cfg)
        input = try MLMultiArray(shape: [1, NSNumber(value: Features.nPoints), NSNumber(value: Features.nInput)],
                                 dataType: .float32)
    }

    public func logProbs(features: [Float]) throws -> [Float] {
        precondition(features.count == Features.nPoints * Features.nInput)
        // Honor the array's strides: Core ML pads the last axis on some paths.
        let inS1 = input.strides[1].intValue, inS2 = input.strides[2].intValue
        let inPtr = input.dataPointer.assumingMemoryBound(to: Float.self)
        for f in 0..<Features.nPoints {
            for c in 0..<Features.nInput { inPtr[f * inS1 + c * inS2] = features[f * Features.nInput + c] }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: input)])
        let out = try model.prediction(from: provider)
        guard let arr = out.featureValue(for: "log_probs")?.multiArrayValue else { throw ModelError.missingOutput }
        let stride = Geometry.nKeys + 1
        let n = Features.nPoints * stride
        precondition(arr.count == n && arr.dataType == .float32)
        // The output multiarray is (1, 64, 27) but its row stride is 32: read through strides.
        let s1 = arr.strides[1].intValue, s2 = arr.strides[2].intValue
        let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
        var result = [Float](repeating: 0, count: n)
        for f in 0..<Features.nPoints {
            for c in 0..<stride { result[f * stride + c] = p[f * s1 + c * s2] }
        }
        return result
    }
}
#endif
