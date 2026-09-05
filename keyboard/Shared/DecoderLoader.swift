import Foundation
import GlyphCore
import os

private let probeLog = Logger(subsystem: "com.edwardgao.glyph", category: "memory")

/// Memory available to this process before jetsam, in MB (iOS 13+).
func availableMemoryMB() -> Int { Int(os_proc_available_memory() / (1024 * 1024)) }

/// Append a line to Documents/diagnostics.log in this process's own data
/// container (no app group: personal teams cannot register one). Pulled from
/// the Mac with `xcrun devicectl device copy from --domain-type appDataContainer
///  --domain-identifier com.edwardgao.glyph.keyboard --source Documents/diagnostics.log`
/// (the extension) or `--domain-identifier com.edwardgao.glyph` (the app).
func diag(_ line: String) {
    probeLog.notice("\(line, privacy: .public)")
    guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("diagnostics.log")
    let stamp = ISO8601DateFormatter().string(from: Date())
    let data = ("\(stamp) \(line)\n").data(using: .utf8)!
    if let h = try? FileHandle(forWritingTo: url) { h.seekToEndOfFile(); h.write(data); try? h.close() }
    else { try? data.write(to: url) }
}

/// The models, trie, LM and tables ship once, in the containing app's bundle
/// (the app's SwipeRacer game decodes with the same stack). The extension lives
/// at App.app/PlugIns/SwipeKeyboardExt.appex and reads them from two levels up;
/// a bundle that carries its own copy (tests, older layouts) is used as is.
enum SharedResources {
    static let bundle: Bundle = {
        let own = Bundle(for: DecoderLoader.self)
        if own.url(forResource: "lexicon", withExtension: "bin") != nil { return own }
        if own.bundleURL.pathExtension == "appex" {
            let appURL = own.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
            if let app = Bundle(url: appURL), app.url(forResource: "lexicon", withExtension: "bin") != nil { return app }
        }
        return own
    }()
}

/// Loads the Core ML encoder and the trie once per process, off the main thread.
final class DecoderLoader {
    private let queue = DispatchQueue(label: "swipe.decoder.load", qos: .userInitiated)
    private var decoder: SwipeDecoder?
    private var error: Error?

    enum LoadError: Error, CustomStringConvertible {
        case missingResource(String)
        var description: String {
            switch self { case .missingResource(let n): return "missing \(n) in bundle" }
        }
    }

    private var lm: CoreMLLanguageModel?
    private var lmError: Error?
    /// Precomputed marginal log P(word) per trie node (`priors.bin`), NaN where absent.
    private(set) var priors: [Float]?

    /// The LM is only loaded when the extension has at least this much memory
    /// left (os_proc_available_memory); the probe in the host app measures what
    /// each variant really needs. Overridable for experiments via UserDefaults.
    static var lmMinAvailableMB: Int {
        let v = UserDefaults.standard.integer(forKey: "swipe.lm.minAvailMB")
        return v > 0 ? v : 60
    }

    enum LMSkipped: Error, CustomStringConvertible {
        case lowMemory(Int, Int)
        var description: String { switch self { case .lowMemory(let a, let need): return "skipped: \(a) MB free < \(need) MB" } }
    }

    func loadLM(_ done: @escaping (Result<CoreMLLanguageModel, Error>) -> Void) {
        queue.async {
            if let lm = self.lm { done(.success(lm)); return }
            if let e = self.lmError { done(.failure(e)); return }
            let avail = availableMemoryMB()
            // 0 = unknown (the simulator reports 0): do not gate on it.
            if avail > 0 && avail < Self.lmMinAvailableMB {
                let e = LMSkipped.lowMemory(avail, Self.lmMinAvailableMB)
                diag("LM \(e)")
                self.lmError = e
                done(.failure(e)); return
            }
            do {
                let bundle = SharedResources.bundle
                guard let modelURL = bundle.url(forResource: "SwipeLM", withExtension: "mlmodelc") else { throw LoadError.missingResource("SwipeLM.mlmodelc") }
                guard let metaURL = bundle.url(forResource: "lm_meta", withExtension: "json"),
                      let vocab = bundle.url(forResource: "vocab", withExtension: "json"),
                      let merges = bundle.url(forResource: "merges", withExtension: "txt") else { throw LoadError.missingResource("LM tokenizer/meta") }
                let meta = try JSONDecoder().decode(CoreMLLanguageModel.Meta.self, from: try Data(contentsOf: metaURL))
                let tok = try GPT2Tokenizer(vocabURL: vocab, mergesURL: merges)
                if let pURL = bundle.url(forResource: "priors", withExtension: "bin"),
                   let data = try? Data(contentsOf: pURL, options: .mappedIfSafe), data.count > 12,
                   data[0] == 0x53, data[1] == 0x57, data[2] == 0x50, data[3] == 0x52 {
                    let n = Int(data.loadLE(UInt32.self, at: 8))
                    self.priors = data.array(Float.self, at: 12, count: n)
                    diag("priors table: \(n) nodes")
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                // CPU only: Core ML keeps the fp16 weights memory-mapped (+~7 MB footprint
                // measured on iPhone 17); the GPU path copies them in (+200 MB) and jetsams the extension.
                let lm = try CoreMLLanguageModel(modelURL: modelURL, meta: meta, tokenizer: tok, computeUnits: .cpuOnly)
                // warm up once so the first swipe does not pay for it
                _ = try lm.score(pairs: [(ctx: "", word: "the"), (ctx: "i think", word: "the")])
                diag(String(format: "LM %@ %@ loaded in %.0f ms", meta.model, meta.quant, (CFAbsoluteTimeGetCurrent() - t0) * 1000))
                self.lm = lm
                done(.success(lm))
            } catch {
                self.lmError = error
                done(.failure(error))
            }
        }
    }

    func load(_ done: @escaping (Result<SwipeDecoder, Error>) -> Void) {
        queue.async {
            if let d = self.decoder { done(.success(d)); return }
            if let e = self.error { done(.failure(e)); return }
            do {
                let bundle = SharedResources.bundle
                let modelURL = bundle.url(forResource: "SwipeEncoder", withExtension: "mlmodelc") ?? URL(fileURLWithPath: "/nonexistent")
                guard let lexURL = bundle.url(forResource: "lexicon", withExtension: "bin") else {
                    throw LoadError.missingResource("lexicon.bin")
                }
                let trie = try Trie(contentsOf: lexURL)
                let d: SwipeDecoder
                if let encURL = bundle.url(forResource: "SwipeAREncoder", withExtension: "mlmodelc"),
                   let stepURL = bundle.url(forResource: "SwipeARStep", withExtension: "mlmodelc"),
                   let metaURL = bundle.url(forResource: "ar_meta", withExtension: "json"),
                   let meta = try? JSONDecoder().decode(ARFirstPass.Meta.self, from: Data(contentsOf: metaURL)) {
                    // The AR decoder (research #82b `ar_mixed_s1`): generalizes to real iPhone
                    // gestures far better than the CTC encoder. Ranking: ar + 0.6·uni + 1.2·len − 0.25·ilm
                    // (the freeze-five cell; fitted on FUTO, checked on the iPhone set).
                    let ar = try ARFirstPass(encoderURL: encURL, stepURL: stepURL, meta: meta, trie: trie,
                                             weights: .init(alpha: 0.6, beta: 1.2, lambda: 0.25), computeUnits: .cpuOnly)
                    if let iURL = bundle.url(forResource: "ilm", withExtension: "bin"),
                       let data = try? Data(contentsOf: iURL, options: .mappedIfSafe), data.count > 12,
                       data[0] == 0x53, data[1] == 0x57, data[2] == 0x49, data[3] == 0x4C {   // "SWIL"
                        let n = Int(data.loadLE(UInt32.self, at: 8))
                        ar.ilm = data.array(Float.self, at: 12, count: n)
                    } else { ar.weights.lambda = 0 }
                    d = SwipeDecoder(firstPass: ar)
                    diag("first pass: AR \(meta.checkpoint), beam \(meta.beam), ilm \(ar.ilm != nil)")
                } else {
                    let model = try CoreMLAcousticModel(contentsOf: modelURL, computeUnits: .cpuOnly)
                    d = SwipeDecoder(model: model, trie: trie)
                    diag("first pass: CTC runs/full")
                }
                self.decoder = d
                done(.success(d))
            } catch {
                self.error = error
                done(.failure(error))
            }
        }
    }
}
