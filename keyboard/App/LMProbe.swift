import CoreML
import Foundation
import os

/// On-device memory/latency probe for the LM variants, run inside the host
/// app (which has no keyboard-extension memory ceiling) and read back from
/// the Mac:
///
///   xcrun devicectl device process launch --device <udid> --terminate-existing \
///       com.edwardgao.glyph --lm-probe
///   xcrun devicectl device copy from --device <udid> --domain-type appDataContainer \
///       --domain-identifier com.edwardgao.glyph --source Documents/lm_probe.txt --destination -
///
/// For every `ProbeModels/*.mlmodelc` in the app bundle and every compute-unit
/// option: phys_footprint before load, after load, after one prediction, and
/// the steady-state prediction latency.
enum LMProbe {
    static var isRequested: Bool { CommandLine.arguments.contains("--lm-probe") }

    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : -1
    }

    static func run(report: @escaping (String) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            var lines: [String] = ["probe \(ISO8601DateFormatter().string(from: Date()))",
                                   String(format: "baseline footprint %.0f MB, available %.0f MB", footprintMB(), Double(os_proc_available_memory()) / 1e6)]
            func emit(_ s: String) { lines.append(s); DispatchQueue.main.async { report(lines.joined(separator: "\n")) } }
            let bundle = Bundle.main
            let models = ((try? FileManager.default.contentsOfDirectory(at: bundle.bundleURL, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "mlmodelc" && $0.lastPathComponent.hasPrefix("LM_") }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            if models.isEmpty { emit("no LM_*.mlmodelc in bundle") }
            let units: [(String, MLComputeUnits)] = [("ane", .cpuAndNeuralEngine), ("cpu", .cpuOnly), ("gpu", .cpuAndGPU), ("all", .all)]
            for url in models {
                let name = url.deletingPathExtension().lastPathComponent
                for (uname, u) in units {
                    autoreleasepool {
                        let before = footprintMB()
                        let cfg = MLModelConfiguration(); cfg.computeUnits = u
                        let t0 = CFAbsoluteTimeGetCurrent()
                        guard let model = try? MLModel(contentsOf: url, configuration: cfg) else {
                            emit("\(name) \(uname): load failed"); return
                        }
                        let loadMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                        let afterLoad = footprintMB()
                        // one prediction with the model's declared shapes
                        guard let desc = model.modelDescription.inputDescriptionsByName["ids"]?.multiArrayConstraint,
                              let pdesc = model.modelDescription.inputDescriptionsByName["pos"]?.multiArrayConstraint else {
                            emit("\(name) \(uname): unexpected inputs"); return
                        }
                        let ids = try! MLMultiArray(shape: desc.shape, dataType: .int32)
                        let pos = try! MLMultiArray(shape: pdesc.shape, dataType: .int32)
                        let tgt = try! MLMultiArray(shape: pdesc.shape, dataType: .int32)
                        for i in 0..<ids.count { ids[i] = 50256 }
                        for i in 0..<pos.count { pos[i] = NSNumber(value: i % 4); tgt[i] = 262 }
                        let provider = try! MLDictionaryFeatureProvider(dictionary: ["ids": MLFeatureValue(multiArray: ids), "pos": MLFeatureValue(multiArray: pos), "tgt": MLFeatureValue(multiArray: tgt)])
                        let t1 = CFAbsoluteTimeGetCurrent()
                        guard (try? model.prediction(from: provider)) != nil else { emit("\(name) \(uname): predict failed"); return }
                        let firstMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
                        let afterPredict = footprintMB()
                        let t2 = CFAbsoluteTimeGetCurrent()
                        for _ in 0..<5 { _ = try? model.prediction(from: provider) }
                        let steadyMs = (CFAbsoluteTimeGetCurrent() - t2) * 200
                        emit(String(format: "%@ %@: footprint %.0f -> %.0f (load) -> %.0f MB (predict); load %.0f ms, first %.0f ms, steady %.0f ms/batch",
                                    name, uname, before, afterLoad, afterPredict, loadMs, firstMs, steadyMs))
                    }
                    Thread.sleep(forTimeInterval: 0.5)
                }
            }
            emit("done")
            if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                try? lines.joined(separator: "\n").write(to: dir.appendingPathComponent("lm_probe.txt"), atomically: true, encoding: .utf8)
            }
        }
    }
}
