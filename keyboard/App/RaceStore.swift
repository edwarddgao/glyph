import Foundation
import UIKit

/// Race records: one JSON per finished sentence (kind "race"), queued on disk
/// and posted to the capture server (`research/iphone/server.py`), so a player
/// with no server in reach loses nothing — pending files upload on the next
/// race. `research/iphone/race_to_capture.py` turns them into capture-shaped
/// files for the research tools.
final class RaceStore {
    static let shared = RaceStore()

    /// Anonymous, per-install; never tied to an account.
    let session: String
    var server: String {
        get { UserDefaults.standard.string(forKey: "record.server") ?? UploadConfig.defaultURL }
    }
    private let dir: URL
    private(set) var uploaded = 0
    private(set) var failed = 0
    var onStatus: ((String) -> Void)?

    private init() {
        let d = UserDefaults.standard
        if let s = d.string(forKey: "race.session") { session = s }
        else { session = "r" + UUID().uuidString.prefix(8).lowercased(); d.set(session, forKey: "race.session") }
        dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("race_queue")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    var pendingCount: Int { (try? FileManager.default.contentsOfDirectory(atPath: dir.path).count) ?? 0 }

    func save(_ record: [String: Any]) {
        let ts = record["ts"] as? Int ?? Int(Date().timeIntervalSince1970 * 1000)
        let url = dir.appendingPathComponent("race_\(session)_\(ts).json")
        if let data = try? JSONSerialization.data(withJSONObject: record) { try? data.write(to: url) }
        flush()
    }

    /// Upload every queued file; delete each on a 200.
    func flush() {
        guard UploadConfig.enabled || server != UploadConfig.productionURL else { return }   // no token: keep records queued
        guard let url = URL(string: server) else { onStatus?("bad server URL"); return }
        let files = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { return }
        for f in files {
            guard let body = try? Data(contentsOf: f) else { continue }
            var req = URLRequest(url: url, timeoutInterval: 15)
            req.httpMethod = "POST"; req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue("Bearer \(UploadConfig.token)", forHTTPHeaderField: "Authorization")   // the LAN server ignores it
            req.httpBody = body
            URLSession.shared.dataTask(with: req) { [weak self] _, resp, err in
                guard let self else { return }
                let ok = (resp as? HTTPURLResponse)?.statusCode == 200 && err == nil
                DispatchQueue.main.async {
                    if ok { try? FileManager.default.removeItem(at: f); self.uploaded += 1 }
                    else { self.failed += 1 }
                    let pending = self.pendingCount
                    self.onStatus?(ok ? (pending == 0 ? "uploaded ✓" : "uploaded ✓ (\(pending) pending)")
                                      : "upload failed — \(pending) saved on this phone, retried next session")
                }
            }.resume()
        }
    }

    static var deviceModel: String {
        var sys = utsname(); uname(&sys)
        return withUnsafePointer(to: &sys.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) } }
    }
}
