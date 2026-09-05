import Foundation

/// Where race records go. The Cloudflare Worker (upload-worker/) is the default;
/// the LAN capture server (research/iphone/server.py) can be chosen behind the
/// game's gear icon. The upload token is not in the source: xcodegen writes
/// `GLYPH_UPLOAD_TOKEN` from the environment into Info.plist at generation
/// (deploy.sh exports it from research/iphone/.secrets). Without a token the
/// game still runs and records stay queued on the phone.
enum UploadConfig {
    static let productionURL = "https://swipe-upload.swipe-edwardgao.workers.dev/save"
    static let lanURL = "http://192.168.188.155:8765/save"
    static let token: String = (Bundle.main.object(forInfoDictionaryKey: "GlyphUploadToken") as? String) ?? ""
    static var defaultURL: String { productionURL }
    static var enabled: Bool { !token.isEmpty }
    /// Build stamp (deploy.sh: yyyymmddHHMM), to tell which build a phone runs.
    static let build: String = (Bundle.main.object(forInfoDictionaryKey: "GlyphBuild") as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "dev"
}
