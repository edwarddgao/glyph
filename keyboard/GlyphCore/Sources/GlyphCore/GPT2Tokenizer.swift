import Foundation

/// GPT-2 byte-level BPE tokenizer (the `gpt2` / `distilgpt2` vocabulary),
/// ported from Hugging Face's `GPT2Tokenizer`. Encoding only.
///
/// Loads `vocab.json` (token string -> id) and `merges.txt`. Text is split by
/// GPT-2's regex into pretokens, each pretoken's UTF-8 bytes are mapped to the
/// printable byte alphabet, then merges are applied by rank.
public final class GPT2Tokenizer {
    public let encoder: [String: Int]
    let ranks: [String: Int]          // "a b" -> merge rank
    let byteEncoder: [UInt8: Character]
    public let eos = 50256

    public init(vocabURL: URL, mergesURL: URL) throws {
        let vocab = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL)) as! [String: Int]
        encoder = vocab
        var r: [String: Int] = [:]
        let lines = try String(contentsOf: mergesURL, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
        var rank = 0
        for line in lines {
            if line.hasPrefix("#version") || line.isEmpty { continue }
            r[String(line)] = rank
            rank += 1
        }
        ranks = r
        byteEncoder = GPT2Tokenizer.makeByteEncoder()
    }

    /// bytes_to_unicode(): printable bytes map to themselves, the rest to 256+.
    static func makeByteEncoder() -> [UInt8: Character] {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var out: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) { out[UInt8(b)] = Character(UnicodeScalar(c)!) }
        return out
    }

    // GPT-2's pretokenization regex:
    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    private static let pattern = try! NSRegularExpression(
        pattern: #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#)

    private var cache: [String: [Int]] = [:]

    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        let ns = text as NSString
        for m in Self.pattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let piece = ns.substring(with: m.range)
            out.append(contentsOf: bpe(piece))
        }
        return out
    }

    /// Token ids for a word appended to running text (GPT-2's leading space).
    public func encodeWord(_ word: String) -> [Int] { encode(" " + word) }

    private func bpe(_ piece: String) -> [Int] {
        if let c = cache[piece] { return c }
        // bytes -> unicode chars
        var word: [String] = Array(piece.utf8).map { String(byteEncoder[$0]!) }
        if word.count > 1 {
            while true {
                var best: (rank: Int, i: Int)? = nil
                for i in 0..<(word.count - 1) {
                    if let r = ranks[word[i] + " " + word[i + 1]], best == nil || r < best!.rank { best = (r, i) }
                }
                guard let (_, _) = best else { break }
                let first = word[best!.i], second = word[best!.i + 1]
                var merged: [String] = []
                var i = 0
                while i < word.count {
                    if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                        merged.append(first + second); i += 2
                    } else {
                        merged.append(word[i]); i += 1
                    }
                }
                word = merged
                if word.count == 1 { break }
            }
        }
        let ids = word.map { encoder[$0] ?? eos }
        if cache.count > 20_000 { cache.removeAll() }
        cache[piece] = ids
        return ids
    }
}
