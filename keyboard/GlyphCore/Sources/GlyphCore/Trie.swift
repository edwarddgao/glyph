import Foundation

/// The lexicon trie, loaded from `lexicon.bin` (see `tools/export.py`).
///
/// Nodes are stored breadth-first with each node's children contiguous and
/// letter-sorted, so a child lookup is a scan of at most 26 entries and a
/// prefix is identified by its node id alone. Node 0 is the root.
public final class Trie {
    public let count: Int
    let letter: [UInt8]
    let flags: [UInt8]
    let childStart: [Int32]
    let childCount: [UInt8]
    let logProb: [Float]
    let parent: [Int32]

    public enum LoadError: Error { case badMagic, badVersion, truncated }

    public convenience init(contentsOf url: URL) throws {
        try self.init(data: try Data(contentsOf: url, options: .mappedIfSafe))
    }

    public init(data: Data) throws {
        guard data.count >= 12, data[0] == 0x53, data[1] == 0x57, data[2] == 0x54, data[3] == 0x52 else {
            throw LoadError.badMagic
        }
        let version = data.loadLE(UInt32.self, at: 4)
        guard version == 2 else { throw LoadError.badVersion }
        let n = Int(data.loadLE(UInt32.self, at: 8))
        let expected = 12 + n * (1 + 1 + 4 + 1 + 4 + 4)
        guard data.count >= expected else { throw LoadError.truncated }
        var off = 12
        letter = data.array(UInt8.self, at: off, count: n); off += n
        flags = data.array(UInt8.self, at: off, count: n); off += n
        childStart = data.array(Int32.self, at: off, count: n); off += 4 * n
        childCount = data.array(UInt8.self, at: off, count: n); off += n
        logProb = data.array(Float.self, at: off, count: n); off += 4 * n
        parent = data.array(Int32.self, at: off, count: n)
        count = n
    }

    /// Child of `node` along letter index `k` (0 = 'a'), or -1.
    @inline(__always)
    public func child(of node: Int32, letterIndex k: Int) -> Int32 {
        let c = Int(childCount[Int(node)])
        if c == 0 { return -1 }
        let start = Int(childStart[Int(node)])
        let target = UInt8(97 + k)
        for i in start..<(start + c) {
            let l = letter[i]
            if l == target { return Int32(i) }
            if l > target { return -1 }
        }
        return -1
    }

    @inline(__always) public func hasChildren(_ node: Int32) -> Bool { childCount[Int(node)] != 0 }
    @inline(__always) public func isWord(_ node: Int32) -> Bool { flags[Int(node)] & 1 != 0 }
    @inline(__always) public func logp(_ node: Int32) -> Double { Double(logProb[Int(node)]) }
    /// Letter index (0…25) of the edge into `node`; undefined for the root.
    @inline(__always) public func letterIndex(_ node: Int32) -> Int { Int(letter[Int(node)]) - 97 }

    public func depth(_ node: Int32) -> Int {
        var d = 0, n = node
        while n > 0 { n = parent[Int(n)]; d += 1 }
        return d
    }

    /// The prefix spelled by the path from the root to `node`.
    public func word(at node: Int32) -> String {
        var bytes = [UInt8](), n = node
        while n > 0 { bytes.append(letter[Int(n)]); n = parent[Int(n)] }
        bytes.reverse()
        return String(decoding: bytes, as: UTF8.self)
    }

    public func node(for word: String) -> Int32? {
        var n: Int32 = 0
        for ch in word {
            guard let k = Geometry.index(of: ch) else { return nil }
            n = child(of: n, letterIndex: k)
            if n < 0 { return nil }
        }
        return n
    }

    public func contains(_ word: String) -> Bool {
        guard let n = node(for: word) else { return false }
        return isWord(n)
    }
}

public extension Data {
    func loadLE<T: FixedWidthInteger>(_: T.Type, at offset: Int) -> T {
        var v: T = 0
        _ = Swift.withUnsafeMutableBytes(of: &v) { copyBytes(to: $0, from: offset..<(offset + MemoryLayout<T>.size)) }
        return T(littleEndian: v)
    }

    /// Copy `count` little-endian elements starting at byte `offset`.
    func array<T>(_: T.Type, at offset: Int, count: Int) -> [T] {
        let bytes = count * MemoryLayout<T>.stride
        return [T](unsafeUninitializedCapacity: count) { buf, initialized in
            _ = copyBytes(to: UnsafeMutableRawBufferPointer(buf), from: offset..<(offset + bytes))
            initialized = count
        }
    }
}
