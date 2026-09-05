import Foundation

/// Decoder-independent "does this gesture trace this word?" cost — a port of
/// `GestureDP.word_cost` in `research/src/swipe_typing/geomllm.py` with the
/// label filter's configuration (#81: `GeomConfig(time_weight=1.25)`, cost per
/// letter above 6.0 = untraced).
///
/// The gesture is resampled to 96 points along its arclength and expressed in
/// key half-extents; a dynamic program aligns the word's letters to landing
/// points along the path, charging each point either a landing cost (distance
/// to the letter's key) or a transit cost (distance to the segment between the
/// previous and next key, weighted by how long the finger dwelt there). The
/// result does not depend on the lexicon, the encoder or the language model, so
/// SwipeRacer uses it to judge a swipe against the prompted word.
public struct GestureTrace {
    public struct Config {
        public var nPoints = 96
        public var sigmaKey = 1.0
        public var sigmaTransit = 2.0
        public var wTransit = 0.3
        public var timeWeight = 1.25
        public init() {}
    }
    /// #81's thresholds: per-letter alignment cost, and the "aborted" rule
    /// (a word of 4+ letters whose path is shorter than half a key width per
    /// letter transition).
    public static let untracedCostPerLetter = 6.0

    let cfg: Config
    let n: Int
    let g: [SIMD2<Double>]          // resampled gesture, key units
    let keys: [SIMD2<Double>]       // 26 key centers, key units
    let tw: [Double]                // per-point dwell weight
    let land: [[Double]]            // [k][j]
    let hover: [[Double]]           // [k][j]
    /// Path length in key widths (2 × the x half-extent), from the raw samples.
    public let pathKeyWidths: Double

    public init(samples: [TouchSample], config: Config = Config()) {
        cfg = config
        n = cfg.nPoints
        let pts = samples.map { SIMD2(Double($0.x), Double($0.y)) }
        let t = samples.map { Double($0.t) }
        let scale = SIMD2(Geometry.radiusX, Geometry.radiusY)
        // arclength parameter
        var u = [0.0]
        for i in 1..<max(pts.count, 1) { u.append(u[i - 1] + (pts[i] - pts[i - 1]).length) }
        pathKeyWidths = (u.last ?? 0) / (2 * Geometry.radiusX)
        let xy: [SIMD2<Double>] = pts.count <= 1 || (u.last ?? 0) <= 0
            ? [SIMD2<Double>](repeating: pts.first ?? .zero, count: n)
            : Features.resample(pts, t: u, n: n)
        g = xy.map { $0 / scale }
        var centers: [SIMD2<Double>] = []
        for ch in Geometry.alphabet {
            for (r, row) in Geometry.rows.enumerated() {
                if let c = row.firstIndex(of: ch) {
                    let col = Double(row.distance(from: row.startIndex, to: c))
                    centers.append(SIMD2(Geometry.rowInset[r] + (col + 0.5) / Geometry.nCols, (Double(r) + 0.5) / Geometry.nRows))
                }
            }
        }
        keys = centers.map { $0 / scale }
        // dwell weights: time per unit arclength along the resampled path, mean 1
        if cfg.timeWeight > 0 && pts.count >= 2 {
            var tt = t
            for i in 1..<tt.count where tt[i] < tt[i - 1] { tt[i] = tt[i - 1] }        // np.maximum.accumulate
            let total = max(u.last ?? 0, 1e-9)
            var ti = [Double](repeating: 0, count: n)
            for i in 0..<n { ti[i] = GestureTrace.interp(i == n - 1 ? total : Double(i) * total / Double(n - 1), xp: u, fp: tt) }
            var dt = [Double](repeating: 0, count: n)
            for i in 0..<n {
                if i == 0 { dt[i] = ti[1] - ti[0] }
                else if i == n - 1 { dt[i] = ti[n - 1] - ti[n - 2] }
                else { dt[i] = (ti[i + 1] - ti[i - 1]) / 2 }
            }
            let mean = max(dt.reduce(0, +) / Double(n), 1e-9)
            let tWeight = config.timeWeight
            tw = dt.map { pow(min(max($0 / mean, 0.25), 4.0), tWeight) }
        } else {
            tw = [Double](repeating: 1, count: n)
        }
        var land = [[Double]](repeating: [Double](repeating: 0, count: n), count: keys.count)
        var hover = land
        for k in 0..<keys.count {
            for j in 0..<n {
                let d2 = (g[j] - keys[k]).lengthSquared
                land[k][j] = d2 / (cfg.sigmaKey * cfg.sigmaKey)
                hover[k][j] = cfg.wTransit * d2 * tw[j] / (cfg.sigmaTransit * cfg.sigmaTransit)
            }
        }
        self.land = land; self.hover = hover
    }

    /// np.interp over a non-decreasing `xp`, duplicates included: numpy takes the
    /// largest j with xp[j] <= x, returns fp[j] on an exact hit, else interpolates
    /// to j+1 (which is strictly greater). The dwell weights depend on this
    /// exact behaviour where a finger paused (zero-length steps).
    static func interp(_ x: Double, xp: [Double], fp: [Double]) -> Double {
        let n = xp.count
        if x < xp[0] { return fp[0] }              // x == xp[0] with duplicates at 0 takes the last of them, as numpy does
        if x >= xp[n - 1] { return fp[n - 1] }
        var lo = 0, hi = n - 1                     // invariant: xp[lo] <= x < xp[hi]
        while hi - lo > 1 { let mid = (lo + hi) / 2; if xp[mid] <= x { lo = mid } else { hi = mid } }
        if xp[lo] == x { return fp[lo] }
        return fp[lo] + (fp[hi] - fp[lo]) * (x - xp[lo]) / (xp[hi] - xp[lo])
    }

    /// (N,) transit cost of each point against the segment key a -> key b.
    func segTransit(_ a: Int, _ b: Int) -> [Double] {
        let p = keys[a], v = keys[b] - p, vv = v.lengthSquared
        if vv < 1e-12 { return hover[a] }
        var out = [Double](repeating: 0, count: n)
        for j in 0..<n {
            let tt = min(max(dot(g[j] - p, v) / vv, 0), 1)
            let proj = p + v * tt
            out[j] = cfg.wTransit * (g[j] - proj).lengthSquared * tw[j] / (cfg.sigmaTransit * cfg.sigmaTransit)
        }
        return out
    }

    func initRow(_ k: Int) -> [Double] {
        var row = [Double](repeating: 0, count: n)
        var lead = 0.0
        for j in 0..<n { row[j] = lead + land[k][j]; lead += hover[k][j] }
        return row
    }

    func extend(_ row: [Double], prev: Int, next k: Int) -> [Double] {
        let trans = segTransit(prev, k)
        var cum = [Double](repeating: 0, count: n)
        var acc = 0.0
        for j in 0..<n { acc += trans[j]; cum[j] = acc }
        var out = [Double](repeating: 0, count: n)
        var bestPrev = Double.infinity          // min_{i<j} row[i] - cum[i]
        for j in 0..<n {
            let between = j == 0 ? Double.infinity : bestPrev + cum[j - 1]
            out[j] = land[k][j] + min(row[j], between)
            bestPrev = min(bestPrev, row[j] - cum[j])
        }
        return out
    }

    func final(_ row: [Double], _ k: Int) -> Double {
        var best = Double.infinity
        var tail = 0.0
        for j in stride(from: n - 1, through: 0, by: -1) {
            best = min(best, row[j] + tail)
            tail += hover[k][j]
        }
        return best
    }

    /// Full alignment cost of `word` (lowercase a–z; other characters are skipped).
    public func cost(of word: String) -> Double {
        let idx = word.lowercased().compactMap { Geometry.alphabet.firstIndex(of: $0) }
        guard let first = idx.first else { return .infinity }
        var row = initRow(first)
        for (a, b) in zip(idx, idx.dropFirst()) { row = extend(row, prev: a, next: b) }
        return final(row, idx[idx.count - 1])
    }

    public func costPerLetter(of word: String) -> Double {
        let len = max(word.filter { $0.isLetter }.count, 1)
        return cost(of: word) / Double(len)
    }

    /// #81's verdict for a gesture labeled `word`: nil = traced, else the reason.
    public func rejection(for word: String, threshold: Double = GestureTrace.untracedCostPerLetter) -> String? {
        let len = word.filter { $0.isLetter }.count
        if len >= 4 && pathKeyWidths < 0.5 * Double(len - 1) { return "aborted" }
        return costPerLetter(of: word) > threshold ? "untraced" : nil
    }
}

private extension SIMD2 where Scalar == Double {
    var lengthSquared: Double { x * x + y * y }
    var length: Double { lengthSquared.squareRoot() }
}
private func dot(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.x + a.y * b.y }
