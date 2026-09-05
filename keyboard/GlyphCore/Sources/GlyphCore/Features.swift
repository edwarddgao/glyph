import Foundation

/// One raw touch sample in canonical grid coordinates, time in milliseconds
/// relative to the gesture start (integers, as the corpora store them).
public struct TouchSample: Equatable {
    public var x: Double
    public var y: Double
    public var t: Double
    public init(x: Double, y: Double, t: Double) { self.x = x; self.y = y; self.t = t }
}

/// Trajectory -> (64, 32) feature tensor, mirroring
/// `SwipeDataset.__getitem__` with no augmentation, `key_units=True`,
/// `resample_mode="time"`: `[key affinity (26) | vx vy ax ay speed curvature]`.
///
/// The float32 round-trips the research code takes (points stored as float32,
/// `to_key_units` and `resample` returning float32) are reproduced so the
/// features match the Python featurizer to float precision.
public enum Features {
    public static let nPoints = 64
    public static let nKinematic = 6
    public static let nInput = Geometry.nKeys + nKinematic
    static let curvatureClip = 1e3
    static let eps = 1e-6

    /// Row-major (nPoints, nInput) float32 features.
    public static func encode(_ samples: [TouchSample]) -> [Float] {
        let n = nPoints
        // Points as the cache would hold them: float32.
        let pts = samples.map { SIMD2(Double(Float($0.x)), Double(Float($0.y))) }
        let t = samples.map { Double(Int32($0.t.rounded())) }

        // --- affinity block: resample canonical points, gaussian per key ---
        let resampled = resample(pts, t: t, n: n)
        var out = [Float](repeating: 0, count: n * nInput)
        let rx = Geometry.radiusX, ry = Geometry.radiusY
        for i in 0..<n {
            let p = SIMD2(Double(Float(resampled[i].x)), Double(Float(resampled[i].y)))
            for k in 0..<Geometry.nKeys {
                let c = Geometry.centers[k]
                let dx = Float((p.x - c.x)) / Float(rx)
                let dy = Float((p.y - c.y)) / Float(ry)
                out[i * nInput + k] = expf(-0.5 * (dx * dx + dy * dy))
            }
        }

        // --- kinematics in key units ---
        let ku = pts.map { SIMD2(Double(Float($0.x) / Float(Geometry.keyScaleX)),
                                 Double(Float($0.y) / Float(Geometry.keyScaleY))) }
        let xy = resample(ku, t: t, n: n).map { SIMD2(Double(Float($0.x)), Double(Float($0.y))) }
        let durationMs = t.isEmpty ? 1000.0 : max(t[t.count - 1] - t[0], 1.0)
        let dt = (durationMs / 1000.0) / Double(max(n - 1, 1))
        let (vel, acc) = SavitzkyGolay.derivatives(xy, dt: dt)
        for i in 0..<n {
            let v = vel[i], a = acc[i]
            let speed = (v.x * v.x + v.y * v.y).squareRoot()
            let cross = v.x * a.y - v.y * a.x
            var curv = cross / (pow(speed, 3) + eps)
            curv = min(max(curv, -curvatureClip), curvatureClip)
            let base = i * nInput + Geometry.nKeys
            out[base + 0] = finite(v.x)
            out[base + 1] = finite(v.y)
            out[base + 2] = finite(a.x)
            out[base + 3] = finite(a.y)
            out[base + 4] = finite(speed)
            out[base + 5] = finite(curv)
        }
        return out
    }

    @inline(__always) static func finite(_ v: Double) -> Float {
        v.isFinite ? Float(v) : 0
    }

    /// `features.resample(mode="time")`: dedupe timestamps to strictly
    /// increasing, then linear interpolation onto a uniform time grid.
    public static func resample(_ pts: [SIMD2<Double>], t: [Double], n: Int) -> [SIMD2<Double>] {
        if pts.isEmpty { return [SIMD2<Double>](repeating: .zero, count: n) }
        if pts.count == 1 { return [SIMD2<Double>](repeating: pts[0], count: n) }
        var u = t
        for i in 1..<u.count where u[i] <= u[i - 1] { u[i] = u[i - 1] + 1e-3 }
        let u0 = u[0], u1 = u[u.count - 1]
        let step = (u1 - u0) / Double(n - 1)
        var out = [SIMD2<Double>](repeating: .zero, count: n)
        var j = 0
        for i in 0..<n {
            // np.linspace evaluates u0 + i*step, with the last point exactly u1.
            let g = i == n - 1 ? u1 : u0 + Double(i) * step
            while j + 1 < u.count - 1 && u[j + 1] < g { j += 1 }
            // segment [j, j+1] with u[j] <= g <= u[j+1] (np.interp semantics)
            while j > 0 && u[j] > g { j -= 1 }
            let a = u[j], b = u[j + 1]
            if g >= b { out[i] = pts[j + 1]; continue }
            if g <= a { out[i] = pts[j]; continue }
            let w = (g - a) / (b - a)
            out[i] = pts[j] + (pts[j + 1] - pts[j]) * w
        }
        return out
    }
}

/// Savitzky–Golay smoothing derivatives, window 7, polynomial order 2,
/// scipy `mode="interp"` semantics: interior points use the centered window,
/// the first/last three points reuse the edge window's polynomial evaluated
/// off-center. Implemented as a local least-squares quadratic fit, which is
/// exactly what the SG coefficients compute.
enum SavitzkyGolay {
    static let window = 7
    static let half = 3

    /// Inverse of the normal matrix for x = 0…6, basis [1, x, x²].
    static let inv: [[Double]] = {
        var m = [[Double]](repeating: [0, 0, 0], count: 3)
        var s = [Double](repeating: 0, count: 5)
        for x in 0..<window { var p = 1.0; for k in 0..<5 { s[k] += p; p *= Double(x) } }
        for r in 0..<3 { for c in 0..<3 { m[r][c] = s[r + c] } }
        return invert3(m)
    }()

    static func invert3(_ m: [[Double]]) -> [[Double]] {
        let a = m[0][0], b = m[0][1], c = m[0][2]
        let d = m[1][0], e = m[1][1], f = m[1][2]
        let g = m[2][0], h = m[2][1], i = m[2][2]
        let A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g
        let D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g)
        let G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d
        let det = a * A + b * B + c * C
        return [[A / det, D / det, G / det], [B / det, E / det, H / det], [C / det, F / det, I / det]]
    }

    /// (velocity, acceleration) per point; falls back to np.gradient-style
    /// finite differences when the series is too short for the window.
    static func derivatives(_ xy: [SIMD2<Double>], dt: Double) -> ([SIMD2<Double>], [SIMD2<Double>]) {
        let n = xy.count
        guard n >= window else { return gradientFallback(xy, dt: dt) }
        var vel = [SIMD2<Double>](repeating: .zero, count: n)
        var acc = [SIMD2<Double>](repeating: .zero, count: n)
        for i in 0..<n {
            let ws = min(max(i - half, 0), n - window)
            let u = Double(i - ws)
            // Moments of the window for x and y.
            var m0 = SIMD2<Double>.zero, m1 = SIMD2<Double>.zero, m2 = SIMD2<Double>.zero
            for k in 0..<window {
                let y = xy[ws + k], xk = Double(k)
                m0 += y; m1 += y * xk; m2 += y * xk * xk
            }
            let a1 = inv[1][0] * m0 + inv[1][1] * m1 + inv[1][2] * m2
            let a2 = inv[2][0] * m0 + inv[2][1] * m1 + inv[2][2] * m2
            vel[i] = (a1 + 2 * a2 * u) / dt
            acc[i] = (2 * a2) / (dt * dt)
        }
        return (vel, acc)
    }

    static func gradientFallback(_ xy: [SIMD2<Double>], dt: Double) -> ([SIMD2<Double>], [SIMD2<Double>]) {
        func grad(_ s: [SIMD2<Double>]) -> [SIMD2<Double>] {
            let n = s.count
            if n < 2 { return [SIMD2<Double>](repeating: .zero, count: n) }
            var g = [SIMD2<Double>](repeating: .zero, count: n)
            g[0] = (s[1] - s[0]) / dt
            g[n - 1] = (s[n - 1] - s[n - 2]) / dt
            for i in 1..<(n - 1) { g[i] = (s[i + 1] - s[i - 1]) / (2 * dt) }
            return g
        }
        let v = grad(xy)
        return (v, grad(v))
    }
}
