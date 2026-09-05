import Foundation

/// The canonical QWERTY grid, mirroring `research/src/swipe_typing/layout.py`.
///
/// x in [0, 1] spans the 10 key columns of the top row; y in [0, 1] spans the
/// three letter rows. Rows are inset by 0 / 0.05 / 0.15 (in grid widths).
public enum Geometry {
    public static let rows: [[Character]] = ["qwertyuiop", "asdfghjkl", "zxcvbnm"].map(Array.init)
    public static let rowInset: [Double] = [0.0, 0.05, 0.15]
    public static let nCols = 10.0
    public static let nRows = 3.0
    public static let alphabet: [Character] = Array("abcdefghijklmnopqrstuvwxyz")
    public static let nKeys = 26

    /// Key half-extents: half a column wide, half a row tall.
    public static let radiusX = 0.5 / nCols
    public static let radiusY = 0.5 / nRows
    /// `features.key_scale`: median key (width, height) — one key in each axis.
    public static let keyScaleX = 1.0 / nCols
    public static let keyScaleY = 1.0 / nRows

    /// Key centers in alphabet order (index i is `alphabet[i]`).
    public static let centers: [SIMD2<Double>] = alphabet.map { center(of: $0)! }

    public static func center(of ch: Character) -> SIMD2<Double>? {
        for (r, row) in rows.enumerated() {
            if let c = row.firstIndex(of: ch) {
                return SIMD2((Double(c) + 0.5) / nCols + rowInset[r], (Double(r) + 0.5) / nRows)
            }
        }
        return nil
    }

    /// Letter index (0…25) of `ch`, or nil.
    public static func index(of ch: Character) -> Int? {
        guard let a = ch.asciiValue, a >= 97, a <= 122 else { return nil }
        return Int(a) - 97
    }

    /// Row index for a canonical y, clamped.
    public static func row(atY y: Double) -> Int {
        min(max(Int(floor(y * nRows)), 0), Int(nRows) - 1)
    }

    /// The letter whose key contains canonical (x, y). Points in a row's side
    /// insets (where iOS puts shift / delete) return nil.
    public static func key(atX x: Double, y: Double) -> Character? {
        guard y >= 0, y <= 1 else { return nil }
        let r = row(atY: y)
        let col = (x - rowInset[r]) * nCols
        let row = rows[r]
        guard col >= 0, col < Double(row.count) else { return nil }
        return row[Int(col)]
    }

    /// Nearest letter to canonical (x, y) within the same row — used for taps
    /// that land a hair outside a key.
    public static func nearestKey(atX x: Double, y: Double) -> Character {
        let r = row(atY: min(max(y, 0), 0.999))
        let col = Int(((x - rowInset[r]) * nCols).rounded(.down))
        let row = rows[r]
        return row[min(max(col, 0), row.count - 1)]
    }
}
