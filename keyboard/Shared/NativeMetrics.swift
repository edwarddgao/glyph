import UIKit

/// The native iOS keyboard's geometry, measured pixel-by-pixel from the
/// system keyboard on an iPhone 17 (402 pt wide, iOS 26.5) — see
/// `tools/measure_layout.py`. Everything is expressed in "pitches" so it
/// scales with the screen width the way the system keyboard does:
///
///   column pitch  p = (W - 2·margin + gap) / 10        (39.47 on 402 pt)
///   letter key    (p - gap) × 43,  row pitch 54
///   row insets    0 / 0.5 p / 1.5 p  (the canonical grid's 0 / 0.05 / 0.15)
///   shift, delete 1.3 p - gap;  bottom row 1.25 p, 1.25 p, 5 p, 2.5 p (minus gap)
///   numbers row 3 five keys on a 1.4 p pitch, 1.4 p - gap wide
///   predictive bar 42.8 pt above the first row's cell; 2.7 pt below the last key
enum NativeMetrics {
    static let margin: CGFloat = 20.0 / 3.0     // 6.67
    static let gap: CGFloat = 6
    static let keyHeight: CGFloat = 43
    static let rowPitch: CGFloat = 54
    static let cornerRadius: CGFloat = 6
    static let barHeight: CGFloat = 42.47
    static let bottomPad: CGFloat = 3.03
    static let pillHeight: CGFloat = 35

    static func pitch(_ width: CGFloat) -> CGFloat { (width - 2 * margin + gap) / 10 }
    static func keyWidth(_ width: CGFloat) -> CGFloat { pitch(width) - gap }
    /// Left edge of the 10-column cell grid (half a gap left of the first key).
    static func gridLeft(_ width: CGFloat) -> CGFloat { margin - gap / 2 }
    static func gridWidth(_ width: CGFloat) -> CGFloat { 10 * pitch(width) }
    /// Vertical offset of a key inside its 54 pt row cell.
    static var keyInset: CGFloat { (rowPitch - keyHeight) / 2 }   // 5.5
    static var totalHeight: CGFloat { barHeight + 4 * rowPitch - keyInset + bottomPad }  // ≈ 256

    /// Frame of the key at `column` (in pitches from the grid's left edge) with
    /// width `units` pitches, in row `row`, inside a view whose row cells start at y = 0.
    static func key(width: CGFloat, row: Int, column: CGFloat, units: CGFloat) -> CGRect {
        let p = pitch(width)
        return CGRect(x: gridLeft(width) + column * p + gap / 2,
                      y: CGFloat(row) * rowPitch + keyInset,
                      width: units * p - gap, height: keyHeight)
    }

    // Typography, matched to the native glyph boxes.
    static let letterFont = UIFont.systemFont(ofSize: 25, weight: .regular)
    /// Native letters sit ~2.5 pt above the key's geometric center.
    static let letterBaselineShift: CGFloat = -2.3
    static let letterXShift: CGFloat = 0.5
    static let smallLabelShift: CGFloat = -1.0
    static let smallFont = UIFont.systemFont(ofSize: 18, weight: .regular)   // "123", "ABC"
    static let tinyFont = UIFont.systemFont(ofSize: 15, weight: .regular)    // "#+="
    static let barFont = UIFont.systemFont(ofSize: 18, weight: .regular)
    static let iconConfig = UIImage.SymbolConfiguration(pointSize: 19, weight: .regular)
}

/// Native light/dark palette, sampled from the system keyboard.
enum Palette {
    static var dark = false
    static var background: UIColor { dark ? UIColor(red: 23/255, green: 23/255, blue: 23/255, alpha: 1)
                                          : UIColor(red: 222/255, green: 223/255, blue: 227/255, alpha: 1) }
    static var key: UIColor { dark ? UIColor(red: 61/255, green: 61/255, blue: 61/255, alpha: 1) : .white }
    static var keyPressed: UIColor { dark ? UIColor(white: 0.45, alpha: 1) : UIColor(white: 0.85, alpha: 1) }
    static var text: UIColor { dark ? .white : .black }
    static var pill: UIColor { dark ? UIColor(white: 118/255, alpha: 1) : UIColor(red: 230/255, green: 231/255, blue: 235/255, alpha: 1) }
    static var trail: UIColor { UIColor(red: 0.17, green: 0.44, blue: 0.94, alpha: 0.85) }
}

/// A native-looking key: flat rounded rect, no shadow (iOS 26), text or SF Symbol.
final class KeyButton: UIButton {
    var symbolName: String? { didSet { applySymbol() } }

    init(title: String? = nil, symbol: String? = nil, font: UIFont = NativeMetrics.letterFont) {
        super.init(frame: .zero)
        if let title { setTitle(title, for: .normal) }
        titleLabel?.font = font
        titleLabel?.adjustsFontSizeToFitWidth = false
        layer.cornerRadius = NativeMetrics.cornerRadius
        layer.cornerCurve = .continuous
        symbolName = symbol
        applySymbol()
        restyle()
    }
    required init?(coder: NSCoder) { fatalError() }

    private func applySymbol() {
        guard let symbolName else { setImage(nil, for: .normal); return }
        setImage(UIImage(systemName: symbolName, withConfiguration: NativeMetrics.iconConfig), for: .normal)
        tintColor = Palette.text
    }

    func restyle() {
        backgroundColor = Palette.key
        setTitleColor(Palette.text, for: .normal)
        tintColor = Palette.text
    }

    override var isHighlighted: Bool {
        didSet { backgroundColor = isHighlighted ? Palette.keyPressed : Palette.key }
    }
}
