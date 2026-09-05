import SwiftUI
import UIKit
import GlyphCore

/// The keyboard's letter grid embedded in the app, on the same `NativeMetrics`
/// geometry as the extension (and the system keyboard), so a gesture recorded
/// here is in the same canonical coordinates as one recorded on the keyboard.
///
/// Two uses: the race pad (interactive, pinned to the bottom so the letters
/// sit exactly where the system keyboard's do) and the hero (`demoWord`: an
/// animated finger trail spells a word and it appears in the bar; touches off).
struct SwipePad: UIViewRepresentable {
    var onSwipe: ([TouchSample]) -> Void = { _ in }
    var onTap: (Character) -> Void = { _ in }
    var demoWord: String? = nil
    /// Pinned at the bottom of the screen: add iOS 26's globe/mic bar below.
    var pinned = true
    @Environment(\.colorScheme) private var scheme

    /// Below the system keyboard's bottom row sits iOS 26's globe/mic bar; on the
    /// iPhone 17 the keyboard's bottom-row cell ends 69.5 pt above the screen edge,
    /// of which 34 is the home-indicator safe area SwiftUI already leaves. Measured
    /// from tools/measure_layout.py on QuickPath (bottom-row keys at y 753, h 43).
    static let systemBottomBar: CGFloat = 35.5
    /// The predictive bar's slot above the letters (the extension's suggestion bar);
    /// the game shows the target sentence there so the eyes never leave the keyboard.
    static let barHeight: CGFloat = NativeMetrics.barHeight
    /// Bar, three letter rows, the bottom row (123 · emoji · space · return), the
    /// keyboard's bottom pad — plus the system bar when pinned.
    static let heroHeight: CGFloat = barHeight + NativeMetrics.rowPitch * 4 + NativeMetrics.bottomPad
    static let height: CGFloat = heroHeight + systemBottomBar

    final class Container: UIView {
        let grid = LetterGridView()
        private let layerKey = KeyButton(title: "123", font: NativeMetrics.smallFont)
        private let emojiKey = KeyButton(symbol: "face.smiling")
        private let spaceKey = KeyButton(title: "space", font: NativeMetrics.smallFont)
        private let returnKey = KeyButton(symbol: "return")
        /// The middle pill of the suggestion bar, for the demo word.
        let pill = UIView()
        let pillLabel = UILabel()
        override init(frame: CGRect) {
            super.init(frame: frame)
            addSubview(grid)
            for k in [layerKey, emojiKey, spaceKey, returnKey] { k.isUserInteractionEnabled = false; addSubview(k) }
            pill.layer.cornerRadius = 8; pill.layer.cornerCurve = .continuous; pill.alpha = 0
            pillLabel.font = NativeMetrics.barFont; pillLabel.textAlignment = .center
            pill.addSubview(pillLabel); addSubview(pill)
        }
        required init?(coder: NSCoder) { fatalError() }
        override func layoutSubviews() {
            super.layoutSubviews()
            let w = bounds.width, m = NativeMetrics.self
            grid.frame = CGRect(x: 0, y: SwipePad.barHeight, width: w, height: m.rowPitch * 3)
            let by = grid.frame.maxY
            func f(_ col: CGFloat, _ units: CGFloat) -> CGRect { m.key(width: w, row: 0, column: col, units: units).offsetBy(dx: 0, dy: by) }
            layerKey.frame = f(0, 1.25); emojiKey.frame = f(1.25, 1.25); spaceKey.frame = f(2.5, 5); returnKey.frame = f(7.5, 2.5)
            layerKey.titleEdgeInsets = UIEdgeInsets(top: 2 * m.smallLabelShift, left: 0, bottom: 0, right: 0)
            pill.frame = CGRect(x: w / 3, y: (SwipePad.barHeight - m.pillHeight) / 2, width: w / 3, height: m.pillHeight)
            pillLabel.frame = pill.bounds
        }
        func restyle() {
            backgroundColor = Palette.background; grid.restyle()
            for k in [layerKey, emojiKey, spaceKey, returnKey] { k.restyle() }
            pill.backgroundColor = Palette.pill; pillLabel.textColor = Palette.text
        }
        func showDemoWord(_ w: String) {
            pillLabel.text = w
            UIView.animate(withDuration: 0.2) { self.pill.alpha = 1 }
            UIView.animate(withDuration: 0.4, delay: 1.6) { self.pill.alpha = 0 }
        }
    }

    func makeUIView(context: Context) -> Container {
        let v = Container()
        v.grid.onSwipe = onSwipe
        v.grid.onTapLetter = onTap
        v.grid.onShift = {}
        v.grid.onBackspace = {}
        v.grid.isUserInteractionEnabled = demoWord == nil
        v.grid.onDemoWordDrawn = { [weak v] w in v?.showDemoWord(w) }
        v.grid.playDemo(demoWord)
        Palette.dark = scheme == .dark
        v.restyle()
        return v
    }

    func updateUIView(_ v: Container, context: Context) {
        v.grid.onSwipe = onSwipe
        v.grid.onTapLetter = onTap
        let dark = scheme == .dark
        if Palette.dark != dark { Palette.dark = dark; v.restyle() }
    }
}
