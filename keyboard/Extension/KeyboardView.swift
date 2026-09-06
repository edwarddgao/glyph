import UIKit
import GlyphCore

protocol KeyboardViewDelegate: AnyObject {
    func keyboardView(_ view: KeyboardView, didTapLetter letter: Character)
    func keyboardView(_ view: KeyboardView, didTapText text: String)
    func keyboardViewDidTapSpace(_ view: KeyboardView)
    func keyboardViewDidTapBackspace(_ view: KeyboardView)
    /// Held delete, escalated: remove the word before the cursor and the spaces after it.
    func keyboardViewDidDeleteWord(_ view: KeyboardView)
    func keyboardViewDidTapReturn(_ view: KeyboardView)
    func keyboardView(_ view: KeyboardView, didSwipe samples: [TouchSample])
    func keyboardView(_ view: KeyboardView, didPickSuggestion index: Int)
    /// Long-press on the 123 key: toggle the sentence LM (benchmark control).
    func keyboardViewDidToggleLM(_ view: KeyboardView)
}

/// Predictive bar + letter grid (swipeable) + bottom row, laid out on
/// `NativeMetrics` so every key sits where the system keyboard's does.
final class KeyboardView: UIView {
    weak var delegate: KeyboardViewDelegate?

    static var preferredHeight: CGFloat { NativeMetrics.totalHeight }

    // predictive bar: three slots, the middle one carries the pill
    private let pill = UIView()
    private var slotButtons: [UIButton] = []
    private let statusLabel = UILabel()
    /// candidate index shown in each slot (left, middle, right)
    private var slotCandidate: [Int?] = [nil, nil, nil]

    let grid = LetterGridView()
    let symbolGrid = SymbolGridView()
    let globeButton = KeyButton(symbol: "globe")
    let emojiPanel = EmojiPanelView()
    private let layerButton = KeyButton(title: "123", font: NativeMetrics.smallFont)
    private let secondButton: KeyButton
    private let spaceButton = KeyButton()
    private let returnButton = KeyButton(symbol: "return")
    private let needsGlobe: Bool

    var shiftState: ShiftState = .off { didSet { grid.shiftState = shiftState } }
    private var showingSymbols = false

    init(needsGlobe: Bool) {
        self.needsGlobe = needsGlobe
        // The system keyboard's emoji key; the globe instead when iOS asks for
        // one (older iOS without the system input-mode bar).
        secondButton = needsGlobe ? KeyButton(symbol: "globe") : KeyButton(symbol: "face.smiling")
        super.init(frame: .zero)
        build()
    }

    required init?(coder: NSCoder) { fatalError() }

    private func build() {
        backgroundColor = Palette.background

        pill.layer.cornerRadius = NativeMetrics.pillHeight / 2
        pill.layer.cornerCurve = .continuous
        pill.isUserInteractionEnabled = false
        addSubview(pill)
        for i in 0..<3 {
            let b = UIButton(type: .custom)
            b.titleLabel?.font = NativeMetrics.barFont
            b.titleLabel?.lineBreakMode = .byTruncatingTail
            b.tag = i
            b.accessibilityIdentifier = "suggestion\(i)"
            b.addTarget(self, action: #selector(slotTapped(_:)), for: .touchUpInside)
            slotButtons.append(b)
            addSubview(b)
        }
        statusLabel.font = .systemFont(ofSize: 14)
        statusLabel.textAlignment = .center
        addSubview(statusLabel)

        grid.onTapLetter = { [weak self] ch in guard let s = self else { return }; s.delegate?.keyboardView(s, didTapLetter: ch) }
        grid.onSwipe = { [weak self] pts in guard let s = self else { return }; s.delegate?.keyboardView(s, didSwipe: pts) }
        grid.onShift = { [weak self] in self?.toggleShift() }
        grid.onBackspace = { [weak self] in guard let s = self else { return }; s.delegate?.keyboardViewDidTapBackspace(s) }
        grid.onBackspaceWord = { [weak self] in guard let s = self else { return }; s.delegate?.keyboardViewDidDeleteWord(s) }
        addSubview(grid)

        symbolGrid.onText = { [weak self] t in guard let s = self else { return }; s.delegate?.keyboardView(s, didTapText: t) }
        symbolGrid.onBackspace = { [weak self] in guard let s = self else { return }; s.delegate?.keyboardViewDidTapBackspace(s) }
        symbolGrid.isHidden = true
        addSubview(symbolGrid)

        spaceButton.accessibilityIdentifier = "space"
        spaceButton.accessibilityLabel = "space"
        returnButton.accessibilityLabel = "return"
        layerButton.accessibilityIdentifier = "layer"
        layerButton.addTarget(self, action: #selector(toggleSymbols), for: .touchUpInside)
        // A benchmark control, not a feature: long enough that a resting thumb
        // never trips it, and the off state announces itself in the bar.
        let lmPress = UILongPressGestureRecognizer(target: self, action: #selector(layerLongPressed(_:)))
        lmPress.minimumPressDuration = 2.0
        layerButton.addGestureRecognizer(lmPress)
        for b in [layerButton, secondButton, spaceButton, returnButton] { b.addTarget(self, action: #selector(keyDown), for: .touchDown) }
        spaceButton.addTarget(self, action: #selector(spaceTapped), for: .touchUpInside)
        returnButton.addTarget(self, action: #selector(returnTapped), for: .touchUpInside)
        if !needsGlobe { secondButton.addTarget(self, action: #selector(emojiTapped), for: .touchUpInside) }
        secondButton.accessibilityIdentifier = needsGlobe ? "globe" : "emoji"
        emojiPanel.isHidden = true
        emojiPanel.onEmoji = { [weak self] e in guard let s = self else { return }; s.delegate?.keyboardView(s, didTapText: e) }
        emojiPanel.onBackspace = { [weak self] in guard let s = self else { return }; s.delegate?.keyboardViewDidTapBackspace(s) }
        emojiPanel.onLetters = { [weak self] in self?.hideEmoji() }
        addSubview(emojiPanel)
        if needsGlobe { globeButton.isHidden = true }   // secondButton *is* the globe then
        for b in [layerButton, secondButton, spaceButton, returnButton] { addSubview(b) }
        addSubview(globeButton)
        restyle()
    }

    /// The globe key exposed to the controller for `handleInputModeList`.
    var inputModeButton: UIButton { needsGlobe ? secondButton : globeButton }

    override func layoutSubviews() {
        super.layoutSubviews()
        let w = bounds.width
        let m = NativeMetrics.self
        // predictive bar
        let barH = m.barHeight
        let slotW = w / 3
        for (i, b) in slotButtons.enumerated() {
            b.frame = CGRect(x: CGFloat(i) * slotW, y: 0, width: slotW, height: barH)
        }
        pill.frame = CGRect(x: slotW, y: (barH - m.pillHeight) / 2, width: slotW, height: m.pillHeight)
        statusLabel.frame = CGRect(x: 0, y: 0, width: w, height: barH)
        // letter rows: three row cells starting right under the bar
        let gridFrame = CGRect(x: 0, y: barH, width: w, height: 3 * m.rowPitch)
        grid.frame = gridFrame
        symbolGrid.frame = gridFrame
        // bottom row (row cell index 0 within its own coordinate space)
        let by = gridFrame.maxY
        func f(_ col: CGFloat, _ units: CGFloat) -> CGRect {
            m.key(width: w, row: 0, column: col, units: units).offsetBy(dx: 0, dy: by)
        }
        layerButton.frame = f(0, 1.25)
        layerButton.titleEdgeInsets = UIEdgeInsets(top: 2 * m.smallLabelShift, left: 0, bottom: 0, right: 0)
        secondButton.frame = f(1.25, 1.25)
        spaceButton.frame = f(2.5, 5)
        returnButton.frame = f(7.5, 2.5)
        globeButton.frame = .zero
        emojiPanel.frame = bounds
    }

    /// Dark when the host asks for a dark keyboard, or leaves it to the system
    /// (`.default`) and the interface style is dark — the same rule the system
    /// keyboard follows. The trait collection reflects the host app's style,
    /// including an app that forces dark while the system is light.
    func applyAppearance(_ a: UIKeyboardAppearance, style: UIUserInterfaceStyle) {
        let dark = a == .dark || (a == .default && style == .dark)
        guard dark != Palette.dark || !styled else { return }
        Palette.dark = dark; styled = true
        restyle()
    }
    private var styled = false

    private func restyle() {
        backgroundColor = Palette.background
        pill.backgroundColor = Palette.pill
        for b in slotButtons { b.setTitleColor(Palette.text, for: .normal) }
        statusLabel.textColor = Palette.text.withAlphaComponent(0.6)
        grid.restyle()
        symbolGrid.restyle()
        for b in [globeButton, layerButton, secondButton, spaceButton, returnButton] { b.restyle() }
        emojiPanel.restyle()
    }

    // MARK: predictive bar

    /// Debug channel for the replay benchmark: readable by XCUITest as the
    /// middle slot's accessibility value.
    func setGestureDebug(_ text: String) { slotButtons[1].accessibilityValue = text }

    /// `text` is what the user sees; `detail` (memory, LM state) rides along as
    /// the label's accessibility value, so the replay benchmark can read it
    /// without it ever appearing on screen.
    func showStatus(_ text: String, detail: String = "") {
        statusLabel.text = text
        statusLabel.accessibilityValue = detail
        statusLabel.isHidden = false
        pill.isHidden = true
        for b in slotButtons { b.isHidden = true }
    }

    /// `words[0]` is the inserted word and goes in the middle pill, like the
    /// system keyboard's highlighted candidate; alternatives flank it.
    func showSuggestions(_ words: [String]) {
        statusLabel.isHidden = true
        pill.isHidden = words.isEmpty
        let order: [Int?] = [words.count > 1 ? 1 : nil, words.isEmpty ? nil : 0, words.count > 2 ? 2 : nil]
        slotCandidate = order
        for (i, b) in slotButtons.enumerated() {
            b.isHidden = false
            let w = order[i].map { words[$0] } ?? ""
            b.setTitle(w, for: .normal)
            b.isEnabled = !w.isEmpty
        }
    }

    func clearSuggestions() {
        statusLabel.isHidden = true
        pill.isHidden = true
        slotCandidate = [nil, nil, nil]
        for b in slotButtons { b.isHidden = false; b.setTitle("", for: .normal); b.isEnabled = false }
    }

    @objc private func slotTapped(_ sender: UIButton) {
        guard let idx = slotCandidate[sender.tag] else { return }
        delegate?.keyboardView(self, didPickSuggestion: idx)
    }

    // MARK: bottom row

    private func toggleShift() {
        switch shiftState {
        case .off: shiftState = .on
        case .on: shiftState = .off
        case .locked: shiftState = .off
        }
    }

    @objc private func toggleSymbols() {
        showingSymbols.toggle()
        grid.isHidden = showingSymbols
        symbolGrid.isHidden = !showingSymbols
        if !showingSymbols { symbolGrid.showNumbers() }
        layerButton.setTitle(showingSymbols ? "ABC" : "123", for: .normal)
    }

    @objc private func layerLongPressed(_ g: UILongPressGestureRecognizer) {
        if g.state == .began { delegate?.keyboardViewDidToggleLM(self) }
    }

    @objc private func keyDown() { UIDevice.current.playInputClick() }
    @objc private func spaceTapped() { delegate?.keyboardViewDidTapSpace(self) }
    @objc private func emojiTapped() {
        emojiPanel.reloadSections()
        emojiPanel.isHidden = false
        bringSubviewToFront(emojiPanel)
    }

    func hideEmoji() { emojiPanel.isHidden = true }
    var isShowingEmoji: Bool { !emojiPanel.isHidden }
    @objc private func returnTapped() { delegate?.keyboardViewDidTapReturn(self) }
}
