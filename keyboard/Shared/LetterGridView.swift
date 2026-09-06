import UIKit
import GlyphCore

enum ShiftState { case off, on, locked }

/// Three letter rows on the native grid, shift and delete in the third row's
/// side insets. Collects raw touch samples for a swipe and draws the trail;
/// decides tap vs swipe on touch-up.
///
/// The view's coordinate space is the canonical grid's: x = 0 at the left edge
/// of the q cell, x = 1 at the right edge of the p cell, y = 0 at the top of
/// the first row cell, y = 1 at the bottom of the third — so a touch maps to
/// corpus coordinates with one division per axis.
final class LetterGridView: UIView {
    var onTapLetter: ((Character) -> Void)?
    var onSwipe: (([TouchSample]) -> Void)?
    var onShift: (() -> Void)?
    var onBackspace: (() -> Void)?
    /// Held delete, after a run of characters: take a whole word at a time (the system keyboard's escalation).
    var onBackspaceWord: (() -> Void)?

    var shiftState: ShiftState = .off { didSet { updateShift() } }

    private var keyBacks: [Character: UIView] = [:]
    private var keyLabels: [Character: UILabel] = [:]
    private let shiftKey = KeyButton(symbol: "shift")
    private let deleteKey = KeyButton(symbol: "delete.left")
    private let trail = CAShapeLayer()
    private let demo = CAShapeLayer()
    private var demoWord: String?
    private var demoTimer: Timer?
    /// Fires when a demo stroke has finished drawing the word.
    var onDemoWordDrawn: ((String) -> Void)?

    private var samples: [TouchSample] = []
    private var rawPoints: [CGPoint] = []
    private var startTime: TimeInterval = 0
    private var activeTouch: UITouch?
    private var pressedKey: Character?
    private var deleteTimer: Timer?

    /// A gesture shorter than this (in key widths) and quicker than
    /// `tapMaxDuration` is a tap on the key it started on.
    private let tapMaxPath = 0.7
    private let tapMaxDuration: TimeInterval = 0.28

    override init(frame: CGRect) {
        super.init(frame: frame)
        isMultipleTouchEnabled = false
        for row in Geometry.rows {
            for ch in row {
                let back = UIView()
                back.layer.cornerRadius = NativeMetrics.cornerRadius
                back.layer.cornerCurve = .continuous
                back.isUserInteractionEnabled = false
                // VoiceOver: each key is an element; activation lands a touch on
                // it, which this view handles like any tap.
                back.isAccessibilityElement = true
                back.accessibilityLabel = String(ch)
                back.accessibilityTraits = .keyboardKey
                addSubview(back)
                keyBacks[ch] = back
                let l = UILabel()
                l.text = String(ch)
                l.font = NativeMetrics.letterFont
                l.textAlignment = .center
                l.isUserInteractionEnabled = false
                addSubview(l)
                keyLabels[ch] = l
            }
        }
        // Touches on these are handled by this view so a swipe that drifts
        // over them keeps recording.
        shiftKey.isUserInteractionEnabled = false
        deleteKey.isUserInteractionEnabled = false
        shiftKey.accessibilityIdentifier = "shift"
        deleteKey.accessibilityIdentifier = "delete"
        shiftKey.accessibilityLabel = "shift"
        deleteKey.accessibilityLabel = "delete"
        for k in [shiftKey, deleteKey] { k.isAccessibilityElement = true; k.accessibilityTraits = .keyboardKey }
        addSubview(shiftKey)
        addSubview(deleteKey)
        trail.fillColor = nil
        trail.strokeColor = Palette.trail.cgColor
        trail.lineWidth = 5
        trail.lineCap = .round
        trail.lineJoin = .round
        layer.addSublayer(trail)
        demo.fillColor = nil; demo.lineWidth = 5; demo.lineCap = .round; demo.lineJoin = .round; demo.strokeEnd = 0
        layer.addSublayer(demo)
        restyle()
    }

    // MARK: demo trail

    /// Loop an animated finger trail through `word`'s key centres (nil stops).
    func playDemo(_ word: String?) {
        demoWord = word
        demoTimer?.invalidate(); demoTimer = nil
        demo.removeAllAnimations(); demo.path = nil
        guard word != nil else { return }
        if bounds.width > 0 { startDemoLoop() }
    }

    private func demoCenter(_ ch: Character) -> CGPoint? {
        for (r, row) in Geometry.rows.enumerated() {
            if let i = row.firstIndex(of: ch) {
                let col = CGFloat(Geometry.rowInset[r]) * 10 + CGFloat(row.distance(from: row.startIndex, to: i))
                let f = NativeMetrics.key(width: bounds.width, row: r, column: col, units: 1)
                return CGPoint(x: f.midX, y: f.midY)
            }
        }
        return nil
    }

    private func startDemoLoop() {
        guard let word = demoWord else { return }
        let pts = word.compactMap { demoCenter($0) }
        guard pts.count >= 2 else { return }
        // a smooth path through the centres (Catmull-Rom -> cubic Béziers)
        let path = UIBezierPath(); path.move(to: pts[0])
        for i in 0..<pts.count - 1 {
            let p0 = pts[max(i - 1, 0)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[min(i + 2, pts.count - 1)]
            let c1 = CGPoint(x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6)
            let c2 = CGPoint(x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6)
            path.addCurve(to: p2, controlPoint1: c1, controlPoint2: c2)
        }
        demo.path = path.cgPath
        let draw = 0.32 * Double(pts.count - 1)
        func run() {
            demo.removeAllAnimations()
            demo.opacity = 1; demo.strokeEnd = 0
            let a = CABasicAnimation(keyPath: "strokeEnd"); a.fromValue = 0; a.toValue = 1; a.duration = draw
            a.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut); a.fillMode = .forwards; a.isRemovedOnCompletion = false
            demo.add(a, forKey: "draw")
            DispatchQueue.main.asyncAfter(deadline: .now() + draw) { [weak self] in
                guard let self, self.demoWord == word else { return }
                self.onDemoWordDrawn?(word)
                let f = CABasicAnimation(keyPath: "opacity"); f.fromValue = 1; f.toValue = 0; f.duration = 0.5; f.beginTime = CACurrentMediaTime() + 0.9
                f.fillMode = .forwards; f.isRemovedOnCompletion = false
                self.demo.add(f, forKey: "fade")
            }
        }
        run()
        demoTimer = Timer.scheduledTimer(withTimeInterval: draw + 2.4, repeats: true) { _ in run() }
    }

    required init?(coder: NSCoder) { fatalError() }

    func restyle() {
        for (_, b) in keyBacks { b.backgroundColor = Palette.key }
        for (_, l) in keyLabels { l.textColor = Palette.text }
        shiftKey.restyle(); deleteKey.restyle()
        trail.strokeColor = Palette.trail.cgColor
        demo.strokeColor = Palette.trail.cgColor
        updateShift()
    }

    private func updateShift() {
        let upper = shiftState != .off
        for (ch, l) in keyLabels { l.text = upper ? String(ch).uppercased() : String(ch) }
        switch shiftState {
        case .off: shiftKey.symbolName = "shift"
        case .on: shiftKey.symbolName = "shift.fill"
        case .locked: shiftKey.symbolName = "capslock.fill"
        }
        shiftKey.tintColor = Palette.text
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let w = bounds.width
        let m = NativeMetrics.self
        for (r, row) in Geometry.rows.enumerated() {
            for (c, ch) in row.enumerated() {
                let col = CGFloat(Geometry.rowInset[r]) * 10 + CGFloat(c)
                let f = m.key(width: w, row: r, column: col, units: 1)
                keyBacks[ch]?.frame = f
                keyLabels[ch]?.frame = f.offsetBy(dx: m.letterXShift, dy: m.letterBaselineShift)
            }
        }
        shiftKey.frame = m.key(width: w, row: 2, column: 0, units: 1.3)
        deleteKey.frame = m.key(width: w, row: 2, column: 10 - 1.3, units: 1.3)
        trail.frame = bounds
        demo.frame = bounds
        if demoWord != nil && demoTimer == nil { startDemoLoop() }
    }

    // MARK: touches

    private func canonical(_ p: CGPoint) -> (Double, Double) {
        let w = bounds.width
        let x = (p.x - NativeMetrics.gridLeft(w)) / NativeMetrics.gridWidth(w)
        let y = p.y / (3 * NativeMetrics.rowPitch)
        return (Double(x), Double(y))
    }

    private enum Zone { case letter, shift, delete }

    /// Row-3 side insets are shift and delete; their touch cells span the
    /// full inset, like the system keyboard.
    private func zone(at p: CGPoint) -> Zone {
        let (x, y) = canonical(p)
        guard Geometry.row(atY: y) == 2 else { return .letter }
        if x < Geometry.rowInset[2] { return .shift }
        if x > 1 - Geometry.rowInset[2] { return .delete }
        return .letter
    }

    /// Called when a touch is ignored or force-finished; the controller logs it.
    var onDiagnostic: ((String) -> Void)?

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first else { return }
        if let stale = activeTouch {
            // A new finger before the previous one's lift was processed: close the
            // old gesture rather than dropping the new one (seen with synthesized input).
            onDiagnostic?("touch began while another was active; finishing the old one")
            finish(stale, event: event, cancelled: false)
        }
        activeTouch = t
        let p = t.location(in: self)
        samples = []
        rawPoints = []
        startTime = t.timestamp
        UIDevice.current.playInputClick()   // the system keyboard's click; honours the user's Keyboard Clicks setting
        switch zone(at: p) {
        case .shift:
            shiftKey.isHighlighted = true
        case .delete:
            deleteKey.isHighlighted = true
            onBackspace?()
            // Hold: characters at 12/s after 0.45 s; after about a second of that, whole words at ~6/s.
            deleteTimer = Timer.scheduledTimer(withTimeInterval: 0.45, repeats: false) { [weak self] _ in
                var repeats = 0
                self?.deleteTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { [weak self] _ in
                    guard let self else { return }
                    repeats += 1
                    if repeats <= 12 { self.onBackspace?(); return }
                    self.deleteTimer?.invalidate()
                    self.deleteTimer = Timer.scheduledTimer(withTimeInterval: 0.16, repeats: true) { [weak self] _ in
                        self?.onBackspaceWord?()
                    }
                }
            }
        case .letter:
            append(t, event: event)
            let (x, y) = canonical(p)
            pressedKey = Geometry.key(atX: x, y: y) ?? Geometry.nearestKey(atX: x, y: y)
            highlight(pressedKey, on: true)
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = activeTouch, touches.contains(t) else { return }
        guard !samples.isEmpty else { return }   // began on shift/delete
        append(t, event: event)
        drawTrail()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = activeTouch, touches.contains(t) else { return }
        finish(t, event: event, cancelled: false)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = activeTouch, touches.contains(t) else { return }
        finish(t, event: event, cancelled: true)
    }

    private func finish(_ t: UITouch, event: UIEvent?, cancelled: Bool) {
        defer {
            activeTouch = nil
            samples = []
            rawPoints = []
            highlight(pressedKey, on: false)
            pressedKey = nil
            shiftKey.isHighlighted = false
            deleteKey.isHighlighted = false
            deleteTimer?.invalidate()
            deleteTimer = nil
            drawTrail()
        }
        if deleteTimer != nil { return }                         // delete handled on touch-down
        if samples.isEmpty {                                     // began on shift
            if !cancelled, zone(at: t.location(in: self)) == .shift {
                if t.tapCount >= 2 { shiftState = .locked } else { onShift?() }
            }
            return
        }
        if cancelled { return }
        append(t, event: nil)
        let duration = t.timestamp - startTime
        if isTap(duration: duration), let k = pressedKey {
            onTapLetter?(k)
            return
        }
        guard samples.count >= 3 else { if let k = pressedKey { onTapLetter?(k) }; return }
        onSwipe?(samples)
    }

    private func isTap(duration: TimeInterval) -> Bool {
        var path = 0.0
        for i in 1..<max(samples.count, 1) {
            let dx = (samples[i].x - samples[i - 1].x) / Geometry.keyScaleX
            let dy = (samples[i].y - samples[i - 1].y) / Geometry.keyScaleY
            path += (dx * dx + dy * dy).squareRoot()
        }
        return path < tapMaxPath && duration < tapMaxDuration
    }

    private func append(_ t: UITouch, event: UIEvent?) {
        let touches = event?.coalescedTouches(for: t) ?? [t]
        for ct in touches {
            let p = ct.location(in: self)
            let (x, y) = canonical(p)
            let ms = ((ct.timestamp - startTime) * 1000).rounded()
            if let last = samples.last, last.x == x, last.y == y, last.t == ms { continue }
            samples.append(TouchSample(x: x, y: y, t: ms))
            rawPoints.append(p)
        }
    }

    private func drawTrail() {
        guard rawPoints.count >= 2 else { trail.path = nil; return }
        let path = UIBezierPath()
        path.move(to: rawPoints[0])
        for p in rawPoints.dropFirst() { path.addLine(to: p) }
        trail.path = path.cgPath
    }

    private func highlight(_ ch: Character?, on: Bool) {
        guard let ch, let b = keyBacks[ch] else { return }
        b.backgroundColor = on ? Palette.keyPressed : Palette.key
    }
}

/// The system keyboard's "123" and "#+=" layers, tap-only, on the same grid.
final class SymbolGridView: UIView {
    var onText: ((String) -> Void)?
    var onBackspace: (() -> Void)?

    private static let numbers: [[String]] = [
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        ["-", "/", ":", ";", "(", ")", "$", "&", "@", "\""],
        [".", ",", "?", "!", "'"],
    ]
    private static let symbols: [[String]] = [
        ["[", "]", "{", "}", "#", "%", "^", "*", "+", "="],
        ["_", "\\", "|", "~", "<", ">", "€", "£", "¥", "•"],
        [".", ",", "?", "!", "'"],
    ]
    private var buttons: [KeyButton] = []
    private let switchKey = KeyButton(title: "#+=", font: NativeMetrics.tinyFont)
    private let deleteKey = KeyButton(symbol: "delete.left")
    private var showingSymbols = false

    override init(frame: CGRect) {
        super.init(frame: frame)
        for _ in 0..<25 {
            let b = KeyButton(font: NativeMetrics.letterFont)
            b.addTarget(self, action: #selector(tapped(_:)), for: .touchUpInside)
            b.addTarget(self, action: #selector(click), for: .touchDown)
            buttons.append(b)
            addSubview(b)
        }
        switchKey.addTarget(self, action: #selector(toggleLayer), for: .touchUpInside)
        deleteKey.addTarget(self, action: #selector(del), for: .touchUpInside)
        for k in [switchKey, deleteKey] { k.addTarget(self, action: #selector(click), for: .touchDown) }
        addSubview(switchKey)
        addSubview(deleteKey)
        showNumbers()
    }
    required init?(coder: NSCoder) { fatalError() }

    func showNumbers() { showingSymbols = false; relabel() }

    private func relabel() {
        let rows = showingSymbols ? Self.symbols : Self.numbers
        var i = 0
        for row in rows { for s in row { buttons[i].setTitle(s, for: .normal); i += 1 } }
        switchKey.setTitle(showingSymbols ? "123" : "#+=", for: .normal)
    }

    func restyle() { for b in buttons { b.restyle() }; switchKey.restyle(); deleteKey.restyle() }

    override func layoutSubviews() {
        super.layoutSubviews()
        let w = bounds.width
        let m = NativeMetrics.self
        var i = 0
        for r in 0..<2 {
            for c in 0..<10 {
                buttons[i].frame = m.key(width: w, row: r, column: CGFloat(c), units: 1)
                buttons[i].titleLabel?.frame = buttons[i].bounds.offsetBy(dx: 0, dy: m.letterBaselineShift)
                i += 1
            }
        }
        switchKey.frame = m.key(width: w, row: 2, column: 0, units: 1.3)
        for c in 0..<5 {
            buttons[i].frame = m.key(width: w, row: 2, column: 1.5 + 1.4 * CGFloat(c), units: 1.4)
            i += 1
        }
        deleteKey.frame = m.key(width: w, row: 2, column: 10 - 1.3, units: 1.3)
    }

    @objc private func click() { UIDevice.current.playInputClick() }
    @objc private func tapped(_ b: UIButton) { if let t = b.title(for: .normal) { onText?(t) } }
    @objc private func del() { onBackspace?() }
    @objc private func toggleLayer() { showingSymbols.toggle(); relabel() }
}
