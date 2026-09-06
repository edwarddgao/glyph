import UIKit
import GlyphCore

/// The custom keyboard. Owns the decoder and the text-proxy side effects;
/// `KeyboardView` owns pixels and touches.
final class KeyboardViewController: UIInputViewController, KeyboardViewDelegate {
    private var keyboardView: KeyboardView!
    private var decoder: SwipeDecoder?
    private var loadError: String?

    /// The word most recently inserted by a swipe, with its trailing space,
    /// so a suggestion tap can replace it.
    private var lastSwipe: (word: String, candidates: [Candidate])?
    private var heightConstraint: NSLayoutConstraint?

    /// The fused sentence beam (nil until the LM has loaded; first pass only until then).
    private var search: SentenceSearch?
    /// Words this sentence that we inserted by swiping, as typed (cased), each followed by a space.
    private var typed: [String] = []
    private var lmLabel = ""
    /// Sentence LM on/off (long-press 123). Persisted so a benchmark run can
    /// set it once; off = first pass only, the research's "no LM" row.
    private var lmEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "swipe.lm.enabled") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "swipe.lm.enabled") }
    }

    /// The bar before the first swipe: a hint, plus the one state a user must
    /// know about (the sentence model switched off). Memory and load state go
    /// into the label's accessibility value for the benchmark, never on screen.
    private func showStatus() {
        let avail = availableMemoryMB()
        let lm = !lmEnabled ? "LM off" : (lmLabel.isEmpty ? "LM loading" : lmLabel)
        let text = lmEnabled ? "swipe a word" : "swipe a word · sentence model off · hold 123 to turn on"
        keyboardView.showStatus(text, detail: "\(avail) MB free · \(lm)")
    }

    static let sharedLoad = DecoderLoader()

    override func viewDidLoad() {
        super.viewDidLoad()
        keyboardView = KeyboardView(needsGlobe: needsInputModeSwitchKey)
        keyboardView.delegate = self
        keyboardView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(keyboardView)
        NSLayoutConstraint.activate([
            keyboardView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            keyboardView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            keyboardView.topAnchor.constraint(equalTo: view.topAnchor),
            keyboardView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        keyboardView.inputModeButton.addTarget(self, action: #selector(handleInputModeList(from:with:)), for: .allTouchEvents)
        keyboardView.grid.onDiagnostic = { diag($0) }
        diag("extension start; available memory \(availableMemoryMB()) MB")
        keyboardView.showStatus("loading…")
        Self.sharedLoad.load { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case .success(let d):
                    self.decoder = d
                    let avail = availableMemoryMB()
                    diag("decoder loaded; available memory \(avail) MB")
                    UserDefaults.standard.set(avail, forKey: "swipe.probe.availMB")
                    self.showStatus()
                    Self.sharedLoad.loadLM { [weak self] result in
                        DispatchQueue.main.async {
                            guard let self else { return }
                            switch result {
                            case .success(let lm):
                                let search = SentenceSearch(lm: lm)
                                if let priors = Self.sharedLoad.priors, let trie = self.decoder?.trie {
                                    search.priorLookup = { w in
                                        guard let n = trie.node(for: w), trie.isWord(n) else { return nil }
                                        let p = priors[Int(n)]
                                        return p.isNaN ? nil : Double(p)
                                    }
                                }
                                self.search = search
                                let avail2 = availableMemoryMB()
                                self.lmLabel = "LM ok"
                                diag("LM loaded; available memory \(avail2) MB")
                                self.showStatus()
                            case .failure(let e):
                                self.lmLabel = "no LM (\(e))"
                                diag("LM failed: \(e)")
                                self.showStatus()
                            }
                        }
                    }
                case .failure(let e):
                    self.loadError = "Glyph could not load — open the Glyph app once"
                    diag("decoder failed: \(e)")
                    self.keyboardView.showStatus(self.loadError!, detail: "\(e)")
                }
            }
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        applyAppearance()
        if heightConstraint == nil {
            let h = view.heightAnchor.constraint(equalToConstant: KeyboardView.preferredHeight)
            h.priority = .init(999)
            h.isActive = true
            heightConstraint = h
        }
        updateShiftFromContext()
    }

    override func textDidChange(_ textInput: UITextInput?) {
        super.textDidChange(textInput)
        updateShiftFromContext()
        applyAppearance()
    }

    private func applyAppearance() {
        keyboardView.applyAppearance(textDocumentProxy.keyboardAppearance ?? .default, style: traitCollection.userInterfaceStyle)
    }

    override func traitCollectionDidChange(_ previous: UITraitCollection?) {
        super.traitCollectionDidChange(previous)
        if previous?.userInterfaceStyle != traitCollection.userInterfaceStyle { applyAppearance() }
    }

    // MARK: context

    private var proxy: UITextDocumentProxy { textDocumentProxy }

    private func updateShiftFromContext() {
        guard keyboardView.shiftState != .locked else { return }
        let before = proxy.documentContextBeforeInput ?? ""
        let trimmed = before.trimmingCharacters(in: .whitespaces)
        let atStart = before.isEmpty || trimmed.isEmpty
        let afterSentence = !trimmed.isEmpty && ".!?".contains(trimmed.last!) && before.last == " "
        keyboardView.shiftState = (atStart || afterSentence) && proxy.autocapitalizationType != .none ? .on : .off
    }

    /// True when the character before the cursor is a space we inserted after a word.
    private var endsWithAutoSpace: Bool {
        lastSwipe != nil && (proxy.documentContextBeforeInput ?? "").hasSuffix(" ")
    }

    /// Anything but a swipe ends the sentence the LM is conditioning on.
    private func resetSentence() {
        search?.reset()
        typed = []
    }

    /// True while the text before the cursor still ends with exactly what we
    /// swiped this sentence; otherwise the user moved on and we start over.
    private func sentenceStillIntact() -> Bool {
        let tail = typed.map { $0 + " " }.joined()
        return (proxy.documentContextBeforeInput ?? "").hasSuffix(tail)
    }

    /// A swiped word after text that does not end in whitespace needs a space
    /// first (a tapped "i" then a swipe), as the system keyboard does.
    private func leadingSpaceIfNeeded() -> String {
        guard let last = (proxy.documentContextBeforeInput ?? "").last else { return "" }
        return last.isWhitespace || last.isNewline ? "" : " "
    }

    /// The current sentence's text before the cursor, as LM context: everything
    /// after the last sentence terminator, lowercased, letters only.
    private func sentencePrefixFromDocument() -> String {
        let before = proxy.documentContextBeforeInput ?? ""
        let tail = before.split(whereSeparator: { ".!?\n".contains($0) }).last.map(String.init) ?? before
        return tail.lowercased().split(whereSeparator: { !$0.isLetter })
            .map(String.init).suffix(24).joined(separator: " ")
    }

    private func cased(_ word: String) -> String {
        switch keyboardView.shiftState {
        case .off: return word
        case .on: return word.prefix(1).uppercased() + word.dropFirst()
        case .locked: return word.uppercased()
        }
    }

    // MARK: KeyboardViewDelegate

    func keyboardView(_ view: KeyboardView, didTapLetter letter: Character) {
        resetSentence()
        lastSwipe = nil
        keyboardView.clearSuggestions()
        proxy.insertText(cased(String(letter)))
        if keyboardView.shiftState == .on { keyboardView.shiftState = .off }
    }

    func keyboardView(_ view: KeyboardView, didTapText text: String) {
        resetSentence()
        // punctuation / digits from the symbol layer
        if ".,?!;:".contains(text), endsWithAutoSpace {
            proxy.deleteBackward()
            proxy.insertText(text + " ")
        } else {
            proxy.insertText(text)
        }
        lastSwipe = nil
        keyboardView.clearSuggestions()
        updateShiftFromContext()
    }

    func keyboardViewDidTapSpace(_ view: KeyboardView) {
        resetSentence()
        if endsWithAutoSpace {
            // Double-space after a swiped word ends the sentence, like iOS.
            proxy.deleteBackward()
            proxy.insertText(". ")
            lastSwipe = nil
            keyboardView.clearSuggestions()
            keyboardView.shiftState = .on
            return
        }
        proxy.insertText(" ")
        lastSwipe = nil
        keyboardView.clearSuggestions()
        updateShiftFromContext()
    }

    func keyboardViewDidTapBackspace(_ view: KeyboardView) {
        resetSentence()
        // Right after a swipe, backspace takes the whole word (and its space) back,
        // as Gboard and QuickPath do; the next press deletes by character.
        if let last = lastSwipe, (proxy.documentContextBeforeInput ?? "").hasSuffix(last.word + " ") {
            for _ in 0..<(last.word.count + 1) { proxy.deleteBackward() }
        } else {
            proxy.deleteBackward()
        }
        lastSwipe = nil
        keyboardView.clearSuggestions()
        updateShiftFromContext()
    }

    func keyboardViewDidDeleteWord(_ view: KeyboardView) {
        resetSentence()
        // Trailing whitespace, then the run of non-whitespace before it — the system keyboard's unit.
        let before = proxy.documentContextBeforeInput ?? ""
        var n = 0, inWord = false
        for ch in before.reversed() {
            if ch.isWhitespace { if inWord { break } } else { inWord = true }
            n += 1
        }
        for _ in 0..<max(n, 1) { proxy.deleteBackward() }
        lastSwipe = nil
        keyboardView.clearSuggestions()
        updateShiftFromContext()
    }

    func keyboardViewDidTapReturn(_ view: KeyboardView) {
        resetSentence()
        proxy.insertText("\n")
        lastSwipe = nil
        keyboardView.clearSuggestions()
        updateShiftFromContext()
    }

    func keyboardView(_ view: KeyboardView, didSwipe samples: [TouchSample]) {
        guard let decoder else {
            keyboardView.showStatus(loadError ?? "loading…")
            return
        }
        if let f = samples.first, let l = samples.last {
            diag(String(format: "gesture: n=%d dur=%.0fms first=(%.3f,%.3f) last=(%.3f,%.3f)", samples.count, l.t - f.t, f.x, f.y, l.x, l.y))
            // Full received path (replay fidelity analysis; compact JSON). Off by
            // default — simulator benchmarks turn it on with
            // `simctl spawn <sim> defaults write com.edwardgao.glyph.keyboard swipe.debug.samples -bool YES`.
            if UserDefaults.standard.bool(forKey: "swipe.debug.samples") {
                let xs = samples.map { String(format: "%.4f", $0.x) }.joined(separator: ",")
                let ys = samples.map { String(format: "%.4f", $0.y) }.joined(separator: ",")
                let ts = samples.map { String(format: "%.0f", $0.t) }.joined(separator: ",")
                diag("samples: {\"x\":[\(xs)],\"y\":[\(ys)],\"t\":[\(ts)]}")
            }
            keyboardView.setGestureDebug(String(format: "n=%d dur=%.0f", samples.count, l.t - f.t))
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let cands: [Candidate]
        do { cands = try decoder.decode(samples) } catch {
            keyboardView.showStatus("could not read that swipe", detail: "\(error)")
            return
        }
        let firstPassMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        guard let best = cands.first else {
            keyboardView.showStatus("no word for that swipe", detail: String(format: "%.0f ms", firstPassMs))
            return
        }
        // Without an LM (off, or failed to load), first pass only.
        guard lmEnabled, let search else {
            let word = cased(best.word)
            proxy.insertText(leadingSpaceIfNeeded() + word + " ")
            lastSwipe = (word, cands)
            keyboardView.showSuggestions(Array(cands.prefix(3)).map { cased($0.word) })
            if keyboardView.shiftState == .on { keyboardView.shiftState = .off }
            return
        }
        if !typed.isEmpty && !sentenceStillIntact() {
            diag("sentence reset: context mismatch (typed \(typed.count) words; before=\((proxy.documentContextBeforeInput ?? "").suffix(40).debugDescription))")
            resetSentence()
        }
        if typed.isEmpty {
            // New swiped sentence: whatever was typed before it (tapped letters,
            // earlier words) is the LM's left context.
            search.reset(prefix: sentencePrefixFromDocument())
        }
        let lead = leadingSpaceIfNeeded()
        let t1 = CFAbsoluteTimeGetCurrent()
        let hyp: Hypothesis
        do { hyp = try search.step(candidates: cands.map { ($0.word, $0.score) }) } catch {
            diag("LM step failed: \(error)")
            search.reset(); typed = []
            let word = cased(best.word)
            proxy.insertText(lead + word + " ")
            lastSwipe = (word, cands)
            keyboardView.showSuggestions(Array(cands.prefix(3)).map { cased($0.word) })
            return
        }
        let lmMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        diag(String(format: "swipe: first pass %.0f ms, fused %.0f ms, %d cands, ctx %d words, first-pass %@ -> fused %@", firstPassMs, lmMs, cands.count, hyp.words.count - 1, best.word, hyp.words.last ?? ""))
        // Reconcile the text with the best hypothesis: with lookahead-1 only the
        // previous word can have changed. Keep each word's casing as typed.
        let newWords = hyp.words
        var keep = 0
        while keep < typed.count && keep < newWords.count - 1 && typed[keep].lowercased() == newWords[keep] { keep += 1 }
        let removed = typed[keep...]
        for _ in 0..<removed.reduce(0) { $0 + $1.count + 1 } { proxy.deleteBackward() }
        var inserted: [String] = []
        for (i, w) in newWords[keep...].enumerated() {
            let isLast = keep + i == newWords.count - 1
            let old = keep + i < typed.count ? typed[keep + i] : nil
            let word: String
            if isLast { word = cased(w) }
            else if let old, old.first?.isUppercase == true { word = w.prefix(1).uppercased() + w.dropFirst() }
            else { word = w }
            inserted.append(word)
        }
        proxy.insertText((keep == 0 && typed.isEmpty ? lead : "") + inserted.map { $0 + " " }.joined())
        typed = Array(typed[..<keep]) + inserted
        let last = inserted.last!
        // Alternatives for the suggestion bar: the fused beam's ranking of the last word.
        let alts = search.alternativesForLastWord()
        let shown = ([last.lowercased()] + alts.filter { $0 != last.lowercased() }).prefix(3)
        lastSwipe = (last, shown.map { w in Candidate(word: w, score: 0, acoustic: 0, unigram: 0, length: w.count) })
        keyboardView.showSuggestions(shown.map { $0 == last.lowercased() ? last : cased(with: last, $0) })
        if keyboardView.shiftState == .on { keyboardView.shiftState = .off }
    }

    /// Case `word` like `model` (capitalized / all caps / lower).
    private func cased(with model: String, _ word: String) -> String {
        if model == model.uppercased() && model.count > 1 { return word.uppercased() }
        if model.first?.isUppercase == true { return word.prefix(1).uppercased() + word.dropFirst() }
        return word
    }

    func keyboardViewDidToggleLM(_ view: KeyboardView) {
        lmEnabled.toggle()
        resetSentence()
        lastSwipe = nil
        diag("LM toggled \(lmEnabled ? "on" : "off")")
        showStatus()
    }

    func keyboardView(_ view: KeyboardView, didPickSuggestion index: Int) {
        guard let last = lastSwipe, index < last.candidates.count else { return }
        let before = proxy.documentContextBeforeInput ?? ""
        guard before.hasSuffix(last.word + " ") else { lastSwipe = nil; keyboardView.clearSuggestions(); return }
        for _ in 0..<(last.word.count + 1) { proxy.deleteBackward() }
        // Keep the capitalization of the word being replaced.
        var pick = last.candidates[index].word
        if last.word == last.word.uppercased() && last.word.count > 1 { pick = pick.uppercased() }
        else if last.word.first?.isUppercase == true { pick = pick.prefix(1).uppercased() + pick.dropFirst() }
        proxy.insertText(pick + " ")
        if let search, !typed.isEmpty {
            typed[typed.count - 1] = pick
            search.forceLastWord(pick.lowercased())
        }
        // Re-order so the picked word takes the middle pill; the others flank it.
        var rest = Array(last.candidates.prefix(3))
        let chosen = rest.remove(at: min(index, rest.count - 1))
        let shown = [chosen] + rest
        lastSwipe = (pick, shown)
        keyboardView.showSuggestions(shown.map { c in
            pick == pick.uppercased() && pick.count > 1 ? c.word.uppercased()
                : (pick.first?.isUppercase == true ? c.word.prefix(1).uppercased() + c.word.dropFirst() : c.word)
        })
    }
}
