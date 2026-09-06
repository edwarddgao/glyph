import SwiftUI
import GlyphCore

/// The practice screens: intro (what is collected), the race, the sentence
/// card, and the race summary. The keyboard pad stays fixed at the bottom.
struct RaceView: View {
    /// Leave practice (home's full-screen cover). Nil on the `--race` developer path.
    var onClose: (() -> Void)? = nil
    /// Onboarding: fewer sentences and a continue button instead of the again screen.
    var sentenceCount = 5
    var onFinished: (() -> Void)? = nil
    @StateObject private var game = RaceGame()
    static var isRequested: Bool { CommandLine.arguments.contains("--race") }

    var body: some View {
        VStack(spacing: 0) {
            switch game.phase {
            case .intro, .loading: loading
            case .racing: race
            case .raceDone: summary
            }
            if game.phase != .raceDone {
                SwipePad(onSwipe: { game.handleSwipe($0) }, onTap: { game.handleTap($0) })
                    .frame(height: SwipePad.height)
                    .accessibilityIdentifier("racePad")
            }
        }
        .navigationTitle("Practice")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)   // also disables the interactive pop gesture
        .toolbar {
            if game.phase == .racing, onFinished == nil, let onClose {
                ToolbarItem(placement: .topBarTrailing) { Button("Quit") { game.quit(); onClose() } }
            }
        }
        // Straight into the round: home already said what practice is.
        .onAppear { game.sentencesPerRace = sentenceCount; game.gentle = onFinished != nil; game.loadDecoder(); if game.phase == .intro { game.startRace() } }
    }

    /// Waiting for the decoder — or, if it never comes, a way out (onboarding
    /// has no other exit).
    private var loading: some View {
        VStack(spacing: 12) {
            Spacer()
            if game.loadFailed {
                Image(systemName: "exclamationmark.triangle").font(.title).foregroundStyle(.secondary)
                Text("The decoder could not load").font(.headline)
                Text(game.loadStatus).font(.caption).foregroundStyle(.tertiary).multilineTextAlignment(.center).padding(.horizontal, 32)
                if let onFinished {
                    Button { onFinished() } label: { Text("Continue to setup").frame(maxWidth: .infinity) }
                        .buttonStyle(.borderedProminent).controlSize(.large).padding(.horizontal, 24).padding(.top, 8)
                } else if let onClose {
                    Button("Back") { onClose() }.buttonStyle(.bordered).controlSize(.large).padding(.top, 8)
                }
            } else {
                ProgressView()
                Text(game.loadStatus).font(.footnote).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    // MARK: race

    /// Progress at the top (the clock runs unseen — the score comes at the end);
    /// one copy of the sentence and the verdict sit just above the keyboard, so
    /// the eyes never travel far from the finger.
    private var race: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("sentence \(game.sentenceIndex + 1) / \(game.sentences.count)").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if !game.lmReady { Text(game.loadStatus).font(.caption2).foregroundStyle(.tertiary) }
            }
            Spacer(minLength: 0)
            WordFlow(words: game.words, states: game.wordStates, costs: game.wordCosts, current: game.wordIndex, flash: game.flashWrong)
                .accessibilityElement(children: .ignore)
                .accessibilityIdentifier("raceSentence")
                .accessibilityLabel(game.words.joined(separator: " "))
                .accessibilityValue("\(game.wordIndex)/\(game.words.count) attempts \(game.raceAttempts)")
            HStack(alignment: .firstTextBaseline) {
                if let d = game.lastDecoded {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(d.hasPrefix("swipe") ? d : "\(d) — swipe it again").font(.subheadline).foregroundStyle(.red)
                        if let r = game.lastRead { Text("keyboard read “\(r)”").font(.caption).foregroundStyle(.secondary) }
                    }
                    .transition(.opacity)
                } else if game.busy {
                    Text("…").font(.subheadline).foregroundStyle(.secondary)
                } else {
                    Text(" ").font(.subheadline)
                }
                Spacer()
                if game.attemptsOnWord >= 2 {
                    Button("Skip word") { game.skipWord() }.font(.subheadline).buttonStyle(.bordered).controlSize(.small)
                }
            }
            .frame(minHeight: 28)
        }
        .padding(.horizontal).padding(.top, 8).padding(.bottom, 6)
    }

    // MARK: summary

    /// The headline numbers, scaling with the user's text size.
    @ScaledMetric(relativeTo: .largeTitle) private var bigNumber: CGFloat = 48
    /// A word tapped on the summary: its swipe against the ideal path.
    @State private var detail: WordDetail?
    struct WordDetail: Identifiable {
        let id: String
        let word: String, samples: [TouchSample], cost: Double?, decoded: String?
    }

    /// wpm and precision, then every sentence of the round with misreads marked;
    /// any swiped word opens its trace against the ideal path.
    private var summary: some View {
        ScrollView {
            VStack(spacing: 16) {
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    bigStat(String(format: "%.0f", game.raceWPM), unit: "wpm")
                    bigStat(String(format: "%.0f", game.racePrecision), unit: "% precision")
                }
                .padding(.top, 16)
                if game.raceWPM >= game.bestWPM && game.raceWPM > 0 { Label("new best", systemImage: "trophy.fill").foregroundStyle(.orange).font(.subheadline) }
                else if game.bestWPM > 0 { Text(String(format: "best %.0f wpm", game.bestWPM)).font(.subheadline).foregroundStyle(.secondary) }
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(game.roundSentences) { s in
                        RoundSentenceView(sentence: s) { i in
                            guard let p = s.samples[i] else { return }
                            detail = WordDetail(id: "\(s.id)-\(i)", word: s.words[i], samples: p, cost: s.costs[i], decoded: s.decoded[i])
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 8)
                Text("Tap a word to see your swipe against the ideal path.").font(.caption).foregroundStyle(.tertiary)
                uploadNote
                if let onFinished {
                    Button { onFinished() } label: { Text("Continue").frame(maxWidth: .infinity) }.buttonStyle(.borderedProminent).controlSize(.large)
                        .accessibilityIdentifier("raceContinue")
                } else {
                    Button { game.startRace() } label: { Text("Again").font(.headline).frame(maxWidth: .infinity) }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .accessibilityIdentifier("raceAgain")
                    if let onClose { Button("Done") { game.quit(); onClose() } }
                }
            }
            .padding(20)
        }
        .sheet(item: $detail) { d in SwipeDetailView(word: d.word, samples: d.samples, cost: d.cost, decoded: d.decoded) }
    }

    private func bigStat(_ value: String, unit: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(value).font(.system(size: bigNumber, weight: .bold, design: .rounded)).monospacedDigit()
            Text(unit).font(.subheadline).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

/// A finished sentence on the summary: words coloured by swipe precision,
/// skipped words struck through, and under a word Glyph misread, in red, what
/// it read instead. Swiped words are buttons.
struct RoundSentenceView: View {
    let sentence: RaceGame.RoundSentence
    var onTap: (Int) -> Void

    var body: some View {
        FlowLayout(spacing: 8) {
            ForEach(Array(sentence.words.enumerated()), id: \.offset) { i, w in
                let misread = sentence.decoded[i].map { RaceGame.norm($0) != RaceGame.norm(w) } ?? false
                Button { onTap(i) } label: {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(w)
                            .font(.system(.title3, design: .rounded))
                            .foregroundStyle(sentence.states[i] == .skipped ? Color.red : costColor(sentence.costs[i]))
                            .strikethrough(sentence.states[i] == .skipped)
                        if misread, let d = sentence.decoded[i] {
                            Text(d).font(.caption).foregroundStyle(.red)
                        }
                    }
                }
                .buttonStyle(.plain)
                .disabled(sentence.samples[i] == nil)
            }
        }
    }
}

extension RaceView {
    /// Upload state only when it needs attention; a success is silent.
    @ViewBuilder var uploadNote: some View {
        if !game.uploadStatus.isEmpty && !game.uploadStatus.hasPrefix("uploaded ✓") {
            Text(game.uploadStatus).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
    }
}

/// Swipe quality as a continuous hue: green at a perfect trace, through amber,
/// to red-orange at the acceptance cut (trace cost 6 per letter).
func costColor(_ cost: Double?) -> Color {
    let t = min(max((cost ?? 0) / GestureTrace.untracedCostPerLetter, 0), 1)
    return Color(hue: 0.36 - 0.33 * t, saturation: 0.72, brightness: 0.78)
}

/// The sentence as wrapped word chips: done green, current highlighted,
/// skipped red and struck through, pending plain.
struct WordFlow: View {
    let words: [String]
    let states: [RaceGame.WordState]
    var costs: [Double?] = []
    let current: Int
    let flash: Bool

    var body: some View {
        FlowLayout(spacing: 6) {
            ForEach(Array(words.enumerated()), id: \.offset) { i, w in
                Text(w)
                    .font(.system(.title2, design: .rounded).weight(i == current ? .bold : .regular))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(background(i), in: RoundedRectangle(cornerRadius: 6))
                    .foregroundStyle(foreground(i))
                    .strikethrough(states[i] == .skipped)
            }
        }
        .animation(.easeInOut(duration: 0.15), value: current)
    }

    private func background(_ i: Int) -> Color {
        if i == current { return flash ? Color.red.opacity(0.25) : Color.accentColor.opacity(0.18) }
        return .clear
    }
    private func foreground(_ i: Int) -> Color {
        switch states[i] {
        case .done: return costColor(i < costs.count ? costs[i] : nil)
        case .skipped: return .red
        case .pending: return i == current ? .primary : .secondary
        }
    }
}

/// Minimal left-to-right wrapping layout.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 360
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0
        for s in subviews {
            let sz = s.sizeThatFits(.unspecified)
            if x + sz.width > width && x > 0 { x = 0; y += rowH + spacing; rowH = 0 }
            x += sz.width + spacing; rowH = max(rowH, sz.height)
        }
        return CGSize(width: width, height: y + rowH)
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0
        for s in subviews {
            let sz = s.sizeThatFits(.unspecified)
            if x + sz.width > bounds.maxX && x > bounds.minX { x = bounds.minX; y += rowH + spacing; rowH = 0 }
            s.place(at: CGPoint(x: x, y: y), proposal: .unspecified)
            x += sz.width + spacing; rowH = max(rowH, sz.height)
        }
    }
}
