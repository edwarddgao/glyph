import SwiftUI
import GlyphCore

/// The practice screens: intro (what is collected), the race, the sentence
/// card, and the race summary. The keyboard pad stays fixed at the bottom.
struct RaceView: View {
    var onClose: (() -> Void)? = nil
    /// Onboarding: fewer sentences and a continue button instead of the again screen.
    var sentenceCount = 5
    var onFinished: (() -> Void)? = nil
    @StateObject private var game = RaceGame()
    @AppStorage("record.server") private var server = UploadConfig.defaultURL
    @State private var showSettings = false
    static var isRequested: Bool { CommandLine.arguments.contains("--race") }

    var body: some View {
        VStack(spacing: 0) {
            switch game.phase {
            case .intro: intro
            case .loading: VStack { Spacer(); ProgressView(); Text(game.loadStatus).font(.footnote).foregroundStyle(.secondary).padding(.top, 8); Spacer() }
            case .racing: race
            case .sentenceDone: sentenceCard
            case .raceDone: summary
            }
            if game.phase == .racing || game.phase == .loading {
                ZStack(alignment: .top) {
                    SwipePad(onSwipe: { game.handleSwipe($0) }, onTap: { game.handleTap($0) })
                        .frame(height: SwipePad.height)
                        .accessibilityIdentifier("racePad")
                    // The target sentence in the suggestion bar's slot, current word centred.
                    SentenceBar(words: game.words, states: game.wordStates, costs: game.wordCosts, current: game.wordIndex, flash: game.flashWrong)
                        .frame(height: SwipePad.barHeight)
                        .accessibilityHidden(true)
                }
            }
        }
        .navigationTitle(game.phase == .intro ? "" : "Practice")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)   // also disables the interactive pop gesture
        .toolbar {
            if game.phase == .racing && onFinished == nil {
                ToolbarItem(placement: .topBarTrailing) { Button("quit") { game.quit() } }
            } else if game.phase == .intro {
                if let onClose { ToolbarItem(placement: .topBarLeading) { Button("close") { onClose() } } }
                ToolbarItem(placement: .topBarTrailing) { Button { showSettings = true } label: { Image(systemName: "gearshape") } }
            }
        }
        .sheet(isPresented: $showSettings) {
            NavigationStack {
                Form {
                    Section("Nickname (optional)") { TextField("shown on the summary only", text: $game.nick).autocorrectionDisabled().textInputAutocapitalization(.never) }
                    Section("Upload server") { TextField("http://…:8765/save", text: $server).autocorrectionDisabled().textInputAutocapitalization(.never).font(.footnote) }
                    Section { Text("Anonymous id \(RaceStore.shared.session) · \(RaceStore.shared.pendingCount) records waiting to upload").font(.footnote).foregroundStyle(.secondary) }
                }
                .navigationTitle("Settings").toolbar { ToolbarItem(placement: .confirmationAction) { Button("done") { showSettings = false } } }
            }
        }
        .onAppear { game.sentencesPerRace = sentenceCount; game.gentle = onFinished != nil; game.loadDecoder(); if onFinished != nil && game.phase == .intro { game.startRace() } }
    }

    // MARK: intro

    private var intro: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 24)
            Text("Practice").font(.system(size: 40, weight: .bold, design: .rounded)).tracking(-1)
            Text("Swipe the words of five sentences, timed. A word counts when your finger traced it.").font(.body).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).padding(.horizontal, 32).padding(.top, 4)
            Spacer(minLength: 20)
            HeroPad(word: "practice")
            Spacer(minLength: 20)
            if game.bestWPM > 0 {
                Text(String(format: "best %.0f wpm · %d sessions", game.bestWPM, game.racesPlayed)).font(.subheadline).foregroundStyle(.secondary)
            }
            Spacer(minLength: 16)
            VStack(spacing: 10) {
                Button { game.startRace() } label: {
                    Text(game.loadStatus.hasPrefix("ready") || game.loadStatus.isEmpty ? "Start · \(sentenceCount) sentences" : game.loadStatus)
                        .font(.headline).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(game.loadStatus.hasPrefix("decoder failed"))
                .accessibilityIdentifier("raceStart")
                Text(game.uploadStatus.isEmpty ? "Swipes on the prompted words are uploaded anonymously." : game.uploadStatus)
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)
            Spacer(minLength: 16)
        }
    }

    // MARK: race

    private var race: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("sentence \(game.sentenceIndex + 1) / \(game.sentences.count)").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f s", game.elapsed)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                Text(String(format: "%.0f wpm", game.liveWPM)).font(.caption.monospacedDigit().bold())
            }
            WordFlow(words: game.words, states: game.wordStates, costs: game.wordCosts, current: game.wordIndex, flash: game.flashWrong)
                .accessibilityElement(children: .ignore)
                .accessibilityIdentifier("raceSentence")
                .accessibilityLabel(game.words.joined(separator: " "))
                .accessibilityValue("\(game.wordIndex)/\(game.words.count) attempts \(game.raceAttempts)")
            TypedLine(words: game.words, decoded: game.decodedWords)
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
                    Button("skip word") { game.skipWord() }.font(.subheadline).buttonStyle(.bordered).controlSize(.small)
                }
            }
            .frame(minHeight: 28)
            Spacer(minLength: 0)
            if !game.lmReady { Text(game.loadStatus).font(.caption2).foregroundStyle(.tertiary) }
        }
        .padding(.horizontal).padding(.top, 8)
    }

    // MARK: cards

    private var sentenceCard: some View {
        let r = game.sentenceResults.last!
        return VStack(spacing: 14) {
            Spacer()
            Text(String(format: "%.0f wpm", r.wpm)).font(.system(size: 56, weight: .bold, design: .rounded))
            Text(String(format: "%.0f%% accuracy · %d of %d words on the first swipe", Double(r.firstTry) / Double(max(r.words, 1)) * 100, r.firstTry, r.words))
                .font(.subheadline).foregroundStyle(.secondary)
            WordFlow(words: game.words, states: game.wordStates, costs: game.wordCosts, current: -1, flash: false).padding(.horizontal)
            TypedLine(words: game.words, decoded: game.decodedWords).padding(.horizontal)
            Button(game.sentenceIndex + 1 < game.sentences.count ? "next sentence" : "finish") { game.nextSentence() }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .accessibilityIdentifier("raceNext")
            Text(game.uploadStatus).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding()
    }

    private var summary: some View {
        VStack(spacing: 14) {
            Spacer()
            Text(String(format: "%.0f wpm", game.raceWPM)).font(.system(size: 56, weight: .bold, design: .rounded))
            if game.raceWPM >= game.bestWPM && game.raceWPM > 0 { Label("new best", systemImage: "trophy.fill").foregroundStyle(.orange) }
            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 6) {
                GridRow { Text("accuracy"); Text(String(format: "%.0f%%  (%d / %d words on the first swipe)", game.raceFirstTryPct, game.raceFirstTry, game.raceWords)).bold() }
                GridRow { Text("skipped"); Text("\(game.raceSkipped)").bold() }
                GridRow { Text("time"); Text(String(format: "%.0f s", game.raceSeconds)).bold() }
            }
            .font(.subheadline)
            Text(game.uploadStatus).font(.footnote).foregroundStyle(.secondary)
            if let onFinished {
                Button { onFinished() } label: { Text("Continue").frame(maxWidth: .infinity) }.buttonStyle(.borderedProminent).controlSize(.large)
                    .accessibilityIdentifier("raceContinue")
            } else {
                Button("again") { game.startRace() }.buttonStyle(.borderedProminent).controlSize(.large)
                Button("done") { game.quit() }
            }
            Spacer()
        }
        .padding()
    }
}

/// What the keyboard would have typed so far: the decoder's reading of each
/// accepted swipe, misreadings in red — the user meets the keyboard's mistakes
/// here, not in a text message later.
struct TypedLine: View {
    let words: [String]
    let decoded: [String?]
    var body: some View {
        let done = zip(words, decoded).filter { $0.1 != nil }
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: "keyboard").font(.caption).foregroundStyle(.tertiary)
            if done.isEmpty {
                Text("what the keyboard types").font(.subheadline).foregroundStyle(.tertiary)
            } else {
                FlowLayout(spacing: 5) {
                    ForEach(Array(done.enumerated()), id: \.offset) { _, pair in
                        let d = pair.1 ?? ""
                        Text(d).font(.subheadline).foregroundStyle(RaceGame.norm(d) == RaceGame.norm(pair.0) ? Color.secondary : Color.red)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Swipe quality as a continuous hue: green at a perfect trace, through amber,
/// to red-orange at the acceptance cut (trace cost 6 per letter).
func costColor(_ cost: Double?) -> Color {
    let t = min(max((cost ?? 0) / GestureTrace.untracedCostPerLetter, 0), 1)
    return Color(hue: 0.36 - 0.33 * t, saturation: 0.72, brightness: 0.78)
}

/// One line of the sentence in the keyboard's suggestion-bar slot, scrolled so
/// the current word sits in the middle — read the next word without looking up.
struct SentenceBar: View {
    let words: [String]
    let states: [RaceGame.WordState]
    var costs: [Double?] = []
    let current: Int
    let flash: Bool
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    Color.clear.frame(width: 120, height: 1)          // lets the first word centre
                    ForEach(Array(words.enumerated()), id: \.offset) { i, w in
                        Text(w)
                            .font(.system(size: 20, weight: i == current ? .semibold : .regular))
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(i == current ? (flash ? Color.red.opacity(0.3) : pill) : .clear, in: RoundedRectangle(cornerRadius: 6))
                            .foregroundStyle(color(i))
                            .id(i)
                    }
                    Color.clear.frame(width: 120, height: 1)
                }
                .padding(.horizontal, 8)
            }
            .frame(maxHeight: .infinity)
            .onAppear { proxy.scrollTo(current, anchor: .center) }
            .onChange(of: current) { _, c in withAnimation(.easeInOut(duration: 0.2)) { proxy.scrollTo(c, anchor: .center) } }
            .onChange(of: words) { _, _ in proxy.scrollTo(0, anchor: .center) }
        }
    }

    private var pill: Color { scheme == .dark ? Color(white: 118 / 255) : Color(red: 230 / 255, green: 231 / 255, blue: 235 / 255) }
    private func color(_ i: Int) -> Color {
        switch states[i] {
        case .done: return costColor(i < costs.count ? costs[i] : nil)
        case .skipped: return .red
        case .pending: return i == current ? .primary : .secondary
        }
    }
}

/// The sentence as wrapped word chips: done green, current highlighted,
/// skipped red, pending plain.
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
                    .font(.system(size: 22, weight: i == current ? .bold : .regular, design: .rounded))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(background(i), in: RoundedRectangle(cornerRadius: 6))
                    .foregroundStyle(foreground(i))
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
