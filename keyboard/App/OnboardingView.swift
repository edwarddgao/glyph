import SwiftUI

/// The app's accent: the keyboard's own trail blue, so buttons and the swipe
/// the user sees are one colour.
extension Color { static let glyph = Color(red: 0.17, green: 0.44, blue: 0.94) }

/// First launch: the keyboard itself spelling a word, one line, one number,
/// Start — then a three-sentence race, then the enable steps. No skip: the race
/// is how you learn to swipe on it, and its swipes are the first data every
/// new user contributes.
struct OnboardingView: View {
    var onDone: () -> Void
    @State private var step: Int = {   // UI tests: --onboarding-step N opens a step directly
        let a = CommandLine.arguments
        if let i = a.firstIndex(of: "--onboarding-step"), i + 1 < a.count, let n = Int(a[i + 1]) { return n }
        return 0
    }()
    @State private var showDetails = false

    var body: some View {
        NavigationStack {
            switch step {
            case 0: welcome
            case 1: RaceView(sentenceCount: 3, onFinished: { step = 2 }).navigationBarBackButtonHidden(true)
            default: EnableView(onDone: onDone)
            }
        }
        .tint(.glyph)
    }

    private var welcome: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 24)
            Text("Glyph").font(.system(size: 52, weight: .bold, design: .rounded)).tracking(-1)
            Text("Swipe. It gets the word.").font(.title2).foregroundStyle(.secondary).padding(.top, 2)
            Spacer(minLength: 20)
            HeroPad()
            Spacer(minLength: 20)
            Button { showDetails = true } label: {
                VStack(spacing: 3) {
                    Text("3 more words right in every 100").font(.title3.weight(.semibold))
                    Text("than Apple's keyboard, on the same swipes · see how").font(.footnote).foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("onboardingDetails")
            Spacer(minLength: 20)
            VStack(spacing: 10) {
                Button { step = 1 } label: { Text("Start").font(.headline).frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .accessibilityIdentifier("onboardingStart")
                Text("Three sentences to learn it. Practicing uploads those swipes, anonymously — nothing else, ever.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            .padding(.horizontal, 24)
            Spacer(minLength: 16)
        }
        .navigationBarHidden(true)
        .sheet(isPresented: $showDetails) { DetailsSheet() }
    }
}

/// The keyboard, spelling "glyph" on a loop, framed like a keyboard.
struct HeroPad: View {
    var word = "glyph"
    var body: some View {
        SwipePad(demoWord: word, pinned: false)
            .frame(height: SwipePad.heroHeight)
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .shadow(color: .black.opacity(0.12), radius: 18, y: 8)
            .padding(.horizontal, 10)
            .accessibilityHidden(true)
    }
}

/// Step three: add the keyboard in Settings.
struct EnableView: View {
    var onDone: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Spacer(minLength: 40)
            Text("Add the keyboard").font(.system(size: 34, weight: .bold, design: .rounded))
            Text("Three taps in Settings, then it is yours in every app.").font(.body).foregroundStyle(.secondary).padding(.top, 6)
            Spacer(minLength: 28)
            Steps()
            Spacer()
            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
            } label: { Label("Open Settings", systemImage: "gear").font(.headline).frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent).controlSize(.large)
            Button { onDone() } label: { Text("Done").frame(maxWidth: .infinity) }
                .buttonStyle(.plain).foregroundStyle(.secondary).padding(.top, 14)
                .accessibilityIdentifier("onboardingDone")
            Spacer(minLength: 16)
        }
        .padding(.horizontal, 24)
        .navigationBarHidden(true)
    }
}

/// The three Settings steps as large numbered rows.
struct Steps: View {
    static let items = [("Settings › General › Keyboard", "then Keyboards"), ("Add New Keyboard", "pick Glyph"), ("Hold 🌐 in any app", "and choose Glyph")]
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            ForEach(Array(Self.items.enumerated()), id: \.offset) { i, s in
                HStack(alignment: .firstTextBaseline, spacing: 14) {
                    Text("\(i + 1)").font(.system(size: 17, weight: .bold, design: .rounded)).foregroundStyle(.white)
                        .frame(width: 30, height: 30).background(Circle().fill(Color.glyph))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(s.0).font(.title3.weight(.medium))
                        Text(s.1).font(.subheadline).foregroundStyle(.secondary)
                    }
                }
            }
            Text("Glyph never asks for Full Access. The keyboard has no network access; nothing you type leaves the phone.")
                .font(.footnote).foregroundStyle(.secondary).padding(.top, 6)
        }
    }
}

/// One tap away from the hero line: the full benchmark, how it was measured,
/// what the race records, and where the source is.
struct DetailsSheet: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Words read right, same swipes").font(.headline)
                        BenchmarkTable()
                    }
                    Text(BenchmarkTable.caption).font(.footnote).foregroundStyle(.secondary)
                    Divider()
                    VStack(alignment: .leading, spacing: 6) {
                        Text("What practice records").font(.headline)
                        Text("Your swipes on the prompted words — finger path, timing, what the decoder read — under a random id with no name or account. That is the whole data set the decoder learns from, so it works for everyone, not only its author. The keyboard itself never sends anything anywhere.")
                            .font(.subheadline).foregroundStyle(.secondary)
                    }
                    Divider()
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Inside").font(.headline)
                        Text(AboutText.body).font(.subheadline).foregroundStyle(.secondary)
                        Link(destination: URL(string: AboutText.repo)!) { Label("Source, models and the research log", systemImage: "chevron.left.forwardslash.chevron.right") }
                            .font(.subheadline)
                        Text("build \(UploadConfig.build)").font(.caption2).foregroundStyle(.tertiary)
                    }
                }
                .padding(24)
            }
            .navigationTitle("How Glyph scores").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .tint(.glyph)
    }
}

/// The headline benchmark: replayed, byte-identical gestures on every keyboard,
/// top-1 committed words. Numbers from research/iphone/README.md.
struct BenchmarkTable: View {
    struct Row { let keyboard: String; let real: Double; let futo: Double }
    static let rows = [
        Row(keyboard: "Glyph", real: 77.9, futo: 93.4),
        Row(keyboard: "QuickPath (Apple)", real: 74.9, futo: 90.2),
        Row(keyboard: "SwiftKey", real: 69.0, futo: 85.9),
        Row(keyboard: "Gboard", real: 68.5, futo: 88.0),
    ]
    static let caption = "Real iPhone swipes: 542 words from one person, fast and sloppy. FUTO: 1,337 words from the public FUTO swipe corpus. The same recorded finger paths were replayed onto every keyboard on an iPhone 17 and scored per word. Against QuickPath: p = 0.09 on the real set, p < 0.001 on FUTO; against Gboard and SwiftKey p < 0.001 on both. More people's swipes — practice — are what settles it."

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
            GridRow {
                Text("").font(.caption)
                Text("real swipes").font(.caption).foregroundStyle(.secondary).gridColumnAlignment(.trailing)
                Text("FUTO").font(.caption).foregroundStyle(.secondary).gridColumnAlignment(.trailing)
            }
            ForEach(Self.rows, id: \.keyboard) { r in
                let me = r.keyboard == "Glyph"
                GridRow {
                    Text(r.keyboard).fontWeight(me ? .semibold : .regular).foregroundStyle(me ? Color.glyph : .primary)
                    Text(String(format: "%.1f%%", r.real)).monospacedDigit().fontWeight(me ? .semibold : .regular).foregroundStyle(me ? Color.glyph : .primary)
                    Text(String(format: "%.1f%%", r.futo)).monospacedDigit().fontWeight(me ? .semibold : .regular).foregroundStyle(me ? Color.glyph : .primary)
                }
            }
        }
    }
}
