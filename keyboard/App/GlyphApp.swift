import SwiftUI

/// Glyph: a swipe keyboard whose decoder runs entirely on the phone.
///
/// First launch is the onboarding: what Glyph is and how it scores, a
/// three-sentence practice run that teaches swiping (and records the swipes), then
/// the steps to enable the keyboard. After that the home screen offers the
/// race, the enable steps and, once the keyboard is on, a field to try it in.
/// Developer screens are reached only by launch argument: `--bench` (replay
/// benchmark field), `--lm-probe` (language-model memory probe), `--race`
/// (straight into practice), `--debug` (the LAN upload-server field in the info
/// sheet), `--onboarded` / `--onboarding` (UI tests: skip or force the
/// first-launch flow), `--no-upload` (records stay on the device).
@main
struct GlyphApp: App {
    var body: some Scene {
        WindowGroup {
            Group {
                if BenchView.isRequested { BenchView() }
                else if RaceView.isRequested { NavigationStack { RaceView(onClose: { exit(0) }) } }   // developer/test path
                else { RootView() }
            }
            .tint(.glyph)
        }
    }
}

struct RootView: View {
    init() { diag("app start, build \(UploadConfig.build)") }
    @AppStorage("glyph.onboarded") private var onboarded = false
    @State private var forced = CommandLine.arguments.contains("--onboarding")   // UI tests
    var body: some View {
        // `--onboarded`: UI tests and screenshots go straight to the home screen.
        if (onboarded || CommandLine.arguments.contains("--onboarded")) && !forced { HomeView() }
        else { OnboardingView(onDone: { onboarded = true; forced = false }) }
    }
}

/// Whether Glyph is in the user's keyboard list. Settings keeps the list in
/// the `AppleKeyboards` default, which the containing app can read.
enum KeyboardStatus {
    static var isEnabled: Bool {
        let ids = UserDefaults.standard.object(forKey: "AppleKeyboards") as? [String] ?? []
        return ids.contains { $0.hasPrefix("com.edwardgao.glyph.keyboard") }
    }
}

/// `--debug`: developer fields (the LAN upload server) in the info sheet.
enum Debug { static let enabled = CommandLine.arguments.contains("--debug") }

/// Home. Before the keyboard is enabled: the demo pad and the enable button.
/// After: your scores, a field to type in, Practice. One screen, no intro.
struct HomeView: View {
    @State private var showRace = false
    @State private var showEnable = false
    @State private var showDetails = false
    @State private var enabled = KeyboardStatus.isEnabled
    @State private var tryText = ""
    @FocusState private var tryFocused: Bool
    @Environment(\.scenePhase) private var scenePhase
    @State private var probe = LMProbe.isRequested ? "running LM probe…" : ""
    @AppStorage("race.bestWPM") private var bestWPM: Double = 0
    @AppStorage("race.lastWPM") private var lastWPM: Double = 0
    @AppStorage("race.lastPrecision") private var lastPrecision: Double = 0
    @AppStorage("race.races") private var racesPlayed = 0

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 20)
            HStack(alignment: .firstTextBaseline) {
                Text("Glyph").font(.system(size: 40, weight: .bold, design: .rounded)).tracking(-1)
                Spacer()
                Button { showDetails = true } label: { Image(systemName: "info.circle").font(.title2) }.buttonStyle(.plain).foregroundStyle(.secondary)
                    .accessibilityIdentifier("homeDetails")
            }
            .padding(.horizontal, 24)
            if LMProbe.isRequested {
                Text(probe).font(.system(size: 11, design: .monospaced)).padding(.horizontal, 24).onAppear { LMProbe.run { probe = $0 } }
            }
            Spacer(minLength: 16)
            if enabled {
                if racesPlayed > 0 { scores.padding(.horizontal, 24).padding(.bottom, 20) }
                // The keyboard is on: the field is the demo now. Hold 🌐 and pick Glyph.
                TextField("Try it here — hold 🌐 and pick Glyph", text: $tryText, axis: .vertical)
                    .lineLimit(4...8)
                    .textInputAutocapitalization(.sentences)
                    .focused($tryFocused)
                    .padding(14)
                    .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .padding(.horizontal, 24)
                    .accessibilityIdentifier("homeTry")
            } else {
                HeroPad()
            }
            Spacer(minLength: 24)
            VStack(spacing: 12) {
                // Full-screen, not pushed: a swipe that starts on q/a/z would
                // otherwise be taken as the navigation stack's back gesture.
                if enabled { practiceButton.buttonStyle(.borderedProminent) } else {
                    Button { showEnable = true } label: { Text("Add the keyboard in Settings").font(.headline).frame(maxWidth: .infinity).padding(.vertical, 4) }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                        .accessibilityIdentifier("homeEnable")
                    practiceButton.buttonStyle(.bordered)
                }
                if !enabled || racesPlayed == 0 {
                    Text("Every session teaches the decoder. Swipes on prompted words only, anonymously.")
                        .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center).padding(.top, 4)
                }
            }
            .padding(.horizontal, 24)
            Spacer(minLength: 24)
        }
        .contentShape(Rectangle())
        .onTapGesture { tryFocused = false }   // tap anywhere else to put the keyboard away
        .tint(.glyph)
        .onChange(of: scenePhase) { _, p in if p == .active { enabled = KeyboardStatus.isEnabled } }
        .onChange(of: showEnable) { _, shown in if !shown { enabled = KeyboardStatus.isEnabled } }
        .fullScreenCover(isPresented: $showRace) { NavigationStack { RaceView(onClose: { showRace = false }) }.tint(.glyph) }
        .sheet(isPresented: $showEnable) {
            NavigationStack { EnableView(onDone: { showEnable = false }) }.presentationDetents([.large]).tint(.glyph)
        }
        .sheet(isPresented: $showDetails) { DetailsSheet() }
    }

    /// Best round, last round, and the last round's swipe precision (trace cost, 0–100).
    private var scores: some View {
        HStack(alignment: .firstTextBaseline, spacing: 0) {
            stat(String(format: "%.0f", bestWPM), unit: "wpm", label: "best")
            stat(String(format: "%.0f", lastWPM), unit: "wpm", label: "last round")
            stat(String(format: "%.0f", lastPrecision), unit: "%", label: "precision")
        }
        .accessibilityElement(children: .combine)
    }

    private func stat(_ value: String, unit: String, label: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value).font(.system(size: 34, weight: .bold, design: .rounded)).monospacedDigit()
                Text(unit).font(.subheadline).foregroundStyle(.secondary)
            }
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var practiceButton: some View {
        Button { tryFocused = false; showRace = true } label: {
            VStack(spacing: 2) {
                Text("Practice").font(.headline)
                Text(racesPlayed > 0 ? "five sentences · round \(racesPlayed + 1)" : "five sentences, timed").font(.caption).opacity(0.85)
            }
            .frame(maxWidth: .infinity).padding(.vertical, 4)
        }
        .controlSize(.large)
        .accessibilityIdentifier("homePractice")
    }
}

enum AboutText {
    static let repo = "https://github.com/edwarddgao/glyph"
    static let privacy = "https://swipe-upload.swipe-edwardgao.workers.dev/privacy"
    static let body = """
    Glyph decodes a swipe with a small transformer trained on public swipe corpora, a 300k-word trie and a sentence \
    language model (distilgpt2), all running on the phone's CPU. The benchmark replays the same recorded finger paths \
    onto each keyboard and counts the words it commits, paired word by word. Everything — the model, the training \
    code, the benchmark harness and the lab notebook — is open source under MIT.
    """
}
