import SwiftUI

/// Glyph: a swipe keyboard whose decoder runs entirely on the phone.
///
/// First launch is the onboarding: what Glyph is and how it scores, a
/// three-sentence practice run that teaches swiping (and records the swipes), then
/// the steps to enable the keyboard. After that the home screen offers the
/// race, the enable steps and the benchmark. Developer screens are reached
/// only by launch argument: `--bench` (replay benchmark field), `--lm-probe`
/// (language-model memory probe), `--race` (straight into practice).
@main
struct GlyphApp: App {
    var body: some Scene {
        WindowGroup {
            Group {
                if BenchView.isRequested { BenchView() }
                else if RaceView.isRequested { NavigationStack { RaceView() } }
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
        if onboarded && !forced { HomeView() } else { OnboardingView(onDone: { onboarded = true; forced = false }) }
    }
}

struct HomeView: View {
    @State private var showRace = false
    @State private var showEnable = false
    @State private var showDetails = false
    @State private var probe = LMProbe.isRequested ? "running LM probe…" : ""
    @AppStorage("race.bestWPM") private var bestWPM: Double = 0
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
            HeroPad()
            Spacer(minLength: 24)
            VStack(spacing: 12) {
                // Full-screen, not pushed: a swipe that starts on q/a/z would
                // otherwise be taken as the navigation stack's back gesture.
                Button { showRace = true } label: {
                    VStack(spacing: 2) {
                        Text("Practice").font(.headline)
                        Text(bestWPM > 0 ? String(format: "best %.0f wpm · %d sessions", bestWPM, racesPlayed) : "five sentences, timed")
                            .font(.caption).opacity(0.85)
                    }
                    .frame(maxWidth: .infinity).padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .accessibilityIdentifier("homePractice")
                Button { showEnable = true } label: { Text("Add the keyboard in Settings").font(.headline).frame(maxWidth: .infinity).padding(.vertical, 4) }
                    .buttonStyle(.bordered).controlSize(.large)
                    .accessibilityIdentifier("homeEnable")
                Text("Every session teaches the decoder. Swipes on prompted words only, anonymously.")
                    .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center).padding(.top, 4)
            }
            .padding(.horizontal, 24)
            Spacer(minLength: 24)
        }
        .tint(.glyph)
        .fullScreenCover(isPresented: $showRace) { NavigationStack { RaceView(onClose: { showRace = false }) }.tint(.glyph) }
        .sheet(isPresented: $showEnable) {
            NavigationStack { EnableView(onDone: { showEnable = false }) }.presentationDetents([.large]).tint(.glyph)
        }
        .sheet(isPresented: $showDetails) { DetailsSheet() }
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
