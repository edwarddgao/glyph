import Foundation
import SwiftUI
import UIKit
import GlyphCore

/// Practice (formerly SwipeRacer): swipe prompted sentences word by word, timed. A
/// word advances only when the decoder commits it (typeracer's rule), so the
/// gestures stay honest under speed pressure; every attempt — accepted or not —
/// is recorded with the prompted word as its label, and one record per
/// sentence goes to the capture server through `RaceStore`.
///
/// Acceptance is geometric, not a decode: the prompted word is known, so the
/// only question is whether the finger traced it — `GestureTrace`, the
/// decoder-independent alignment cost the training-label filter uses (#81),
/// per-letter cost ≤ 6. The shipped stack still decodes every swipe in the
/// background (AR first pass, fused search with the sentence's true preceding
/// words as context) and its reading is recorded, so each record also says
/// where the keyboard would have ranked the word — but the language model has
/// no say in whether a swipe counts.
struct Phrase { let tag: String; let text: String }

@MainActor
final class RaceGame: ObservableObject {
    enum Phase { case intro, loading, racing, sentenceDone, raceDone }
    enum WordState { case pending, done, skipped }

    @Published var phase: Phase = .intro
    @Published var loadStatus = ""
    @Published var sentences: [Phrase] = []
    @Published var sentenceIndex = 0
    @Published var words: [String] = []
    @Published var wordStates: [WordState] = []
    /// Trace cost per letter of the accepted swipe for each word (nil until done/skipped).
    @Published var wordCosts: [Double?] = []
    /// What the keyboard would have typed for each word (the decoder's reading of the
    /// accepted swipe) — shown so the real keyboard's behaviour is no surprise later.
    @Published var decodedWords: [String?] = []

    @Published var wordIndex = 0
    @Published var attemptsOnWord = 0
    @Published var lastDecoded: String? = nil       // shown in red after a miss (why the trace failed)
    @Published var lastRead: String? = nil          // what the keyboard's decoder read, for the same miss
    @Published var flashWrong = false
    @Published var elapsed: TimeInterval = 0
    @Published var uploadStatus = ""
    @Published var busy = false                     // a decode is in flight

    // race totals
    @Published var raceAttempts = 0
    @Published var raceAccepted = 0
    @Published var raceFirstTry = 0
    @Published var raceWords = 0
    @Published var raceSkipped = 0
    @Published var raceSeconds: TimeInterval = 0
    @Published var sentenceResults: [(wpm: Double, firstTry: Int, words: Int)] = []
    @AppStorage("race.bestWPM") var bestWPM: Double = 0
    @AppStorage("race.nick") var nick = ""
    @AppStorage("race.races") var racesPlayed = 0

    /// Sentences per race: 5 normally, 3 in onboarding.
    var sentencesPerRace = 5
    /// Onboarding: everyday words only, at most six per sentence — a first swipe should not meet "ensuing".
    var gentle = false
    /// Everyday : tail ratio 3 : 2 — the everyday share is the unbiased test set, the tail share buys coverage.
    var everydayPerRace: Int { max(1, Int((Double(sentencesPerRace) * 0.6).rounded())) }

    /// The prompt pool (`Resources/race_prompts.json`, scripts/build_race_prompts.py):
    /// real modern text, chosen for word coverage. Falls back to the phrase sets.
    struct Prompt: Decodable { let id: Int; let text: String; let source: String; let tag: String }
    struct PromptFile: Decodable { let version: Int; let sentences: [Prompt] }
    static let pool: [Prompt] = {
        guard let url = Bundle.main.url(forResource: "race_prompts", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let f = try? JSONDecoder().decode(PromptFile.self, from: data) else { return [] }
        return f.sentences
    }()
    /// Prompt ids this player has already raced, so the pool is walked without repeats.
    static var seen: Set<Int> {
        get { Set(UserDefaults.standard.string(forKey: "race.seen")?.split(separator: ",").compactMap { Int($0) } ?? []) }
        set { UserDefaults.standard.set(newValue.map(String.init).joined(separator: ","), forKey: "race.seen") }
    }
    private var promptIds: [Int?] = []
    private var promptSources: [String?] = []

    /// Everyday + tail from the pool, unseen first; shuffled.
    func drawRace() -> [(Phrase, Prompt?)] {
        let pool = Self.pool
        guard !pool.isEmpty else { return [] }
        var seen = Self.seen
        if seen.count >= pool.count - sentencesPerRace { seen = [] }
        let short: (Prompt) -> Bool = { !self.gentle || $0.text.split(separator: " ").count <= 6 }
        func draw(_ tag: String, _ k: Int) -> [Prompt] {
            var fresh = pool.filter { $0.tag == tag && short($0) && !seen.contains($0.id) }.shuffled()
            if fresh.count < k { fresh += pool.filter { $0.tag == tag && short($0) && seen.contains($0.id) }.shuffled() }
            return Array(fresh.prefix(k))
        }
        let picks = gentle ? draw("everyday", sentencesPerRace)
                           : (draw("everyday", everydayPerRace) + draw("tail", sentencesPerRace - everydayPerRace)).shuffled()
        seen.formUnion(picks.map { $0.id }); Self.seen = seen
        return picks.map { (Phrase(tag: $0.tag, text: $0.text), $0) }
    }
    /// UI tests: `--race-set N` races pool sentences N·5 … N·5+4 in order.
    static var fixedSet: Int? {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: "--race-set"), i + 1 < a.count, let n = Int(a[i + 1]), n >= 1 else { return nil }
        return n - 1
    }
    static let loader = DecoderLoader()
    private var decoder: SwipeDecoder?
    private var lm: CoreMLLanguageModel?
    private var priors: [Float]?
    private let decodeQueue = DispatchQueue(label: "race.decode", qos: .userInteractive)

    private var sentenceStart: Date? = nil
    private var firstTouch: Date? = nil
    private var timer: Timer?
    private var gestures: [[String: Any]] = []       // this sentence's attempts
    private var setIndex = 0
    private var attemptsPerWord: [Int] = []

    var lmReady: Bool { lm != nil }

    // MARK: lifecycle

    func loadDecoder() {
        guard decoder == nil else { return }
        loadStatus = "loading decoder…"
        DispatchQueue.global(qos: .utility).async { _ = Self.pool }   // decode the prompt pool (~150k sentences) off the main thread
        Self.loader.load { [weak self] r in
            Task { @MainActor in
                guard let self else { return }
                switch r {
                case .success(let d):
                    self.decoder = d
                    self.loadStatus = "decoder ready · loading language model…"
                    Self.loader.loadLM { [weak self] r in
                        Task { @MainActor in
                            guard let self else { return }
                            if case .success(let lm) = r { self.lm = lm; self.priors = Self.loader.priors; self.loadStatus = "ready" }
                            else { self.loadStatus = "ready (first pass only)" }
                        }
                    }
                case .failure(let e): self.loadStatus = "decoder failed: \(e)"
                }
            }
        }
    }

    func startRace() {
        loadDecoder()
        if let fixed = Self.fixedSet, !Self.pool.isEmpty {   // UI tests: deterministic slice of the pool
            let slice = Array(Self.pool.dropFirst(fixed * sentencesPerRace).prefix(sentencesPerRace))
            setIndex = fixed; sentences = slice.map { Phrase(tag: $0.tag, text: $0.text) }
            promptIds = slice.map { $0.id }; promptSources = slice.map { $0.source }
        } else {
            let drawn = drawRace()
            setIndex = 0; sentences = drawn.map { $0.0 }
            promptIds = drawn.map { $0.1?.id }; promptSources = drawn.map { $0.1?.source }
        }
        guard !sentences.isEmpty else { loadStatus = "decoder failed: no prompt pool in the bundle"; return }
        sentenceIndex = 0
        raceAttempts = 0; raceAccepted = 0; raceFirstTry = 0; raceWords = 0; raceSkipped = 0; raceSeconds = 0
        sentenceResults = []
        uploadStatus = !UploadConfig.enabled ? "no upload token in this build — records stay on the phone"
            : RaceStore.shared.pendingCount > 0 ? "\(RaceStore.shared.pendingCount) earlier records pending upload" : ""
        RaceStore.shared.onStatus = { [weak self] s in self?.uploadStatus = s }
        RaceStore.shared.flush()
        beginSentence()
    }

    private func beginSentence() {
        words = sentences[sentenceIndex].text.split(separator: " ").map(String.init)
        wordStates = Array(repeating: .pending, count: words.count)
        wordCosts = Array(repeating: nil, count: words.count)
        decodedWords = Array(repeating: nil, count: words.count)
        attemptsPerWord = Array(repeating: 0, count: words.count)
        wordIndex = 0; attemptsOnWord = 0; lastDecoded = nil; lastRead = nil; flashWrong = false
        gestures = []
        elapsed = 0; sentenceStart = nil; firstTouch = nil
        phase = decoder == nil ? .loading : .racing
        if decoder == nil {
            // wait for the loader, then race
            Task { @MainActor in
                while self.decoder == nil && !self.loadStatus.hasPrefix("decoder failed") { try? await Task.sleep(nanoseconds: 100_000_000) }
                if self.decoder != nil { self.phase = .racing }
            }
        }
    }

    func nextSentence() {
        sentenceIndex += 1
        if sentenceIndex >= sentences.count { finishRace() } else { beginSentence() }
    }

    private func finishRace() {
        racesPlayed += 1
        if raceWPM > bestWPM { bestWPM = raceWPM }
        phase = .raceDone
    }


    func quit() {
        timer?.invalidate(); timer = nil
        phase = .intro
    }

    // MARK: scoring

    var currentWord: String? { wordIndex < words.count ? words[wordIndex] : nil }

    /// Typeracer's convention: characters (spaces included) / 5 per minute.
    func wpm(chars: Int, seconds: TimeInterval) -> Double { seconds > 0 ? Double(chars) / 5.0 / (seconds / 60.0) : 0 }
    var sentenceChars: Int { sentences[sentenceIndex].text.count }
    var liveWPM: Double { wpm(chars: sentenceCharsDone, seconds: elapsed) }
    private var sentenceCharsDone: Int { words.prefix(wordIndex).reduce(0) { $0 + $1.count + 1 } }
    var raceWPM: Double { wpm(chars: sentences.prefix(sentenceResults.count).reduce(0) { $0 + $1.text.count }, seconds: raceSeconds) }
    var raceAccuracy: Double { raceAttempts > 0 ? Double(raceAccepted) / Double(raceAttempts) * 100 : 0 }
    var raceFirstTryPct: Double { raceWords > 0 ? Double(raceFirstTry) / Double(raceWords) * 100 : 0 }

    private func startClock() {
        guard sentenceStart == nil else { return }
        sentenceStart = Date(); firstTouch = sentenceStart
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let s = self.sentenceStart, self.phase == .racing else { return }
                self.elapsed = Date().timeIntervalSince(s)
            }
        }
    }

    // MARK: gestures

    /// Single-letter words ("i", "a") are typed with a tap, as on any keyboard;
    /// the tap is recorded as an attempt (no path, `input: "tap"`) and never
    /// enters the swipe corpus. A tap on a longer word is just a hint.
    func handleTap(_ ch: Character) {
        guard phase == .racing, let word = currentWord, !busy else { return }
        guard word.count == 1 else { lastDecoded = "swipe the word, don't tap"; lastRead = nil; return }
        startClock()
        let idx = wordIndex, attempt = attemptsPerWord[idx] + 1
        let accepted = Self.norm(String(ch)) == Self.norm(word)
        gestures.append(["word": word, "word_idx": idx, "attempt": attempt, "accepted": accepted, "input": "tap", "tapped": String(ch)])
        attemptsPerWord[idx] = attempt
        raceAttempts += 1
        if accepted {
            raceAccepted += 1
            if attempt == 1 { raceFirstTry += 1 }
            wordStates[idx] = .done; wordCosts[idx] = 0; decodedWords[idx] = String(ch)
            lastDecoded = nil; lastRead = nil
            advance()
        } else {
            attemptsOnWord = attempt
            lastDecoded = "tapped “\(ch)”, the word is “\(word)”"; lastRead = nil
            flashWrong = true
            Task { @MainActor in try? await Task.sleep(nanoseconds: 350_000_000); self.flashWrong = false }
        }
    }

    func handleSwipe(_ samples: [TouchSample]) {
        guard phase == .racing, let decoder, let word = currentWord, !busy else { return }
        if word.count == 1 { lastDecoded = "tap “\(word)” — one letter, no swipe"; lastRead = nil; return }
        startClock()
        busy = true
        let idx = wordIndex, attempt = attemptsPerWord[idx] + 1
        let context = words.prefix(idx).joined(separator: " ")
        let lm = self.lm, priors = self.priors
        decodeQueue.async { [weak self] in
            // geometric verdict first: does the path trace the prompted word?
            let trace = GestureTrace(samples: samples)
            let traceCost = trace.costPerLetter(of: word)
            let rejection = trace.rejection(for: word)
            let t0 = CFAbsoluteTimeGetCurrent()
            var cands: [Candidate] = []
            do { cands = try decoder.decode(samples) } catch { }
            let firstMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            var fused: String? = nil
            if let lm, !cands.isEmpty {
                let search = SentenceSearch(lm: lm)
                if let priors {
                    let trie = decoder.trie
                    search.priorLookup = { w in
                        guard let n = trie.node(for: w), trie.isWord(n) else { return nil }
                        let p = priors[Int(n)]; return p.isNaN ? nil : Double(p)
                    }
                }
                search.reset(prefix: context)
                fused = (try? search.step(candidates: cands.map { ($0.word, $0.score) }))?.words.last
            }
            let decoded = fused ?? cands.first?.word
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            Task { @MainActor in
                guard let self else { return }
                self.busy = false
                guard self.phase == .racing, self.wordIndex == idx else { return }
                let accepted = rejection == nil
                let decoderRight = decoded.map { Self.norm($0) == Self.norm(word) } ?? false
                self.gestures.append([
                    "word": word, "word_idx": idx, "attempt": attempt, "accepted": accepted,
                    "trace_cost": traceCost, "rejection": rejection ?? "", "path_key_widths": trace.pathKeyWidths,
                    "decoder_right": decoderRight,
                    "x": samples.map { $0.x }, "y": samples.map { $0.y }, "t": samples.map { $0.t },
                    "aspect": Self.aspect, "first_pass": Array(cands.prefix(5)).map { $0.word },
                    "fused": fused ?? "", "decoded": decoded ?? "", "ms": Int(ms), "first_pass_ms": Int(firstMs),
                ])
                self.attemptsPerWord[idx] = attempt
                self.raceAttempts += 1
                if accepted {
                    self.raceAccepted += 1
                    if attempt == 1 { self.raceFirstTry += 1 }
                    self.wordStates[idx] = .done
                    self.wordCosts[idx] = traceCost
                    self.decodedWords[idx] = decoded
                    self.lastDecoded = nil; self.lastRead = nil
                    self.advance()
                } else {
                    self.attemptsOnWord = attempt
                    self.lastDecoded = rejection == "aborted" ? "too short for “\(word)”" : "didn’t trace “\(word)”"
                    self.lastRead = decoded
                    self.flashWrong = true
                    Task { @MainActor in try? await Task.sleep(nanoseconds: 350_000_000); self.flashWrong = false }
                }
            }
        }
    }

    /// Offered after two misses; the word is recorded as skipped (its attempts are kept).
    func skipWord() {
        guard phase == .racing, wordIndex < words.count else { return }
        wordStates[wordIndex] = .skipped
        wordCosts[wordIndex] = nil
        raceSkipped += 1
        lastDecoded = nil
        advance()
    }

    private func advance() {
        wordIndex += 1
        attemptsOnWord = 0
        raceWords += 1
        if wordIndex >= words.count { endSentence() }
    }

    private func endSentence() {
        timer?.invalidate(); timer = nil
        let seconds = sentenceStart.map { Date().timeIntervalSince($0) } ?? 0
        elapsed = seconds
        raceSeconds += seconds
        let firstTry = zip(attemptsPerWord, wordStates).filter { $0 == 1 && $1 == .done }.count
        sentenceResults.append((wpm(chars: sentenceChars, seconds: seconds), firstTry, words.count))
        phase = .sentenceDone
        upload(seconds: seconds)
    }

    private func upload(seconds: TimeInterval) {
        let p = sentences[sentenceIndex]
        let screen = UIScreen.main.bounds
        let rec: [String: Any] = [
            "kind": "race", "session": RaceStore.shared.session, "nick": nick,
            "set": setIndex + 1, "sentence": p.text, "tag": p.tag,
            "prompt_id": promptIds[sentenceIndex] as Any, "prompt_source": promptSources[sentenceIndex] ?? "",
            "onboarding": sentencesPerRace == 3,
            "ts": Int(Date().timeIntervalSince1970 * 1000), "ms": Int(seconds * 1000),
            "wpm": sentenceResults.last?.wpm ?? 0,
            "words": words.enumerated().map { i, w in
                ["word": w, "word_idx": i, "attempts": attemptsPerWord[i], "accepted": wordStates[i] == .done, "skipped": wordStates[i] == .skipped] as [String: Any] },
            "gestures": gestures,
            "device": RaceStore.shared.deviceModelName, "ios": UIDevice.current.systemVersion,
            "screen": ["w": screen.width, "h": screen.height],
            "grid": ["width": NativeMetrics.gridWidth(screen.width), "rowPitch": NativeMetrics.rowPitch, "aspect": Self.aspect],
            "lm": lm != nil, "ua": "swipe-app-race",
            "acceptance": ["rule": "trace", "cost_per_letter_max": GestureTrace.untracedCostPerLetter],
        ]
        RaceStore.shared.save(rec)
    }

    /// Letter-grid width / height in points on this screen (capture records carry the same field).
    static var aspect: Double {
        let w = UIScreen.main.bounds.width
        return Double(NativeMetrics.gridWidth(w) / (3 * NativeMetrics.rowPitch))
    }
    static func norm(_ s: String) -> String { s.lowercased().filter { $0.isLetter } }
}

extension RaceStore {
    var deviceModelName: String { RaceStore.deviceModel }
}
