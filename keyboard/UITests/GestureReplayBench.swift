import XCTest

/// Replays recorded swipes (canonical coordinates) onto the active keyboard
/// and records what it committed — the same input for every keyboard.
///
/// Configuration through the test runner's environment:
///   BENCH_KEYBOARD  quickpath | gboard | swiftkey | swipe | swipe-nolm   (which keyboard to switch to)
///   BENCH_SOURCE    capture | futo | all              (which sentences)
///   BENCH_LIMIT     max sentences (default all)
///   BENCH_SHARD     "i/n": take every n-th sentence starting at i (fan out over simulators)
///   BENCH_SPEED     time scale for replay (1.0 = recorded timing)
///
/// Results are printed one JSON object per line prefixed with "BENCH ",
/// in the shape `benchmark_keyboards.py` scores (kind "bench").
final class GestureReplayBench: XCTestCase {
    struct Gesture: Decodable { let x, y, t: [Double] }
    struct Sentence: Decodable { let source, session: String; let set: Int; let tag: String; let words: [String]; let gestures: [Gesture] }
    struct Data_: Decodable { let sentences: [Sentence] }

    /// Where a keyboard's letter grid sits on this screen, in points: the left
    /// edge of the q cell, the width of the 10 cells, the top of the first row
    /// cell and the row pitch. Canonical (x, y) maps to
    /// (left + x·width, top + y·3·rowPitch).
    struct Grid { let left, width, top, rowPitch: CGFloat }

    /// Measured with tools/measure_layout.py on the iPhone 17 (402 × 874 pt),
    /// predictive bar showing. Swipe is laid out from the same numbers, and
    /// its status bar gives its top at runtime.
    static let quickPathGrid = Grid(left: 20.0 / 3.0 - 3, width: 10 * (402 - 2 * 20.0 / 3.0 + 6) / 10, top: 591 - 5.5, rowPitch: 54)

    let env = ProcessInfo.processInfo.environment

    func log(_ obj: [String: Any]) {
        let d = try! JSONSerialization.data(withJSONObject: obj)
        print("BENCH " + String(decoding: d, as: UTF8.self))
    }

    /// Long-press 123 until the Swipe status bar reports the wanted LM state.
    func setSwipeLM(_ app: XCUIApplication, enabled: Bool) {
        let status = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch
        for _ in 0..<3 {
            guard status.waitForExistence(timeout: 20) else { return }
            let off = status.label.contains("LM off")
            if off == !enabled { return }
            if enabled && status.label.contains("LM loading") { sleep(2); continue }
            app.buttons["layer"].press(forDuration: 1.0)
            sleep(1)
        }
    }

    func switchKeyboard(_ app: XCUIApplication, to name: String) {
        let labels = ["quickpath": "English", "gboard": "Gboard", "swiftkey": "SwiftKey", "swipe": "Glyph", "swipe-nolm": "Glyph"]
        let want = labels[name]!
        for _ in 0..<4 {
            let globe = app.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 42, dy: app.frame.height - 42))
            globe.press(forDuration: 1.3)
            let item = app.descendants(matching: .any).matching(NSPredicate(format: "label BEGINSWITH %@", want)).firstMatch
            if item.waitForExistence(timeout: 2) { item.tap(); sleep(2); return }
            sleep(1)
        }
    }

    /// BENCH_GRID "left,width,top,rowPitch" in points (from tools/measure_layout.py --grid).
    func envGrid() -> Grid? {
        guard let s = env["BENCH_GRID"] else { return nil }
        let v = s.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard v.count == 4 else { return nil }
        return Grid(left: v[0], width: v[1], top: v[2], rowPitch: v[3])
    }

    /// Bring up BENCH_KEYBOARD over the bench field and attach a screenshot, so
    /// a third-party keyboard's letter grid can be measured (tools/measure_layout.py).
    func testScreenshotKeyboard() throws {
        guard let keyboard = env["BENCH_KEYBOARD"] else { throw XCTSkip("set BENCH_KEYBOARD") }
        let app = XCUIApplication()
        app.launchArguments = ["--bench"]
        app.launch()
        let field = app.textFields["benchField"].exists ? app.textFields["benchField"] : app.textViews["benchField"]
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.tap(); sleep(1)
        switchKeyboard(app, to: keyboard)
        if keyboard == "swipe" { _ = swipeGrid(app) }
        for label in ["Continue", "Get started", "Got it", "OK", "Not now", "Skip"] {
            let b = app.buttons[label].firstMatch
            if b.waitForExistence(timeout: 1) { b.tap(); sleep(1) }
        }
        sleep(2)
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let att = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        att.name = "keyboard_\(keyboard)"; att.lifetime = .keepAlways
        add(att)
        log(["event": "screenshot", "keyboard": keyboard, "screen": ["w": app.frame.width, "h": app.frame.height]])
    }

    func swipeGrid(_ app: XCUIApplication) -> Grid? {
        let bar = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch
        guard bar.waitForExistence(timeout: 120) else { return nil }   // first launch compiles two Core ML models and maps the LM
        let f = bar.frame
        let w = app.frame.width
        return Grid(left: 20.0 / 3.0 - 3, width: 10 * (w - 2 * 20.0 / 3.0 + 6) / 10, top: f.maxY, rowPitch: 54)
    }

    func testDescribePrivateAPI() {
        print("PRIVATE_API" + TouchSynth.describePrivateAPI())
    }

    /// Synthetic straight swipes across the top row at several point spacings
    /// and durations; logs what the Swipe extension received for each, so the
    /// synthesizer's timing behaviour can be characterised and compensated.
    func testTimingCalibration() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--bench"]
        app.launch()
        let field = app.textFields["benchField"].exists ? app.textFields["benchField"] : app.textViews["benchField"]
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.tap(); sleep(1)
        switchKeyboard(app, to: "swipe")
        guard let grid = swipeGrid(app) else { XCTFail("Swipe keyboard not up"); return }
        let y = grid.top + 0.5 * grid.rowPitch
        let sweep = (env["CALIB_SWEEP"] ?? "600:8,600:16,600:33,600:50,300:16,1200:16,600:100")
            .split(separator: ",").map { p -> (Int, Int) in let a = p.split(separator: ":"); return (Int(a[0])!, Int(a[1])!) }
        for (durationMs, spacingMs) in sweep {
            app.buttons["benchClear"].tap(); usleep(200_000)
            let n = max(2, durationMs / spacingMs + 1)
            var pts: [NSValue] = [], times: [NSNumber] = []
            for k in 0..<n {
                let f = Double(k) / Double(n - 1)
                pts.append(NSValue(cgPoint: CGPoint(x: grid.left + CGFloat(0.05 + 0.9 * f) * grid.width, y: y)))
                times.append(NSNumber(value: Double(durationMs) / 1000.0 * f))
            }
            let t0 = Date()
            try TouchSynth.replayPoints(pts, times: times)
            let wall = Date().timeIntervalSince(t0) * 1000
            usleep(400_000)
            let got = (app.buttons["suggestion1"].value as? String) ?? "?"
            log(["event": "calib", "sent_ms": durationMs, "spacing_ms": spacingMs, "sent_n": n,
                 "wall_ms": Int(wall), "received": got, "typed": (field.value as? String) ?? ""])
        }
    }

    func testReplay() throws {
        // Only meaningful when driven by tools/replay_bench.py; skip in plain test runs.
        guard let keyboard = env["BENCH_KEYBOARD"] else { throw XCTSkip("set BENCH_KEYBOARD to run the replay benchmark") }
        let source = env["BENCH_SOURCE"] ?? "capture"
        let limit = Int(env["BENCH_LIMIT"] ?? "") ?? Int.max
        let speed = Double(env["BENCH_SPEED"] ?? "") ?? 1.2   // simulator calibration; 1.0 = no compensation

        let url = Bundle(for: GestureReplayBench.self).url(forResource: "bench_gestures", withExtension: "json")!
        let data = try JSONDecoder().decode(Data_.self, from: Data(contentsOf: url))
        var pool = Array(data.sentences.filter { source == "all" || $0.source == source }.prefix(limit))
        // BENCH_SHARD "i/n": every n-th sentence starting at i, so one source can be
        // spread over several simulators; the scorer merges shards by sentence.
        if let sh = env["BENCH_SHARD"]?.split(separator: "/"), sh.count == 2, let i = Int(sh[0]), let n = Int(sh[1]), n > 1 {
            pool = pool.enumerated().filter { $0.offset % n == i }.map { $0.element }
        }
        let sentences = pool

        let app = XCUIApplication()
        app.launchArguments = ["--bench"]
        app.launch()
        let field = app.textFields["benchField"].exists ? app.textFields["benchField"] : app.textViews["benchField"]
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.tap()
        sleep(1)
        switchKeyboard(app, to: keyboard)

        let grid: Grid
        switch keyboard {
        case "swipe", "swipe-nolm":
            setSwipeLM(app, enabled: keyboard == "swipe")
            guard let g = swipeGrid(app) else { XCTFail("Swipe keyboard not up"); return }; grid = g
        case "gboard", "swiftkey": guard let g = envGrid() else { XCTFail("\(keyboard) grid not measured: pass BENCH_GRID (tools/replay_bench.py --measure)"); return }; grid = g
        default: grid = envGrid() ?? Self.quickPathGrid
        }
        log(["event": "start", "keyboard": keyboard, "source": source, "sentences": sentences.count,
             "grid": ["left": grid.left, "width": grid.width, "top": grid.top, "rowPitch": grid.rowPitch]])

        // Dismiss any first-use sheet on the system keyboard.
        let cont = app.buttons["Continue"].firstMatch
        if cont.waitForExistence(timeout: 1) { cont.tap(); sleep(1) }

        var done = 0
        var intended = 0.0, actual = 0.0   // replay time dilation, summed over gestures
        var receivedLog: [[String: Any]] = []   // what the Swipe extension measured per gesture
        for s in sentences {
            // clear the field
            let clear = app.buttons["benchClear"]
            clear.tap()
            usleep(300_000)
            let t0 = Date()
            for g in s.gestures {
                // Resample the recorded path onto a uniform 30 Hz grid (linear
                // interpolation): the synthesizer bursts events spaced closer than
                // ~33 ms and stretches time by ~1.2x at 33 ms (calibrated in the
                // simulator, testTimingCalibration); BENCH_SPEED pre-compensates,
                // and iOS interpolates the delivered path to ~60 Hz anyway.
                let tStart = g.t.first ?? 0
                let dur = ((g.t.last ?? 0) - tStart) / 1000.0 / speed
                let hz = 30.0
                let nFrames = max(2, Int((dur * hz).rounded()) + 1)
                var pts: [NSValue] = [], times: [NSNumber] = []
                var j = 0
                for k in 0..<nFrames {
                    let tk = Double(k) / hz                          // seconds since touch-down
                    let tRec = tStart + tk * speed * 1000.0          // recorded ms
                    while j + 1 < g.t.count - 1 && g.t[j + 1] < tRec { j += 1 }
                    let a = g.t[j], b = g.t[min(j + 1, g.t.count - 1)]
                    let w = b > a ? min(max((tRec - a) / (b - a), 0), 1) : 0
                    let cx = g.x[j] + (g.x[min(j + 1, g.x.count - 1)] - g.x[j]) * w
                    let cy = g.y[j] + (g.y[min(j + 1, g.y.count - 1)] - g.y[j]) * w
                    pts.append(NSValue(cgPoint: CGPoint(x: grid.left + CGFloat(cx) * grid.width,
                                                        y: grid.top + CGFloat(cy) * 3 * grid.rowPitch)))
                    times.append(NSNumber(value: tk))
                }
                let tg = Date()
                defer { actual += Date().timeIntervalSince(tg); intended += (times.last?.doubleValue ?? 0) }
                do { try TouchSynth.replayPoints(pts, times: times) } catch {
                    log(["event": "error", "message": error.localizedDescription])
                    XCTFail("replay failed: \(error.localizedDescription)")
                    return
                }
                usleep(400_000)   // let the lift be processed and the keyboard commit (LM included)
                if keyboard.hasPrefix("swipe"), let v = app.buttons["suggestion1"].value as? String, !v.isEmpty {
                    receivedLog.append(["intended_ms": Int(dur * 1000), "received": v])
                }
            }
            usleep(400_000)
            let typed = (field.value as? String) ?? ""
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            log(["kind": "bench", "session": "replay-\(s.source)-\(s.session)", "set": s.set, "keyboard": keyboard,
                 "order": 0, "ts": Int(Date().timeIntervalSince1970 * 1000), "sentence": s.words.joined(separator: " "),
                 "tag": s.tag, "typed": typed, "ms": ms, "ms_shown": ms, "deletions": 0, "input_events": s.gestures.count,
                 "ua": "replay"])
            done += 1
        }
        log(["event": "done", "keyboard": keyboard, "sentences": done, "speed": speed,
             "dilation": intended > 0 ? actual / intended : 0, "received": Array(receivedLog.prefix(12))])
    }
}
