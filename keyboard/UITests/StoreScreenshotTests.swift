import XCTest

/// App Store screenshots, on an iPhone 17 Pro Max simulator (6.9", 1320×2868):
///   SWIPE_SHOTS=/path xcodegen generate && xcodebuild test ... -only-testing:GlyphUITests/StoreScreenshotTests
/// 01 welcome · 02 practice mid-sentence · 03 round summary · 04 the keyboard in a text field · 05 the benchmark table.
final class StoreScreenshotTests: GlyphUITests {
    func store(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        if let dir = ProcessInfo.processInfo.environment["SWIPE_SHOTS"] { try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name + ".png")) }
    }

    /// Straight legs through the key centres, 40 ms per sample, 12 samples per leg — the race test's gesture.
    func replay(_ word: String, center: (Character) -> CGPoint) throws {
        let keys = Array(word).map(center)
        if keys.count == 1 { try TouchSynth.replayPoints([NSValue(cgPoint: keys[0]), NSValue(cgPoint: keys[0])], times: [0, 0.08]); return }
        var pts: [NSValue] = [], times: [NSNumber] = []; var t = 0.0
        for (a, b) in zip(keys, keys.dropFirst()) {
            for k in 0..<12 {
                let u = Double(k) / 12
                pts.append(NSValue(cgPoint: CGPoint(x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u))); times.append(NSNumber(value: t)); t += 0.04
            }
        }
        pts.append(NSValue(cgPoint: keys.last!)); times.append(NSNumber(value: t))
        try TouchSynth.replayPoints(pts, times: times)
    }

    static let rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"], inset = [0.0, 0.5, 1.5]
    static func grid(left: Double, top: Double, pitch: Double, rowPitch: Double = 54) -> (Character) -> CGPoint {
        return { ch in
            for (r, row) in rows.enumerated() {
                if let c = row.firstIndex(of: ch) {
                    let col = inset[r] + Double(row.distance(from: row.startIndex, to: c))
                    return CGPoint(x: left + (col + 0.5) * pitch, y: top + (Double(r) + 0.5) * rowPitch)
                }
            }
            return CGPoint(x: left, y: top)
        }
    }

    func testStoreScreens() throws {
        // 01 welcome, 05 details
        let w = XCUIApplication(); w.launchArguments = ["--onboarding"]; w.launch()
        XCTAssertTrue(w.buttons["onboardingStart"].waitForExistence(timeout: 10)); sleep(2); store("01_welcome")
        if w.buttons["onboardingDetails"].exists { w.buttons["onboardingDetails"].tap(); sleep(2); store("05_details") }
        w.terminate()

        // 02 practice mid-sentence, 03 sentence card
        let r = XCUIApplication(); r.launchArguments = ["--race", "--race-set", "1", "--no-upload"]; r.launch()
        let pad = r.otherElements["racePad"], sentence = r.otherElements["raceSentence"]
        XCTAssertTrue(pad.waitForExistence(timeout: 120) && sentence.waitForExistence(timeout: 120))
        sleep(4)   // let the language model finish loading so the verdicts use the shipped stack
        let f = pad.frame, margin = 20.0 / 3.0, gap = 6.0
        let center = Self.grid(left: f.minX + margin - gap / 2, top: f.minY + 42.47, pitch: (f.width - 2 * margin + gap) / 10)
        // Five sentences, no card in between: each finished line gives way to the next after a beat.
        for s in 0..<5 {
            _ = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "value BEGINSWITH '0/'"), object: sentence)], timeout: 10)
            let words = sentence.label.split(separator: " ").map(String.init)
            for (i, word) in words.enumerated() {
                try replay(word.lowercased(), center: center)
                let ok = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "value BEGINSWITH '\(i + 1)/'"), object: sentence)], timeout: 6) == .completed
                if !ok { try replay(word.lowercased(), center: center); sleep(1) }
                if s == 0 && i == 2 { sleep(1); store("02_practice") }
            }
            if s < 4 { sleep(2) }
        }
        if r.buttons["raceAgain"].waitForExistence(timeout: 15) { sleep(1); store("03_sentence") }
        r.terminate()
    }

    /// 04 the keyboard in a text field: Messages if the simulator has it, else the home screen's try-it field.
    func testKeyboardShot() throws {
        if try messagesShot() { return }
        var app = XCUIApplication(); app.launchArguments = ["--onboarded", "--no-upload"]; app.launch()
        if !app.textViews["homeTry"].waitForExistence(timeout: 5) {   // keyboard not enabled yet: the home screen shows the enable button
            enableInSettings(); app.terminate()
            app = XCUIApplication(); app.launchArguments = ["--onboarded", "--no-upload"]; app.launch()
        }
        let field = app.textViews["homeTry"].exists ? app.textViews["homeTry"] : app.textFields["homeTry"]
        XCTAssertTrue(field.waitForExistence(timeout: 5), "home try-it field"); field.tap()
        XCTAssertTrue(switchToSwipe(app), "Glyph keyboard did not come up")
        sleep(3)  // decoder load
        let bar = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.frame
        let width = app.frame.width
        let kb = Self.grid(left: 20.0 / 3.0 - 3, top: bar.maxY, pitch: (width - 2 * 20.0 / 3.0 + 6) / 10)
        for word in ["typing", "should", "feel", "like", "this"] { try replay(word, center: kb); sleep(1) }
        sleep(1); store("04_keyboard")
    }

    /// Glyph inside Messages' compose field. Returns false if Messages is not there or the field never appears.
    func messagesShot() throws -> Bool {
        let host = XCUIApplication(); host.launchArguments = ["--bench"]; host.launch()
        let field0 = host.textViews.firstMatch.exists ? host.textViews.firstMatch : host.textFields.firstMatch
        _ = field0.waitForExistence(timeout: 5); field0.tap()
        if !host.keyboards.buttons["Next keyboard"].waitForExistence(timeout: 3) { enableInSettings(); host.activate() }
        host.terminate()
        let msgs = XCUIApplication(bundleIdentifier: "com.apple.MobileSMS"); msgs.launch()
        guard msgs.wait(for: .runningForeground, timeout: 10) else { return false }
        sleep(2)
        let compose = msgs.buttons["New Message"].exists ? msgs.buttons["New Message"] : msgs.buttons.matching(NSPredicate(format: "label CONTAINS[c] 'compose' OR label CONTAINS[c] 'new message'")).firstMatch
        if compose.waitForExistence(timeout: 5) { compose.tap() }
        let body = msgs.textViews.matching(NSPredicate(format: "label CONTAINS[c] 'message' OR placeholderValue CONTAINS[c] 'message'")).firstMatch
        guard body.waitForExistence(timeout: 8) else { msgs.terminate(); return false }
        body.tap(); sleep(1)
        guard switchToSwipe(msgs) else { msgs.terminate(); return false }
        sleep(3)
        let bar = msgs.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.frame
        guard bar.height > 0 else { msgs.terminate(); return false }
        let kb = Self.grid(left: 20.0 / 3.0 - 3, top: bar.maxY, pitch: (msgs.frame.width - 2 * 20.0 / 3.0 + 6) / 10)
        for word in ["running", "late", "order", "without", "me"] { try replay(word, center: kb); sleep(1) }
        sleep(1); store("04_keyboard"); msgs.terminate()
        return true
    }
}
