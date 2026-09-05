import XCTest

/// Drives the real extension in the simulator: focus the host app's text
/// field, switch to the Swipe keyboard, tap a letter, drag across keys, pick
/// a suggestion, and check what landed in the text field.
///
/// Requires the keyboard to be enabled in the simulator first:
///   xcrun simctl spawn <udid> defaults write com.apple.Preferences AppleKeyboards \
///     -array com.edwardgao.glyph.keyboard "en_US@sw=QWERTY;hw=Automatic"
final class GlyphUITests: XCTestCase {
    let shots = ProcessInfo.processInfo.environment["SWIPE_SHOTS"]

    func snap(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let att = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        att.name = name; att.lifetime = .keepAlways
        add(att)
        if let dir = shots { try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name + ".png")) }
    }

    /// Switch keyboards with the globe key until our status/suggestion bar shows.
    func isSwipeUp(_ app: XCUIApplication) -> Bool {
        app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.exists || app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'loading'")).firstMatch.exists
            || app.keyboards.buttons["⌫"].exists
    }

    func switchToSwipe(_ app: XCUIApplication) -> Bool {
        let screen = app.frame
        for attempt in 0..<4 {
            if isSwipeUp(app) { break }
            // The system keyboard's globe key sits at the bottom-left corner of
            // the screen; it is not exposed to accessibility queries here.
            let globe = app.coordinate(withNormalizedOffset: .zero)
                .withOffset(CGVector(dx: 42, dy: screen.height - 42))
            globe.press(forDuration: 1.3)
            snap("00d_globe_menu_\(attempt)")
            let item = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Glyph'")).firstMatch
            if item.waitForExistence(timeout: 2) { item.tap() } else { globe.tap() }
            sleep(2)
        }
        for _ in 0..<10 {
            if app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.exists { return true }
            sleep(1)
        }
        if let dir = shots { try? app.debugDescription.write(toFile: dir + "/app_tree.txt", atomically: true, encoding: .utf8) }
        return app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.exists
    }

    /// Adds the keyboard through the Settings app, as a user would.
    /// Settings › General › Keyboard › Keyboards › Add New Keyboard › <name>;
    /// ENABLE_KEYBOARD=Gboard runs it for a third-party keyboard.
    func testEnableKeyboardInSettings() {
        enableInSettings(ProcessInfo.processInfo.environment["ENABLE_KEYBOARD"] ?? "Glyph")
    }

    func enableInSettings(_ name: String = "Glyph") {
        let settings = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        settings.launch()
        func tapCell(_ name: String, timeout: TimeInterval = 8) -> Bool {
            let cell = settings.cells.staticTexts[name].firstMatch
            if cell.waitForExistence(timeout: timeout) { cell.tap(); return true }
            let any = settings.descendants(matching: .any).matching(NSPredicate(format: "label == %@ OR label BEGINSWITH %@", name, name)).firstMatch
            if any.waitForExistence(timeout: 2) { any.tap(); return true }
            if let dir = shots { try? settings.debugDescription.write(toFile: dir + "/settings_tree_\(name).txt", atomically: true, encoding: .utf8) }
            // A full-page swipe overshoots rows near the fold (the phone's root list
            // lost "General" that way); scroll a third of a screen at a time.
            let h = settings.frame.height, w = settings.frame.width
            for _ in 0..<12 {
                settings.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: w / 2, dy: h * 0.75))
                    .press(forDuration: 0.05, thenDragTo: settings.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: w / 2, dy: h * 0.45)))
                usleep(400_000)
                if cell.exists && cell.isHittable { cell.tap(); return true }
                if any.exists && any.isHittable { any.tap(); return true }
            }
            return false
        }
        XCTAssertTrue(tapCell("General"), "Settings › General")
        XCTAssertTrue(tapCell("Keyboard"), "General › Keyboard")
        XCTAssertTrue(tapCell("Keyboards"), "Keyboard › Keyboards")
        snap("00a_keyboards_list")
        if settings.staticTexts[name].firstMatch.exists { settings.terminate(); return }
        XCTAssertTrue(tapCell("Add New Keyboard") || tapCell("Add New Keyboard...", timeout: 2), "Add New Keyboard")
        snap("00b_add_keyboard")
        XCTAssertTrue(tapCell(name), "\(name) in third-party keyboards")
        sleep(1)
        snap("00c_after_add")
        settings.terminate()
    }

    func testTapSwipeAndSuggestion() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--bench"]
        app.launch()
        do {
            let field0 = app.textViews.firstMatch.exists ? app.textViews.firstMatch : app.textFields.firstMatch
            XCTAssertTrue(field0.waitForExistence(timeout: 5))
            field0.tap()
            if !app.keyboards.buttons["Next keyboard"].waitForExistence(timeout: 3) {
                enableInSettings()
                app.activate()
            }
        }
        let field = app.textViews.firstMatch.exists ? app.textViews.firstMatch : app.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 5), "text field")
        field.tap()
        XCTAssertTrue(switchToSwipe(app), "Swipe keyboard did not come up (is it enabled in AppleKeyboards?)")
        snap("01_keyboard_up")

        // Our suggestion/status bar is 44pt tall and sits at the top of the
        // keyboard; the grid starts 4pt below it with three 50pt rows.
        // Native metrics (see Extension/NativeMetrics.swift): bar 42.8, row pitch 54,
        // column pitch (W - 2*6.67 + 6) / 10, grid left edge at 6.67 - 3.
        let bar = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.frame
        let width = app.frame.width
        let gridTop = bar.maxY
        let pitch = (width - 2 * 20.0 / 3.0 + 6) / 10, gridLeft = 20.0 / 3.0 - 3
        func at(_ x: Double, _ y: Double) -> XCUICoordinate {
            app.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: x, dy: y))
        }
        func key(_ ch: Character) -> XCUICoordinate {
            let rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"], inset = [0.0, 0.5, 1.5]
            let r = rows.firstIndex { $0.contains(ch) }!
            let c = Array(rows[r]).firstIndex(of: ch)!
            return at(gridLeft + (inset[r] + Double(c) + 0.5) * pitch, gridTop + (Double(r) + 0.5) * 54)
        }
        let deleteKey = at(gridLeft + (10 - 0.65) * pitch, gridTop + 2.5 * 54)
        func text() -> String { (field.value as? String) ?? "" }

        // 1. tap a letter (auto-capitalized at the start of the field)
        key("i").tap()
        sleep(1)
        snap("02_after_tap")
        XCTAssertEqual(text(), "I", "after tapping i")

        // 2. space, then a straight-line swipe t -> o on the top row
        app.buttons["space"].firstMatch.tap()
        sleep(1)
        XCTAssertEqual(text(), "I ", "after space")
        key("t").press(forDuration: 0.05, thenDragTo: key("o"), withVelocity: .slow, thenHoldForDuration: 0.05)
        sleep(1)
        snap("03_after_swipe")
        let afterSwipe = text()
        XCTAssertTrue(afterSwipe.hasPrefix("I ") && afterSwipe.count > 3 && afterSwipe.hasSuffix(" "),
                      "swipe should append a word and a space: '\(afterSwipe)'")
        let swiped = afterSwipe.dropFirst(2).trimmingCharacters(in: .whitespaces)
        XCTAssertFalse(swiped.isEmpty)

        // 3. the suggestion bar shows alternatives; picking one replaces the word
        let first = app.buttons.matching(identifier: "suggestion1").firstMatch
        let secondBtn = app.buttons.matching(identifier: "suggestion0").firstMatch
        XCTAssertTrue(first.waitForExistence(timeout: 3), "suggestion bar")
        let slots = (0..<3).map { app.buttons.matching(identifier: "suggestion\($0)").firstMatch }
        let slotDesc = slots.map { "\($0.label)@x=\(Int($0.frame.minX))" }.joined(separator: " | ")
        XCTAssertEqual(first.label, swiped, "middle pill is the inserted word; slots: \(slotDesc)")
        if secondBtn.exists, !secondBtn.label.isEmpty {
            let second = secondBtn.label
            secondBtn.tap()
            sleep(1)
            snap("04_after_pick")
            XCTAssertEqual(text(), "I " + second + " ", "after picking \(second)")
        }

        // 4. a curved swipe: h -> e -> l -> o ("hello"): drag h->e then continue is not
        //    possible with a single XCUITest drag, so use another straight word: w -> e ("we")
        key("w").press(forDuration: 0.05, thenDragTo: key("e"), withVelocity: .slow, thenHoldForDuration: 0.05)
        sleep(1)
        snap("05_after_we")
        let weText = text().lowercased()
        let weWord = weText.split(separator: " ").last.map(String.init) ?? ""
        let weSlots = (0..<3).map { app.buttons.matching(identifier: "suggestion\($0)").firstMatch.label.lowercased() }
        XCTAssertTrue(weSlots.contains("we"), "w->e should surface 'we' in the bar: \(weSlots)")
        XCTAssertEqual(weSlots[1], weWord, "middle pill is the inserted word: '\(weText)'")

        // 5. backspace removes one character
        let before = text().count
        deleteKey.tap()
        sleep(1)
        XCTAssertEqual(text().count, before - 1, "backspace")
        snap("06_after_backspace")
    }
}


final class EmojiPanelTests: XCTestCase {
    let shots = ProcessInfo.processInfo.environment["SWIPE_SHOTS"]
    func snap(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        if let dir = shots { try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name + ".png")) }
    }

    func testEmojiPanel() {
        let app = XCUIApplication()
        app.launchArguments = ["--bench"]
        app.launch()
        let field = app.textViews.firstMatch.exists ? app.textViews.firstMatch : app.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.tap()
        // make sure Swipe is up (it was left selected by the other tests)
        if !app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.waitForExistence(timeout: 5) && !app.buttons["emoji"].exists {
            let globe = app.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 42, dy: app.frame.height - 42))
            globe.press(forDuration: 1.3)
            let item = app.descendants(matching: .any).matching(NSPredicate(format: "label == 'Glyph'")).firstMatch
            if item.waitForExistence(timeout: 2) { item.tap() }
            _ = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.waitForExistence(timeout: 8)
        }
        let bar = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'swipe a word'")).firstMatch.frame
        let kbTop = bar.minY
        let emojiKey = app.buttons["emoji"]
        XCTAssertTrue(emojiKey.waitForExistence(timeout: 3), "emoji key")
        emojiKey.tap()
        sleep(1)
        snap("emoji_panel")
        // first frequently-used emoji: column 0 row 0
        let first = app.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 26.3, dy: kbTop + 41.3))
        first.tap()
        sleep(1)
        let text = field.value as? String ?? ""
        XCTAssertTrue(text.unicodeScalars.contains { $0.properties.isEmojiPresentation }, "emoji inserted: '\(text)'")
        // category tap: flags (last icon) then animals (third)
        app.buttons["Flags"].tap(); sleep(1); snap("emoji_flags")
        app.buttons["Animals & Nature"].tap(); sleep(1); snap("emoji_animals")
        app.buttons["emoji.delete"].tap(); sleep(1)
        let after = field.value as? String ?? ""
        XCTAssertFalse(after.unicodeScalars.contains { $0.properties.isEmojiPresentation }, "delete removed the emoji: '\(after)'")
        app.buttons["emoji.abc"].tap(); sleep(1)
        XCTAssertTrue(app.buttons["emoji"].exists, "back to letters")
        snap("emoji_back")
    }
}

/// SwipeRacer in the simulator: start a fixed race, swipe the first word through
/// its key centres on the embedded pad, and check the game accepted it.
final class RaceUITests: XCTestCase {
    func testRaceAcceptsSwipedWord() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--race", "--race-set", "1"]
        app.launch()
        let start = app.buttons["raceStart"]
        XCTAssertTrue(start.waitForExistence(timeout: 10))
        // the decoder loads in the background; the button title flips to "Race" when ready
        let ready = NSPredicate(format: "label BEGINSWITH 'Start'")
        XCTAssertTrue(start.waitForExistence(timeout: 5))
        _ = XCTNSPredicateExpectation(predicate: ready, object: start)
        XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: ready, object: start)], timeout: 120)
        start.tap()
        let pad = app.otherElements["racePad"]
        XCTAssertTrue(pad.waitForExistence(timeout: 10))
        let sentence = app.otherElements["raceSentence"]
        XCTAssertTrue(sentence.waitForExistence(timeout: 10))
        let first = String(sentence.label.split(separator: " ").first ?? "")
        XCTAssertTrue(first.count >= 2 && first.allSatisfy { $0.isLetter }, "first word: \(first)")
        // pad frame -> grid geometry (same numbers as the keyboard: NativeMetrics)
        let f = pad.frame
        let margin = 20.0 / 3.0, gap = 6.0
        let pitch = (f.width - 2 * margin + gap) / 10
        let left = margin - gap / 2, gridTop = f.minY + 42.47, rowPitch = 54.0   // SwipePad.barHeight
        let rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"], inset = [0.0, 0.5, 1.5]
        func center(_ ch: Character) -> CGPoint {
            for (r, row) in rows.enumerated() {
                if let c = row.firstIndex(of: ch) {
                    let col = inset[r] + Double(row.distance(from: row.startIndex, to: c))
                    return CGPoint(x: f.minX + left + (col + 0.5) * pitch, y: gridTop + (Double(r) + 0.5) * rowPitch)
                }
            }
            fatalError()
        }
        // the first word through its key centres, 40 ms per sample, ~12 samples per leg
        var pts: [NSValue] = [], times: [NSNumber] = []
        let keys = Array(first).map(center)
        var t = 0.0
        for (a, b) in zip(keys, keys.dropFirst()) {
            for k in 0..<12 {
                let u = Double(k) / 12
                pts.append(NSValue(cgPoint: CGPoint(x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u)))
                times.append(NSNumber(value: t)); t += 0.04
            }
        }
        pts.append(NSValue(cgPoint: keys.last!)); times.append(NSNumber(value: t))
        try TouchSynth.replayPoints(pts, times: times)
        let advanced = NSPredicate(format: "value BEGINSWITH '1/'")
        let ok = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: advanced, object: sentence)], timeout: 10) == .completed
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let att = XCTAttachment(data: png, uniformTypeIdentifier: "public.png"); att.name = "race"; att.lifetime = .keepAlways; add(att)
        if let dir = ProcessInfo.processInfo.environment["SWIPE_SHOTS"] { try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("race.png")) }
        XCTAssertTrue(ok, "first word not accepted; sentence value = \(sentence.value ?? "nil")")
    }
}

extension RaceUITests {
    /// A normal race (no --race-set) draws from the bundled prompt pool.
    func testRaceDrawsFromPool() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--race"]
        app.launch()
        let start = app.buttons["raceStart"]
        XCTAssertTrue(start.waitForExistence(timeout: 10))
        XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: NSPredicate(format: "label BEGINSWITH 'Start'"), object: start)], timeout: 120)
        start.tap()
        let sentence = app.otherElements["raceSentence"]
        XCTAssertTrue(sentence.waitForExistence(timeout: 10))
        let words = sentence.label.split(separator: " ")
        XCTAssertTrue((4...9).contains(words.count), "pool sentences are 4–9 words: \(sentence.label)")
        XCTAssertFalse(sentence.label.isEmpty)
    }
}


/// First launch: welcome → three-sentence race → enable steps.
final class OnboardingUITests: XCTestCase {
    func testOnboardingReachesRace() throws {
        let app = XCUIApplication()
        app.launchArguments = ["--onboarding"]
        app.launch()
        let start = app.buttons["onboardingStart"]
        XCTAssertTrue(start.waitForExistence(timeout: 10))
        start.tap()
        let pad = app.otherElements["racePad"]
        XCTAssertTrue(pad.waitForExistence(timeout: 120), "race pad after Start")
        let sentence = app.otherElements["raceSentence"]
        XCTAssertTrue(sentence.waitForExistence(timeout: 120))
        XCTAssertTrue(app.staticTexts["sentence 1 / 3"].waitForExistence(timeout: 5), "onboarding races three sentences")
        XCTAssertFalse(app.buttons["quit"].exists, "no skip in onboarding")
        let png = XCUIScreen.main.screenshot().pngRepresentation
        if let dir = ProcessInfo.processInfo.environment["SWIPE_SHOTS"] { try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("onboarding_race.png")) }
    }
}


/// Screenshots of every screen, for design review (SWIPE_SHOTS).
final class ScreenshotTests: XCTestCase {
    func shot(_ app: XCUIApplication, _ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        if let dir = ProcessInfo.processInfo.environment["SWIPE_SHOTS"] { try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name + ".png")) }
    }
    func testAllScreens() {
        for (args, name, wait) in [(["--onboarding"], "01_welcome", "onboardingStart"), (["--onboarding", "--onboarding-step", "2"], "03_enable", "onboardingDone")] {
            let app = XCUIApplication(); app.launchArguments = args; app.launch()
            _ = app.buttons[wait].waitForExistence(timeout: 10); sleep(1); shot(app, name); app.terminate()
        }
        let app = XCUIApplication(); app.launch()
        _ = app.buttons["onboardingStart"].waitForExistence(timeout: 5)
        if app.buttons["onboardingStart"].exists {   // not onboarded yet on this simulator: mark done via the race-free path
            app.terminate(); let a2 = XCUIApplication(); a2.launchArguments = ["--onboarding", "--onboarding-step", "2"]; a2.launch()
            _ = a2.buttons["onboardingDone"].waitForExistence(timeout: 10); a2.buttons["onboardingDone"].tap(); sleep(1); shot(a2, "04_home"); a2.terminate()
        } else { sleep(1); shot(app, "04_home") }
        let r = XCUIApplication(); r.launchArguments = ["--race"]; r.launch()
        _ = r.buttons["raceStart"].waitForExistence(timeout: 10); sleep(1); shot(r, "05_race_intro")
    }
}
