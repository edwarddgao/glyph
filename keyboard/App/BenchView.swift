import SwiftUI
import UIKit

/// A bare text field for the gesture-replay benchmark (`--bench`): the UI test
/// focuses it, replays recorded swipes on whichever keyboard is active, and
/// reads the committed text back through accessibility.
struct BenchView: View {
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("gesture replay bench").font(.headline)
            TextField("", text: $text, axis: .vertical)
                .lineLimit(3...6)
                .textInputAutocapitalization(.sentences)
                .autocorrectionDisabled(false)
                .focused($focused)
                .padding(8)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .accessibilityIdentifier("benchField")
            HStack {
                Button("clear") { text = ""; focused = true }
                    .accessibilityIdentifier("benchClear")
                Spacer()
                Text(text).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    .accessibilityIdentifier("benchEcho")
            }
            Spacer()
        }
        .padding()
        .onAppear { focused = true }
    }

    static var isRequested: Bool { CommandLine.arguments.contains("--bench") }
}
