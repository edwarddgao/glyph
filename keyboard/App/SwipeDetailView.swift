import SwiftUI
import GlyphCore

/// One accepted swipe against the word's ideal path: the keyboard, the
/// straight-through-the-key-centres route in grey, the finger's trace in the
/// keyboard's trail blue, and the precision the trace cost gives it.
struct SwipeDetailView: View {
    let word: String
    let samples: [TouchSample]
    let cost: Double?
    let decoded: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .firstTextBaseline) {
                    Text(word).font(.system(size: 34, weight: .bold, design: .rounded))
                    Spacer()
                    if let cost {
                        HStack(alignment: .firstTextBaseline, spacing: 3) {
                            Text(String(format: "%.0f", RaceGame.precision(cost: cost))).font(.system(size: 28, weight: .bold, design: .rounded)).monospacedDigit()
                            Text("%").font(.subheadline).foregroundStyle(.secondary)
                        }
                    }
                }
                if let decoded, RaceGame.norm(decoded) != RaceGame.norm(word) {
                    Text("Glyph read “\(decoded)”").font(.subheadline).foregroundStyle(.red)
                }
                ZStack {
                    SwipePad(demoWord: nil, pinned: false).allowsHitTesting(false)
                    Canvas { ctx, size in
                        let w = size.width
                        func pt(_ x: Double, _ y: Double) -> CGPoint {
                            CGPoint(x: NativeMetrics.gridLeft(w) + x * NativeMetrics.gridWidth(w),
                                    y: SwipePad.barHeight + y * 3 * NativeMetrics.rowPitch)
                        }
                        // ideal: key centre to key centre
                        let centres = word.compactMap { Geometry.center(of: $0) }.map { pt($0.x, $0.y) }
                        if centres.count >= 2 {
                            var ideal = Path(); ideal.move(to: centres[0])
                            for c in centres.dropFirst() { ideal.addLine(to: c) }
                            ctx.stroke(ideal, with: .color(.secondary.opacity(0.55)), style: StrokeStyle(lineWidth: 4, lineCap: .round, lineJoin: .round, dash: [1, 8]))
                        }
                        for c in centres {
                            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 5, y: c.y - 5, width: 10, height: 10)), with: .color(.secondary.opacity(0.7)))
                        }
                        // yours
                        let pts = samples.map { pt($0.x, $0.y) }
                        if pts.count >= 2 {
                            var trace = Path(); trace.move(to: pts[0])
                            for p in pts.dropFirst() { trace.addLine(to: p) }
                            ctx.stroke(trace, with: .color(Color.glyph.opacity(0.85)), style: StrokeStyle(lineWidth: 5, lineCap: .round, lineJoin: .round))
                        }
                    }
                }
                .frame(height: SwipePad.heroHeight)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                HStack(spacing: 18) {
                    Label { Text("ideal path") } icon: { Circle().fill(Color.secondary.opacity(0.7)).frame(width: 10, height: 10) }
                    Label { Text("your swipe") } icon: { Capsule().fill(Color.glyph).frame(width: 18, height: 5) }
                }
                .font(.caption).foregroundStyle(.secondary)
                Text("Precision is how close the finger stayed to the ideal path, 100 on it, 0 at the point a swipe stops counting.")
                    .font(.caption).foregroundStyle(.tertiary)
                Spacer()
            }
            .padding(20)
            .navigationTitle("Your swipe").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .tint(.glyph)
    }
}
