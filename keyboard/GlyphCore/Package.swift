// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GlyphCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "GlyphCore", targets: ["GlyphCore"]),
    ],
    targets: [
        .target(name: "GlyphCore"),
        .testTarget(name: "GlyphCoreTests", dependencies: ["GlyphCore"]),
    ]
)
