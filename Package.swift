// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Kittens",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "KittenTTS", type: .static, targets: ["KittenTTS"]),
    ],
    dependencies: [
        .package(path: "Vendor/mlx-swift"),
    ],
    targets: [
        // C++ phonemizer engine (reads espeak-ng rule files)
        .target(
            name: "CEPhonemizer",
            path: "Sources/CEPhonemizer",
            sources: ["phonemizer.cpp", "swift_bridge.cpp"],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("."),
                .unsafeFlags(["-std=c++17"]),
            ]
        ),
        // TTS library — self-contained, no network dependencies
        .target(
            name: "KittenTTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                "CEPhonemizer",
            ],
            path: "Sources/KittenTTS",
            resources: [
                .copy("Resources/nano"),
                .copy("Resources/mlx.metallib"),
            ]
        ),
        // CLI test harness
        .executableTarget(
            name: "KittenCLI",
            dependencies: ["KittenTTS"],
            path: "Tests/KittenCLI"
        ),
    ],
    swiftLanguageModes: [.v5]
)
