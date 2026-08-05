// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "swift-qwen3-tts",
    platforms: [
        .macOS("15.0"),
        .iOS("18.0"),
    ],
    products: [
        // Core Qwen3 TTS library
        .library(
            name: "Qwen3TTS",
            targets: ["Qwen3TTS"]
        ),
        // Command line demo tool
        .executable(
            name: "Qwen3TTSDemo",
            targets: ["Qwen3TTSDemo"]
        ),
        // Vocal's Qwen3 background worker: loads the model once, speaks sentences
        // streamed from the Rust host over stdin (one sentence per line).
        .executable(
            name: "VocalWorker",
            targets: ["VocalWorker"]
        ),
        // Native Chatterbox (pure Swift/MLX, no Python) — port of Chatterbox-Turbo.
        .library(
            name: "Chatterbox",
            targets: ["Chatterbox"]
        ),
        // Vocal's NATIVE Chatterbox worker (same stdin protocol as VocalWorker).
        .executable(
            name: "ChatterboxWorker",
            targets: ["ChatterboxWorker"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples/", from: "2.29.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    ],
    targets: [
        // MARK: - Core Library
        .target(
            name: "Qwen3TTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Qwen3TTS",
            swiftSettings: [
                .unsafeFlags(["-Xfrontend", "-warn-concurrency"], .when(configuration: .debug))
            ]
        ),

        // MARK: - CLI Demo
        .executableTarget(
            name: "Qwen3TTSDemo",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Qwen3TTSDemo"
        ),

        // MARK: - Vocal Worker (stdin sentence server for the Rust host)
        .executableTarget(
            name: "VocalWorker",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/VocalWorker"
        ),

        // MARK: - Chatterbox (native Swift/MLX port, default voice)
        .target(
            name: "Chatterbox",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/Chatterbox"
        ),

        // MARK: - Chatterbox Worker (native stdin sentence server, no Python)
        .executableTarget(
            name: "ChatterboxWorker",
            dependencies: [
                "Chatterbox",
            ],
            path: "Sources/ChatterboxWorker"
        ),

        // MARK: - Chatterbox ML Worker (multilingual Hindi, writes wav)
        .executableTarget(
            name: "ChatterboxMLWorker",
            dependencies: [
                "Chatterbox",
            ],
            path: "Sources/ChatterboxMLWorker"
        ),

        // MARK: - Tests
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: [
                "Qwen3TTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/Qwen3TTSTests"
        ),

        .testTarget(
            name: "ChatterboxTests",
            dependencies: [
                "Chatterbox",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/ChatterboxTests"
        ),
    ]
)
