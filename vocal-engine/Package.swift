// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
import PackageDescription

let package = Package(
    name: "vocal-engine",
    platforms: [
        .macOS(.v14) // MLX requires modern macOS
    ],
    dependencies: [
        // Apple's official MLX framework for Apple Silicon
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.12.0"),
        // Official Hugging Face Hub downloader for Swift
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.0")
    ],
    targets: [
        .executableTarget(
            name: "vocal-engine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers")
            ]
        )
    ]
)