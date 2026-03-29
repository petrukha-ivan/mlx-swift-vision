// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-swift-vision",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [.library(name: "MLXVision", targets: ["MLXVision"])],
    dependencies: [
        .package(url: "https://github.com/reers/ReerCodable", .upToNextMinor(from: "1.4.0")),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.2")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", .upToNextMinor(from: "0.9.0")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "MLXVision",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "ReerCodable", package: "ReerCodable"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ]
        ),
        .executableTarget(
            name: "MLXVisionCLI",
            dependencies: ["MLXVision", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Examples/MLXVisionCLI"
        ),
        .testTarget(
            name: "MLXVisionTests",
            dependencies: ["MLXVision"],
            resources: [
                .process("Processing/Resources")
            ]
        ),
    ]
)
