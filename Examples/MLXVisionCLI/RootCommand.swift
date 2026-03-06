//
//  RootCommand.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 02.11.2025.
//

import Foundation
import ArgumentParser
import CoreImage
import MLXVision
import MLX

@main
struct RootCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        subcommands: [
            ImageClassificationCommand.self,
            ObjectDetectionCommand.self,
            ImageSegmentationCommand.self,
            ZeroShotClassificationCommand.self,
            ZeroShotSegmentationCommand.self,
        ]
    )
}

struct CommonOptions: ParsableArguments {

    @Option(transform: { URL(filePath: $0) })
    var inputImagePath: URL

    @Option
    var inputSize: Int?

    @Option(transform: { URL(filePath: $0) })
    var outputImagePath: URL?

    @Option
    var modelID: String

    @Option
    var modelRevision: String = "main"

    @Option
    var quantizeBits: Int?

    @Flag
    var disableCompile: Bool = false

    func validate() throws {
        MLX.compile(enable: !disableCompile)
    }
}

extension CommonOptions {

    var image: CIImage {
        CIImage(contentsOf: inputImagePath)!
    }

    var modelSource: ModelSource {
        ModelSource(
            id: modelID,
            revision: modelRevision
        )
    }

    var overrides: ModelOverrides {
        ModelOverrides(
            inputSize: inputSize.map(CGFloat.init).map({ CGSize(width: $0, height: $0) }),
            quantizeBits: quantizeBits
        )
    }

    func save(_ image: CIImage) throws {
        guard let outputImagePath else {
            return
        }

        let context = CIContext()
        let format = CIFormat.RGBA8
        let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
        try context.writePNGRepresentation(
            of: image,
            to: outputImagePath,
            format: format,
            colorSpace: colorSpace
        )
    }
}

struct BenchmarkOptions: ParsableArguments {

    @Flag
    var benchmark: Bool = false

    func benchmark(
        warmupSteps: Int = 20,
        processingSteps: Int = 100,
        processing: () throws -> Void
    ) throws {
        try measure("Warmup") {
            for _ in 0..<warmupSteps {
                try processing()
            }
        }
        try measure("Benchmark") {
            for _ in 0..<processingSteps {
                try processing()
            }
        }
    }
}
