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
            InstanceSegmentationCommand.self,
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

    @Option
    var benchmarkWarmupSteps: Int = 100

    @Option
    var benchmarkProcessingSteps: Int = 1000

    func benchmark(processing: () throws -> Void) throws {
        try measure("Warmup") {
            for _ in 0..<benchmarkWarmupSteps {
                try processing()
            }
        }

        let clock = ContinuousClock()
        var durations: [Duration] = []
        try measure("Benchmark") {
            for _ in 0..<benchmarkProcessingSteps {
                let start = clock.now
                try processing()
                let duration = clock.now - start
                durations.append(duration)
            }
        }

        durations.sort()
        let min = durations.first!
        let max = durations.last!
        let median = durations[durations.count / 2]
        let average = durations.reduce(Duration.seconds(0), +) / durations.count
        let medianFPS = Duration.seconds(1) / median
        let averageFPS = Duration.seconds(1) / average
        print("Benchmark min: \(min.formatted(.units(allowed: [.minutes, .seconds, .milliseconds])))")
        print("Benchmark max: \(max.formatted(.units(allowed: [.minutes, .seconds, .milliseconds])))")
        print("Benchmark median: \(median.formatted(.units(allowed: [.minutes, .seconds, .milliseconds])))")
        print("Benchmark average: \(average.formatted(.units(allowed: [.minutes, .seconds, .milliseconds])))")
        print("Benchmark median FPS: \(medianFPS.formatted(.number.precision(.fractionLength(1))))")
        print("Benchmark average FPS: \(averageFPS.formatted(.number.precision(.fractionLength(1))))")
    }
}
