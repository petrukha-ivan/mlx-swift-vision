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

func measure<T>(_ label: String, terminator: String = "\n", operation: () async throws -> T) async rethrows -> T {
    let clock = ContinuousClock()
    let start = clock.now
    let value = try await operation()
    let duration = clock.now - start
    print("\(label) finished (\(duration.formatted(.units(allowed: [.minutes, .seconds, .milliseconds]))))", terminator: terminator)
    return value
}

func measure<T>(_ label: String, terminator: String = "\n", operation: () throws -> T) rethrows -> T {
    let clock = ContinuousClock()
    let start = clock.now
    let value = try operation()
    let duration = clock.now - start
    print("\(label) finished (\(duration.formatted(.units(allowed: [.minutes, .seconds, .milliseconds]))))", terminator: terminator)
    return value
}
