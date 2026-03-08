//
//  InstanceSegmentationCommand.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.11.2025.
//

import ArgumentParser
import MLXVision
import CoreImage

struct InstanceSegmentationCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "image-segmentation",
        abstract: "Run image segmentation on an image"
    )

    @OptionGroup
    var commonOptions: CommonOptions

    @OptionGroup
    var benchmarkOptions: BenchmarkOptions

    @Option
    var scoreThreshold: Float = 0.5

    func run() async throws {
        let image = commonOptions.image
        let model = try await ModelFactory.shared.load(commonOptions.modelSource, for: InstanceSegmentationTask.self, overrides: commonOptions.overrides)
        let request = InstanceSegmentationRequest(image: image, scoreThreshold: scoreThreshold)

        if benchmarkOptions.benchmark {
            try benchmarkOptions.benchmark {
                _ = try model(request)
            }
        }

        let detections = try measure("Model processing") { try model(request) }
        print("Top 5 detections: \(detections.top(5))")

        let annotator = MaskAnnotator<InstanceSegmentationResult>()
        let annotatedImage = annotator.annotate(image: image, detections: detections)
        try commonOptions.save(annotatedImage)
    }
}
