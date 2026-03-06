//
//  ObjectDetectionCommand.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 06.11.2025.
//

import ArgumentParser
import CoreImage
import MLXVision
import MLX

struct ObjectDetectionCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "object-detection",
        abstract: "Run object detection on an image"
    )

    @OptionGroup
    var commonOptions: CommonOptions

    @OptionGroup
    var benchmarkOptions: BenchmarkOptions

    @Option
    var scoreThreshold: Float = 0.5

    func run() async throws {
        let image = commonOptions.image
        let model = try await ModelFactory.shared.load(commonOptions.modelSource, for: ObjectDetectionTask.self, overrides: commonOptions.overrides)
        let request = ObjectDetectionRequest(image: image, scoreThreshold: scoreThreshold)

        if benchmarkOptions.benchmark {
            try benchmarkOptions.benchmark {
                _ = try model.process(request)
            }
        }

        let detections = try measure("Model processing") { try model.process(request) }
        print("Top 5 detections: \(detections.top(5))")

        if commonOptions.outputImagePath != nil {
            var annotatedImage = BoxAnnotator().annotate(image: image, detections: detections)
            annotatedImage = LabelAnnotator().annotate(image: annotatedImage, detections: detections)
            try commonOptions.save(annotatedImage)
        }
    }
}
