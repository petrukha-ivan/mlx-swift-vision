//
//  ImageSegmentationCommand.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.11.2025.
//

import ArgumentParser
import MLXVision
import CoreImage

struct ImageSegmentationCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "image-segmentation",
        abstract: "Run image segmentation on an image"
    )

    @OptionGroup
    var commonOptions: CommonOptions

    @Option
    var scoreThreshold: Float = 0.5

    func run() async throws {
        let image = commonOptions.image
        let model = try await ModelFactory.shared.load(commonOptions.modelSource, for: ImageSegmentationTask.self, overrides: commonOptions.overrides)
        let request = ImageSegmentationRequest(image: image, scoreThreshold: scoreThreshold)
        let segments = try measure("Model processing") { try model.process(request) }
        let annotatedImage = MaskAnnotator().annotate(image: image, detections: segments)
        try commonOptions.save(annotatedImage)
        print("Segments: \(segments.count)")
    }
}
