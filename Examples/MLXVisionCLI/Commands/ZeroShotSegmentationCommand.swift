//
//  ZeroShotSegmentationCommand.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 14.12.2025.
//

import ArgumentParser
import MLXVision
import CoreImage
import MLX

struct ZeroShotSegmentationCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "zero-shot-segmentation",
        abstract: "Run zero-shot segmentation with a text prompt"
    )

    @OptionGroup
    var commonOptions: CommonOptions

    @Option
    var prompt: String

    @Option
    var scoreThreshold: Float = 0.5

    @Option
    var maskThreshold: Float = 0.5

    func run() async throws {
        let image = commonOptions.image
        let model = try await ModelFactory.shared.load(commonOptions.modelSource, for: ZeroShotSegmentationTask.self, overrides: commonOptions.overrides)
        let request = ZeroShotSegmentationRequest(image: image, prompt: prompt, scoreThreshold: scoreThreshold, maskThreshold: maskThreshold)
        let result = try measure("Model processing") { try model.process(request) }
        let annotatedImage = MaskAnnotator().annotate(image: image, detections: result)
        try commonOptions.save(annotatedImage)
    }
}
