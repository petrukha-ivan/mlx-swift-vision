//
//  ImageClassificationCommand.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 06.11.2025.
//

import ArgumentParser
import MLXVision

struct ImageClassificationCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "image-classification",
        abstract: "Run image classification on an image"
    )

    @OptionGroup
    var commonOptions: CommonOptions

    @Option
    var topK: Int = 5

    func run() async throws {
        let image = commonOptions.image
        let model = try await ModelFactory.shared.load(commonOptions.modelSource, for: ImageClassificationTask.self, overrides: commonOptions.overrides)
        let request = ImageClassificationRequest(image: image)
        let result = try measure("Model processing") { try model.process(request).top(topK) }
        print(result)
    }
}
