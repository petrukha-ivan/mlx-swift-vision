//
//  ZeroShotClassificationCommand.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 14.12.2025.
//

import ArgumentParser
import MLXVision
import CoreImage

struct ZeroShotClassificationCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "zero-shot-classification",
        abstract: "Run zero-shot classification with custom labels"
    )

    @OptionGroup
    var commonOptions: CommonOptions

    @Option(parsing: .upToNextOption)
    var labels: [String] = []

    func run() async throws {
        let image = commonOptions.image
        let model = try await ModelFactory.shared.load(commonOptions.modelSource, for: ZeroShotClassificationTask.self, overrides: commonOptions.overrides)
        let request = ZeroShotClassificationRequest(image: image, labels: labels)
        let result = try measure("Model processing") { try model.process(request) }
        print(result)
    }
}
