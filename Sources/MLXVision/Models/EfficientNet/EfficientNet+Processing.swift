//
//  EfficientNet+Processing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import MLX

final class EfficientNetProcessor: Processor {

    private let labels: [String]
    private let imagePreprocessor: ImagePreprocessor

    init(modelConfig: EfficientNetForImageClassificationConfig, imagePreprocessor: ImagePreprocessor) {
        self.labels = modelConfig.id2label.flattened
        self.imagePreprocessor = imagePreprocessor
    }

    func preprocess(_ request: ImageClassificationRequest) throws -> ImageInput {
        try imagePreprocessor.preprocess(image: request.image)
    }

    func postprocess(_ output: EfficientNetModelForImageClassification.Output, _ request: ImageClassificationRequest) throws -> [ClassificationResult] {
        let scores = output.probs.asArray(Float.self)
        return zip(labels, scores).map { label, score in
            ClassificationResult(
                label: label,
                score: score
            )
        }
    }
}
