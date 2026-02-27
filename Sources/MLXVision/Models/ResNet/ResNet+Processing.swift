//
//  ResNet+Processing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 09.11.2025.
//

import MLX

final class ResNetProcessor: Processor {

    private let labels: [String]
    private let imagePreprocessor: ImagePreprocessor

    init(modelConfig: ResNetForImageClassificationConfig, imagePreprocessor: ImagePreprocessor) {
        self.labels = modelConfig.id2label.flattened
        self.imagePreprocessor = imagePreprocessor
    }

    func preprocess(_ request: ImageClassificationRequest) throws -> ImageInput {
        try imagePreprocessor.preprocess(image: request.image)
    }

    func postprocess(_ logits: MLXArray, _ request: ImageClassificationRequest) throws -> [ClassificationResult] {
        let scores = logits.softmax().asArray(Float.self)
        return zip(labels, scores).map { label, score in
            ClassificationResult(
                label: label,
                score: score
            )
        }
    }
}
