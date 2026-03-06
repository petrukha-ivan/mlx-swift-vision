//
//  RfDetr+Processing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 03.03.2026.
//

import CoreImage
import MLX

final class RfDetrForObjectDetectionProcessor: Processor {

    private let labels: [String]
    private let numQueries: Int
    private let imagePreprocessor: ImagePreprocessor

    init(modelConfig: RfDetrForObjectDetectionConfig, imagePreprocessor: ImagePreprocessor) {
        self.labels = modelConfig.id2label.flattened
        self.numQueries = modelConfig.numQueries
        self.imagePreprocessor = imagePreprocessor
    }

    func preprocess(_ request: ObjectDetectionRequest) throws -> ImageInput {
        try imagePreprocessor.preprocess(image: request.image)
    }

    func postprocess(_ output: RfDetrModelForObjectDetection.Output, _ request: ObjectDetectionRequest) throws -> [ObjectDetectionResult] {
        let probs = output.probs.squeezed()[.ellipsis, 0..<labels.count]
        var scores = probs.max(axis: -1).asArray(Float.self)
        let keep = scores.indices(where: { $0 > request.scoreThreshold })
        guard !keep.isEmpty else {
            return []
        }

        let (cx, cy, w, h) = output.boxes.squeezed().split(axis: -1)
        let boxes = MLX.stacked([cx - 0.5 * w, cy - 0.5 * h, w, h], axis: -1).split(parts: numQueries)[keep]
        let labels = probs.argmax(axis: -1).asArray(Int.self)[keep].map({ self.labels[$0] })
        scores = scores[keep]

        return zip(boxes, labels, scores).map { bbox, label, score in
            ObjectDetectionResult(
                bbox: bbox.asArray(Float.self),
                label: label,
                score: score
            )
        }
    }
}
