//
//  Detr+Processing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 09.11.2025.
//

import CoreImage
import MLX

final class DetrForObjectDetectionProcessor: Processor {

    private let labels: [String]
    private let numQueries: Int
    private let imagePreprocessor: ImagePreprocessor

    init(modelConfig: DetrForObjectDetectionConfig, imagePreprocessor: ImagePreprocessor) {
        self.labels = modelConfig.id2label.flattened
        self.numQueries = modelConfig.queriesCount
        self.imagePreprocessor = imagePreprocessor
    }

    func preprocess(_ request: ObjectDetectionRequest) throws -> ImageInput {
        try imagePreprocessor.preprocess(image: request.image)
    }

    func postprocess(_ output: DetrModelForObjectDetection.Output, _ request: ObjectDetectionRequest) throws -> [ObjectDetectionResult] {
        let probs = output.logits.softmax(axis: -1)[.ellipsis, 0..<labels.count]
        var scores = probs.max(axis: -1).asArray(Float.self)
        let keep = scores.indices(where: { $0 > request.scoreThreshold })
        guard !keep.isEmpty else {
            return []
        }

        let (cx, cy, w, h) = output.boxes.split(axis: -1)
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

final class DetrForImageSegmentationProcessor: Processor {

    private let labels: [String]
    private let numQueries: Int
    private let imagePreprocessor: ImagePreprocessor

    init(modelConfig: DetrForObjectDetectionConfig, imagePreprocessor: ImagePreprocessor) {
        self.labels = modelConfig.id2label.flattened
        self.numQueries = modelConfig.queriesCount
        self.imagePreprocessor = imagePreprocessor
    }

    func preprocess(_ request: ImageSegmentationRequest) throws -> ImageInput {
        try imagePreprocessor.preprocess(image: request.image)
    }

    func postprocess(_ output: DetrModelForImageSegmentation.Output, _ request: ImageSegmentationRequest) throws -> [ImageSegmentationResult] {
        let probs = output.logits.softmax(axis: -1)[.ellipsis, 0..<labels.count]
        var scores = probs.max(axis: -1).asArray(Float.self)
        let keep = scores.indices(where: { $0 > request.scoreThreshold })
        guard !keep.isEmpty else {
            return []
        }

        let masks = output.segmentationMask.split(parts: numQueries)[keep]
        let labels = probs.argmax(axis: -1).asArray(Int.self)[keep].map({ self.labels[$0] })
        scores = scores[keep]

        let targetSize = Array(output.pixelValues.shape.dropFirst().dropLast())
        let interpolatedMasks = zip(scores, masks).map { score, mask in
            (mask * score)
                .expandedDimensions(axis: -1)
                .interpolate(size: targetSize, mode: .linear(alignCorners: false))
                .squeezed()
        }

        let finalMask = MLX.stacked(interpolatedMasks, axis: 0)
            .flattened(start: 1, end: 2)
            .transposed(axes: [1, 0])
            .softmax(axis: -1)
            .argMax(axis: -1)
            .reshaped(targetSize)

        MLX.eval(finalMask)
        return (0..<keep.count).map { i in
            ImageSegmentationResult(
                mask: MLX.equal(finalMask, i),
                label: labels[i],
                score: scores[i]
            )
        }.filter {
            !$0.label.starts(with: "LABEL_")
        }
    }
}
