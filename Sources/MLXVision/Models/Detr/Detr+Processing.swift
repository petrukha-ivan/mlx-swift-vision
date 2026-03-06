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

final class DetrForInstanceSegmentationProcessor: Processor {

    private let labels: [String]
    private let numQueries: Int
    private let imagePreprocessor: ImagePreprocessor

    init(modelConfig: DetrForObjectDetectionConfig, imagePreprocessor: ImagePreprocessor) {
        self.labels = modelConfig.id2label.flattened
        self.numQueries = modelConfig.queriesCount
        self.imagePreprocessor = imagePreprocessor
    }

    func preprocess(_ request: InstanceSegmentationRequest) throws -> ImageInput {
        try imagePreprocessor.preprocess(image: request.image)
    }

    func postprocess(_ output: DetrModelForInstanceSegmentation.Output, _ request: InstanceSegmentationRequest) throws -> [InstanceSegmentationResult] {
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
            mask.expandedDimensions(axis: -1)
                .interpolate(size: targetSize, mode: .linear(alignCorners: false))
                .squeezed()
        }

        let finalMask = MLX.stacked(interpolatedMasks, axis: 0)
            .flattened(start: 1, end: 2)
            .transposed(axes: [1, 0])
            .softmax(axis: -1)
            .argMax(axis: -1)
            .reshaped(targetSize)

        return (0..<keep.count).map { i in
            let (height, width) = finalMask.shape2
            let pixelValues = MLX.where(
                finalMask .== i,
                MLXArray(255, dtype: .uint8),
                MLXArray(0, dtype: .uint8)
            )

            let imageData = pixelValues.asData()
            let imageSize = CGSize(width: CGFloat(width), height: CGFloat(height))
            let mask = CIImage(
                bitmapData: imageData.data,
                bytesPerRow: width,
                size: imageSize,
                format: .L8,
                colorSpace: CGColorSpace(name: CGColorSpace.linearGray)
            )

            return InstanceSegmentationResult(
                mask: mask,
                label: labels[i],
                score: scores[i]
            )
        }
    }
}
