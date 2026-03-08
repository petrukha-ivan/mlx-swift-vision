//
//  Sam3+Processing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import Foundation
import CoreImage
import ReerCodable
import MLX

@Codable
struct Sam3ProcessorConfig {

    @CodingKey("image_processor")
    var imageProcessor: ImagePreprocessorConfig
}

final class Sam3Processor: Processor {

    private let imagePreprocessor: ImagePreprocessor
    private let textTokenizer: CLIPTokenizer

    init(imagePreprocessor: ImagePreprocessor, textTokenizer: CLIPTokenizer) {
        self.imagePreprocessor = imagePreprocessor
        self.textTokenizer = textTokenizer
    }

    func preprocess(_ input: ZeroShotSegmentationRequest) throws -> MultimodalInput {
        let tokens = textTokenizer.encode(text: input.prompt)
        let paddingLength = 32  // TODO: Take from the config, input size should always match 'max_position_embeddings'
        let paddedTokens = MLXArray(tokens + Array(repeating: textTokenizer.eos, count: paddingLength))[..<paddingLength].expandedDimensions(axis: 0)
        let textMask = MLXArray(Array(repeating: 1, count: tokens.count) + Array(repeating: 0, count: paddingLength - tokens.count))
        let textInput = TextInput(textTokens: paddedTokens, textMask: textMask)
        let imageInput = try imagePreprocessor.preprocess(image: input.image)
        return MultimodalInput(
            textInput: textInput,
            imageInput: imageInput
        )
    }

    func postprocess(_ output: Sam3Model.Output, _ request: ZeroShotSegmentationRequest) throws -> [InstanceSegmentationResult] {
        let keep = MLXArray(output.predProbs.asArray(Float.self).indices(where: { $0 > request.scoreThreshold }))
        guard keep.count > 0 else {
            return []
        }

        let (xMin, yMin, xMax, yMax) = output.predBoxes.squeezed().split(axis: -1)
        let boxes = MLX.stacked([xMin, yMin, xMax - xMin, yMax - yMin], axis: -1)[keep].split(parts: keep.count)
        let scores = output.predProbs[0, keep].split(parts: keep.count)
        let masks = output.predMasks[0, keep].split(parts: keep.count)

        return zip(scores, masks, boxes).map { score, mask, bbox in
            let (_, height, width) = mask.shape3
            let pixelValues = MLX.where(
                mask .>= request.maskThreshold,
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
                bbox: bbox.asArray(Float.self),
                label: request.prompt,
                score: score.item(Float.self)
            )
        }
    }
}
