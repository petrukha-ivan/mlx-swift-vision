//
//  SigLIP+Processing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import Foundation
import MLX

final class SigLIPProcessor: Processor {

    private let imagePreprocessor: ImagePreprocessor
    private let textTokenizer: SigLIPTokenizer

    init(imagePreprocessor: ImagePreprocessor, textTokenizer: SigLIPTokenizer) {
        self.imagePreprocessor = imagePreprocessor
        self.textTokenizer = textTokenizer
    }

    func preprocess(_ input: ZeroShotClassificationRequest) throws -> MultimodalInput {
        let paddingLength = 64
        let tokens = input.labels
            .map { textTokenizer.encode(text: $0) }
            .map { $0 + Array(repeating: textTokenizer.eos, count: paddingLength - $0.count) }
        let paddedTokens = MLX.stacked(tokens.map({ MLXArray($0) }))
        let textMask = paddedTokens .!= textTokenizer.eos
        let textInput = TextInput(textTokens: paddedTokens, textMask: textMask)
        let imageInput = try imagePreprocessor.preprocess(image: input.image)
        return MultimodalInput(
            textInput: textInput,
            imageInput: imageInput
        )
    }

    func postprocess(_ logits: MLXArray, _ request: ZeroShotClassificationRequest) throws -> [ClassificationResult] {
        let scores = logits.squeezed().sigmoid().asArray(Float.self)
        return zip(request.labels, scores).map { label, score in
            ClassificationResult(
                label: label,
                score: score
            )
        }
    }
}

final class SigLIPEmbeddingsProcessor: Processor {

    private let imagePreprocessor: ImagePreprocessor
    private let textTokenizer: SigLIPTokenizer

    init(imagePreprocessor: ImagePreprocessor, textTokenizer: SigLIPTokenizer) {
        self.imagePreprocessor = imagePreprocessor
        self.textTokenizer = textTokenizer
    }

    func preprocess(_ input: EmbeddingsExtractionRequest) throws -> SigLIPEmbeddingsModel.Input {
        switch input {
        case .text(let text):
            let tokens = textTokenizer.encode(text: text)
            let tokenIds = MLXArray(tokens).expandedDimensions(axis: 0)
            return .text(tokens: tokenIds)
        case .image(let image):
            let pixelValues = try imagePreprocessor.preprocess(image: image).pixelValues
            return .image(pixelValues: pixelValues)
        }
    }

    func postprocess(_ embeddings: MLXArray, _ request: EmbeddingsExtractionRequest) throws -> Embeddings {
        embeddings.squeezed().asArray(Float.self)
    }
}
