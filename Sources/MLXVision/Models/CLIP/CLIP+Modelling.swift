//
//  CLIP+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import Foundation
import MLXNN
import MLX

final class CLIPModel: Module, Predictor {

    @ModuleInfo(key: "text_model") var textModel: CLIPTextModel
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "vision_model") var visionModel: CLIPVisionModel
    @ModuleInfo(key: "visual_projection") var visionProjection: Linear
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray

    init(_ config: CLIPConfig) {
        _textModel.wrappedValue = CLIPTextModel(config.textConfig)
        _textProjection.wrappedValue = Linear(config.textConfig.hiddenSize, config.projectionDim, bias: false)
        _visionModel.wrappedValue = CLIPVisionModel(config.visionConfig)
        _visionProjection.wrappedValue = Linear(config.visionConfig.hiddenSize, config.projectionDim, bias: false)
        _logitScale.wrappedValue = MLXArray(config.logitScaleInitValue)
    }

    func predict(_ input: MultimodalInput) throws -> MLXArray {
        let textOutputs = textModel(input.textInput.textTokens)
        var textFeatures = textProjection(textOutputs)
        textFeatures /= textFeatures.norm()

        let visionOutputs = visionModel(input.imageInput.pixelValues)
        var visionFeatures = visionProjection(visionOutputs)
        visionFeatures /= visionFeatures.norm()

        let logitsPerText = MLX.matmul(textFeatures, visionFeatures.T) * logitScale.exp()
        let logitsPerImage = logitsPerText.T

        return logitsPerImage
    }
}

final class CLIPEmbeddingsModel: Module, Predictor {

    enum Input {
        case text(tokens: MLXArray)
        case image(pixelValues: MLXArray)
    }

    @ModuleInfo(key: "text_model") var textModel: CLIPTextModel
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "vision_model") var visionModel: CLIPVisionModel
    @ModuleInfo(key: "visual_projection") var visionProjection: Linear
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray

    init(_ config: CLIPConfig) {
        _textModel.wrappedValue = CLIPTextModel(config.textConfig)
        _textProjection.wrappedValue = Linear(config.textConfig.hiddenSize, config.projectionDim, bias: false)
        _visionModel.wrappedValue = CLIPVisionModel(config.visionConfig)
        _visionProjection.wrappedValue = Linear(config.visionConfig.hiddenSize, config.projectionDim, bias: false)
        _logitScale.wrappedValue = MLXArray(config.logitScaleInitValue)
    }

    func predict(_ input: Input) throws -> MLXArray {
        switch input {
        case .text(let tokens):
            let textOutputs = textModel(tokens)
            var textFeatures = textProjection(textOutputs)
            textFeatures /= textFeatures.norm()
            return textFeatures
        case .image(let pixelValues):
            let visionOutputs = visionModel(pixelValues)
            var visionFeatures = visionProjection(visionOutputs)
            visionFeatures /= visionFeatures.norm()
            return visionFeatures
        }
    }
}

final class CLIPTextModelWithProjection: Module {

    @ModuleInfo(key: "text_model") var textModel: CLIPTextModel
    @ModuleInfo(key: "text_projection") var textProjection: Linear

    init(_ config: CLIPTextConfig) {
        _textModel.wrappedValue = CLIPTextModel(config)
        _textProjection.wrappedValue = Linear(config.hiddenSize, config.projectionDim, bias: false)
    }

    func callAsFunction(_ inputIds: MLXArray, textMask: MLXArray) -> MLXArray {
        var hidden = textModel.embeddings(inputIds)
        let mask = MLXArray.createAdditiveCausalMask(32) .&& textMask
        hidden = textModel.encoder(hidden, mask: .array(mask))
        hidden = textModel.normalization(hidden)
        return hidden
    }
}

final class CLIPTextModel: Module {

    @ModuleInfo var embeddings: CLIPTextEmbeddings
    @ModuleInfo var encoder: CLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var normalization: LayerNorm

    init(_ config: CLIPTextConfig) {
        embeddings = CLIPTextEmbeddings(config)
        encoder = CLIPEncoder(config)
        _normalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("text_model.", with: "")
        }
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var hidden = embeddings(inputIds)
        hidden = encoder(hidden)
        hidden = normalization(hidden)
        let pooledOutput = hidden[MLXArray(0..<hidden.count), inputIds.argmax(axis: -1)]
        return pooledOutput
    }
}

final class CLIPTextEmbeddings: Module {

    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    @ParameterInfo(key: "_position_ids") var positionIds: MLXArray

    init(_ config: CLIPTextConfig) {
        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: config.maxPositionEmbeddings, dimensions: config.hiddenSize)
        _positionIds.wrappedValue = Array(0..<config.maxPositionEmbeddings).asMLXArray(dtype: .uint32).expandedDimensions(axis: 0)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.filterKeys {
            $0 != "position_ids"
        }
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let positionIds = positionIds[.ellipsis, 0..<inputIds.shape[1]]
        let inputEmbeddings = tokenEmbedding(inputIds)
        let positionEmbeddings = positionEmbedding(positionIds)
        let embeddings = inputEmbeddings + positionEmbeddings
        return embeddings
    }
}

final class CLIPVisionModel: Module {

    @ModuleInfo var embeddings: CLIPVisionEmbeddings
    @ModuleInfo var encoder: CLIPEncoder
    @ModuleInfo(key: "pre_layrnorm") var preNormalization: LayerNorm
    @ModuleInfo(key: "post_layernorm") var postNormalization: LayerNorm

    init(_ config: CLIPVisionConfig) {
        embeddings = CLIPVisionEmbeddings(config)
        encoder = CLIPEncoder(config)
        _preNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _postNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("vision_model.", with: "")
        }
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = embeddings(pixelValues)
        hidden = preNormalization(hidden)
        hidden = encoder(hidden, mask: .none)
        hidden = hidden[0..., 0, 0...]
        hidden = postNormalization(hidden)
        return hidden
    }
}

final class CLIPVisionEmbeddings: Module {

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "position_ids") var positionIds: MLXArray

    init(_ config: CLIPVisionConfig) {
        let numPatches = Int(pow(Double(config.imageSize / config.patchSize), 2))
        let numPositions = numPatches + 1
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize),
            bias: false
        )
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: numPositions, dimensions: config.hiddenSize)
        _classEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _positionIds.wrappedValue = MLXArray.zeros([1, numPositions])
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let patchEmbeddings = patchEmbedding(pixelValues).flattened(start: 1, end: 2)
        let classEmbeddings = classEmbedding.expandedDimensions(axes: [0, 1])
        let positionEmbeddings = positionEmbedding(positionIds)
        let embeddings = [classEmbeddings, patchEmbeddings].concatenated(axis: 1) + positionEmbeddings
        return embeddings
    }
}

final class CLIPEncoder: Module {

    @ModuleInfo var layers: [CLIPEncoderLayer]

    init(_ config: CLIPSharedConfig) {
        layers = (0..<config.numHiddenLayers).map { _ in
            CLIPEncoderLayer(config)
        }
    }

    func callAsFunction(_ hidden: MLXArray, mask: CLIPAttention.Mask = .causal) -> MLXArray {
        var hidden = hidden
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return hidden
    }
}

final class CLIPEncoderLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: CLIPAttention
    @ModuleInfo(key: "layer_norm1") var preNormalization: LayerNorm
    @ModuleInfo(key: "layer_norm2") var postNormalization: LayerNorm
    @ModuleInfo var mlp: CLIPMLP

    init(_ config: CLIPSharedConfig) {
        _attention.wrappedValue = CLIPAttention(config)
        _preNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _postNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        mlp = CLIPMLP(config)
    }

    func callAsFunction(_ hidden: MLXArray, mask: CLIPAttention.Mask = .causal) -> MLXArray {
        var hidden = hidden
        var residual = hidden
        hidden = preNormalization(hidden)
        hidden = attention(queries: hidden, keys: hidden, values: hidden, mask: mask)

        hidden += residual
        residual = hidden

        hidden = postNormalization(hidden)
        hidden = mlp(hidden)
        hidden += residual

        return hidden
    }
}

final class CLIPAttention: Module {

    typealias Mask = MLXFast.ScaledDotProductAttentionMaskMode

    let numHeads: Int

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    init(_ config: CLIPSharedConfig) {
        numHeads = config.numAttentionHeads
        _queryProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _outProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: Mask = .causal,
    ) -> MLXArray {
        var queries = queryProjection(queries)
        var keys = keyProjection(keys)
        var values = valueProjection(values)

        let (B, L, _) = queries.shape3
        let (_, S, _) = keys.shape3

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

        let scale = sqrt(1 / Float(queries.dim(-1)))
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outProjection(output)
    }
}

final class CLIPMLP: Module {

    @ModuleInfo(key: "fc1") var upProjection: Linear
    @ModuleInfo(key: "fc2") var downProjection: Linear
    @ModuleInfo var activation: UnaryLayer

    init(_ config: CLIPSharedConfig) {
        _upProjection.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _downProjection.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        activation = GELU(approximation: .none)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden
        hidden = upProjection(hidden)
        hidden = activation(hidden)
        hidden = downProjection(hidden)
        return hidden
    }
}

extension MLXArray {
    static func createAdditiveCausalMask(_ n: Int) -> MLXArray {
        let indices = MLXArray(0..<n)
        var mask = MLX.expandedDimensions(indices, axis: 1) .< MLX.expandedDimensions(indices, axis: 0)
        mask = .!mask
        return mask
    }
}
