//
//  SigLIP+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import Foundation
import MLX
import MLXNN

final class SigLIPModel: Module, Predictor {

    typealias Output = (
        logits: MLXArray,
        probs: MLXArray
    )

    @ModuleInfo(key: "text_model") var textModel: SigLIPTextModel
    @ModuleInfo(key: "vision_model") var visionModel: SigLIPVisionModel
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray
    @ParameterInfo(key: "logit_bias") var logitBias: MLXArray

    init(_ config: SigLIPConfig) {
        _textModel.wrappedValue = SigLIPTextModel(config.textConfig)
        _visionModel.wrappedValue = SigLIPVisionModel(config.visionConfig)
        _logitScale.wrappedValue = MLXArray([0])
        _logitBias.wrappedValue = MLXArray([0])
    }

    private lazy var _predict = MLX.compile { [unowned self] inputs in
        let textTokens = inputs[0]
        let pixelValues = inputs[1]

        var textEmbeddings = textModel(textTokens)
        textEmbeddings /= textEmbeddings.norm()

        var imageEmbeddings = visionModel(pixelValues)
        imageEmbeddings /= imageEmbeddings.norm()

        let logitsPerText = MLX.matmul(textEmbeddings, imageEmbeddings.T) * logitScale.exp() + logitBias
        let logitsPerImage = logitsPerText.T
        let probs = logitsPerImage.squeezed().sigmoid()
        return [logitsPerImage, probs]
    }

    func predict(_ input: MultimodalInput) throws -> Output {
        let outputs = _predict([input.textInput.textTokens, input.imageInput.pixelValues])
        return (outputs[0], outputs[1])
    }
}

final class SigLIPEmbeddingsModel: Module, Predictor {

    enum Input {
        case text(tokens: MLXArray)
        case image(pixelValues: MLXArray)
    }

    @ModuleInfo(key: "text_model") var textModel: SigLIPTextModel
    @ModuleInfo(key: "vision_model") var visionModel: SigLIPVisionModel
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray
    @ParameterInfo(key: "logit_bias") var logitBias: MLXArray

    init(_ config: SigLIPConfig) {
        _textModel.wrappedValue = SigLIPTextModel(config.textConfig)
        _visionModel.wrappedValue = SigLIPVisionModel(config.visionConfig)
        _logitScale.wrappedValue = MLXArray([0])
        _logitBias.wrappedValue = MLXArray([0])
    }

    private lazy var _predictText = MLX.compile { [unowned self] inputs in
        let tokens = inputs[0]
        var textEmbeddings = textModel(tokens)
        textEmbeddings /= textEmbeddings.norm()
        return [textEmbeddings]
    }

    private lazy var _predictImage = MLX.compile { [unowned self] inputs in
        let pixelValues = inputs[0]
        var imageEmbeddings = visionModel(pixelValues)
        imageEmbeddings /= imageEmbeddings.norm()
        return [imageEmbeddings]
    }

    func predict(_ input: Input) throws -> MLXArray {
        switch input {
        case .text(let tokens):
            let outputs = _predictText([tokens])
            return outputs[0]
        case .image(let pixelValues):
            let outputs = _predictImage([pixelValues])
            return outputs[0]
        }
    }
}

final class SigLIPTextModel: Module {

    @ModuleInfo var embeddings: SigLIPTextEmbeddings
    @ModuleInfo var encoder: CLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var normalization: LayerNorm
    @ModuleInfo(key: "head") var head: Linear

    init(_ config: SigLIPTextConfig) {
        embeddings = SigLIPTextEmbeddings(config)
        encoder = CLIPEncoder(config)
        _normalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _head.wrappedValue = Linear(config.hiddenSize, config.projectionSize ?? config.hiddenSize)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("text_model.", with: "")
        }
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var hidden = embeddings(inputIds)
        hidden = encoder(hidden, mask: .none)
        hidden = normalization(hidden)
        let pooledOutput = hidden[0..., hidden.shape[1] - 1, 0...]
        return head(pooledOutput)
    }
}

final class SigLIPTextEmbeddings: Module {

    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    @ParameterInfo(key: "_position_ids") var positionIds: MLXArray

    init(_ config: SigLIPTextConfig) {
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
        return inputEmbeddings + positionEmbeddings
    }
}

final class SigLIPVisionModel: Module {

    @ModuleInfo var embeddings: SigLIPVisionEmbeddings
    @ModuleInfo var encoder: CLIPEncoder
    @ModuleInfo(key: "post_layernorm") var normalization: LayerNorm
    @ModuleInfo(key: "head") var head: SigLIPMultiheadAttentionPoolingHead

    init(_ config: SigLIPVisionConfig) {
        embeddings = SigLIPVisionEmbeddings(config)
        encoder = CLIPEncoder(config)
        _normalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _head.wrappedValue = SigLIPMultiheadAttentionPoolingHead(config)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = embeddings(pixelValues)
        hidden = encoder(hidden, mask: .none)
        hidden = normalization(hidden)
        return head(hidden)
    }
}

final class SigLIPVisionEmbeddings: Module {

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    @ModuleInfo(key: "_position_ids") var positionIds: MLXArray

    init(_ config: SigLIPVisionConfig) {
        let numPatches = Int(pow(Double(config.imageSize / config.patchSize), 2))
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize)
        )
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: numPatches, dimensions: config.hiddenSize)
        _positionIds.wrappedValue = MLXArray(Array(0..<numPatches)).expandedDimensions(axis: 0)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let patchEmbeddings = patchEmbedding(pixelValues).flattened(start: 1, end: 2)
        let positionEmbeddings = positionEmbedding(positionIds)
        return patchEmbeddings + positionEmbeddings
    }
}

final class SigLIPMultiheadAttentionPoolingHead: Module {

    @ParameterInfo(key: "probe") var probe: MLXArray
    @ModuleInfo(key: "attention") var attention: SigLIPPoolingAttention
    @ModuleInfo(key: "layernorm") var normalization: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: CLIPMLP

    init(_ config: SigLIPVisionConfig) {
        _probe.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
        _attention.wrappedValue = SigLIPPoolingAttention(config)
        _normalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = CLIPMLP(config)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let batchSize = hidden.shape[0]
        let probe = MLX.repeated(probe, count: batchSize, axis: 0)
        let attended = attention(queries: probe, keys: hidden, values: hidden)
        let normalized = normalization(attended)
        let pooled = attended + mlp(normalized)
        return pooled[0..., 0, 0...]
    }
}

final class SigLIPPoolingAttention: Module {

    let numHeads: Int

    @ParameterInfo(key: "in_proj_weight") var inProjectionWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjectionBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    init(_ config: SigLIPVisionConfig) {
        numHeads = config.numAttentionHeads
        _inProjectionWeight.wrappedValue = MLXArray.zeros([config.hiddenSize * 3, config.hiddenSize])
        _inProjectionBias.wrappedValue = MLXArray.zeros([config.hiddenSize * 3])
        _outProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        func linear(_ input: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
            MLX.matmul(input, weight.T) + bias
        }

        let (qWeight, kWeight, vWeight) = inProjectionWeight.split(axis: 0)
        let (qBias, kBias, vBias) = inProjectionBias.split(axis: 0)

        var queries = linear(queries, weight: qWeight, bias: qBias)
        var keys = linear(keys, weight: kWeight, bias: kBias)
        var values = linear(values, weight: vWeight, bias: vBias)

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
            mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outProjection(output)
    }
}
