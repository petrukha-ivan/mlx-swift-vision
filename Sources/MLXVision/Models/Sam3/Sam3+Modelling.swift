//
//  Sam3+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import Foundation
import MLXNN
import MLX

final class Sam3Model: Module, Predictor {

    typealias Output = (
        pixelValues: MLXArray,
        semanticSeg: MLXArray,
        predLogits: MLXArray,
        predBoxes: MLXArray,
        predMasks: MLXArray
    )

    @ModuleInfo(key: "vision_encoder") var visionEncoder: Sam3VisionModel
    @ModuleInfo(key: "text_encoder") var textEncoder: CLIPTextModelWithProjection
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "geometry_encoder") var geometryEncoder: Sam3GeometryEncoder  // TODO: Add geometry features
    @ModuleInfo(key: "detr_encoder") var detrEncoder: Sam3DetrEncoder
    @ModuleInfo(key: "detr_decoder") var detrDecoder: Sam3DetrDecoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: Sam3MaskDecoder
    @ModuleInfo(key: "dot_product_scoring") var dotProductScoring: Sam3DotProductScoring

    init(_ config: Sam3Config) {
        _visionEncoder.wrappedValue = Sam3VisionModel(config.visionConfig)
        _textEncoder.wrappedValue = CLIPTextModelWithProjection(config.textConfig)
        _textProjection.wrappedValue = Linear(config.textConfig.hiddenSize, config.detrEncoderConfig.hiddenSize)
        _geometryEncoder.wrappedValue = Sam3GeometryEncoder(config.geometryEncoderConfig)
        _detrEncoder.wrappedValue = Sam3DetrEncoder(config.detrEncoderConfig)
        _detrDecoder.wrappedValue = Sam3DetrDecoder(config.detrDecoderConfig)
        _maskDecoder.wrappedValue = Sam3MaskDecoder(config.maskDecoderConfig)
        _dotProductScoring.wrappedValue = Sam3DotProductScoring(config)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.filterKeys {
            !$0.starts(with: "tracker_")
        }.renameKeys {
            $0.replacing("detector_model.", with: "")
        }
    }

    func predict(_ input: MultimodalInput) throws -> Output {
        let visionOutputs = visionEncoder(input.imageInput.pixelValues)
        let visionFeatures = visionOutputs.lastFPNHiddenStates.dropLast()
        let visionPositionEmbeddings = visionOutputs.lastFPNPositionEncodings.dropLast()

        var textFeatures = textEncoder(input.textInput.textTokens, textMask: input.textInput.textMask)
        textFeatures = textProjection(textFeatures)

        let encoderOutputs = detrEncoder(
            visionFeatures: visionFeatures.last!,
            visionPositionEmbeddings: visionPositionEmbeddings.last!,
            textFeatures: textFeatures,
            textMask: input.textInput.textMask
        )

        let decoderOutputs = detrDecoder(
            visionFeatures: encoderOutputs.lastHiddenState,
            visionPositionEmbeddings: encoderOutputs.flattenedPositionEmbeddings,
            spatialShapes: encoderOutputs.spatialShapes,
            textFeatures: textFeatures,
            textMask: input.textInput.textMask
        )

        let maskOutputs = maskDecoder(
            encoderHidden: encoderOutputs.lastHiddenState,
            decoderHidden: decoderOutputs.lastHiddenState,
            backboneFeatures: Array(visionFeatures),
            spatialShapes: encoderOutputs.spatialShapes,
            textFeatures: textFeatures,
            textMask: input.textInput.textMask
        )

        let logits = dotProductScoring(
            decoderHiddenState: decoderOutputs.allHiddenStates,
            textFeatures: textFeatures,
            textMask: input.textInput.textMask,
        )

        return (
            pixelValues: input.imageInput.pixelValues,
            semanticSeg: maskOutputs.semanticSeg,
            predLogits: logits,
            predBoxes: decoderOutputs.predBoxes,
            predMasks: maskOutputs.predMasks
        )
    }
}

final class Sam3VisionModel: Module {

    typealias Output = (
        lastHiddenState: MLXArray,
        lastFPNHiddenStates: [MLXArray],
        lastFPNPositionEncodings: [MLXArray]
    )

    @ModuleInfo var backbone: Sam3ViTModel
    @ModuleInfo var neck: Sam3VisionNeck

    init(_ config: Sam3VisionConfig) {
        backbone = Sam3ViTModel(config.backboneConfig)
        neck = Sam3VisionNeck(config)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> Output {
        let hidden = backbone(pixelValues)
        let (fpnHidden, fpnPositionEncoding) = neck(hidden)
        return (
            lastHiddenState: hidden,
            lastFPNHiddenStates: fpnHidden,
            lastFPNPositionEncodings: fpnPositionEncoding
        )
    }
}

final class Sam3VisionNeck: Module {

    typealias Output = (
        hiddenStates: [MLXArray],
        positionEncodings: [MLXArray]
    )

    @ModuleInfo(key: "fpn_layers") var layers: [UnaryLayer]
    @ModuleInfo(key: "_position_embedding") var positionEmbedding: Sam3SinePositionEmbedding

    init(_ config: Sam3VisionConfig) {
        _positionEmbedding.wrappedValue = Sam3SinePositionEmbedding(embeddingDimensions: config.fpnHiddenSize / 2)
        _layers.wrappedValue = config.scaleFactors.map {
            Sam3FPNLayer(config, $0)
        }
    }

    func callAsFunction(_ hidden: MLXArray) -> Output {
        var hiddenStates: [MLXArray] = []
        var positionEncodings: [MLXArray] = []

        for layer in layers {
            let hidden = layer(hidden)
            let positionEncoding = positionEmbedding(.ones(hidden.shape.dropLast()).asType(hidden.dtype))
            hiddenStates.append(hidden)
            positionEncodings.append(positionEncoding)
        }

        return (hiddenStates, positionEncodings)
    }
}

final class Sam3FPNLayer: Module, UnaryLayer {

    @ModuleInfo(key: "proj1") var upProjection: Conv2d
    @ModuleInfo(key: "proj2") var downProjection: Conv2d
    @ModuleInfo(key: "scale_layers") var layers: [UnaryLayer]

    init(_ config: Sam3VisionConfig, _ scale: Float) {
        let inputChannels = config.backboneConfig.hiddenSize
        let intermediateChannels = inputChannels / max(Int(scale), 1)
        let fpnHiddenSize = config.fpnHiddenSize
        _upProjection.wrappedValue = Conv2d(inputChannels: intermediateChannels, outputChannels: fpnHiddenSize, kernelSize: 1)
        _downProjection.wrappedValue = Conv2d(inputChannels: fpnHiddenSize, outputChannels: fpnHiddenSize, kernelSize: 3, padding: 1)
        _layers.wrappedValue =
            switch scale {
            case 0.5:
                [MaxPool2d(kernelSize: 2, stride: 2)]
            case 1.0:
                [Identity()]
            case 2.0:
                [ConvTransposed2d(inputChannels: inputChannels, outputChannels: inputChannels / 2, kernelSize: 2, stride: 2)]
            case 4.0:
                [
                    ConvTransposed2d(inputChannels: inputChannels, outputChannels: inputChannels / 2, kernelSize: 2, stride: 2),
                    GELU(approximation: .none),
                    ConvTransposed2d(inputChannels: inputChannels / 2, outputChannels: inputChannels / 4, kernelSize: 2, stride: 2),
                ]
            default:
                preconditionFailure("Sam3FPNLayer scale factor \(scale) is not supported.")
            }
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = layers(hidden)
        hidden = upProjection(hidden)
        hidden = downProjection(hidden)
        return hidden
    }
}

final class Sam3SinePositionEmbedding: Module {

    let embeddingDimensions: Int
    let temperature: Float
    let normalize: Bool
    let scale: Float

    init(embeddingDimensions: Int, temperature: Float = 10_000, normalize: Bool = true, scale: Float = 2 * .pi) {
        self.embeddingDimensions = embeddingDimensions
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
    }

    func encode(boxes: MLXArray) -> MLXArray {
        let xEmbeddings = boxes[0..., 0..., 0] * scale
        let yEmbeddings = boxes[0..., 0..., 1] * scale
        let wEmbeddings = boxes[0..., 0..., 2] * scale
        let hEmbeddings = boxes[0..., 0..., 3] * scale

        var dimT = Array(0..<embeddingDimensions).asMLXArray(dtype: boxes.dtype)
        dimT = temperature ** (2 * MLX.floorDivide(dimT, 2) / embeddingDimensions)

        var posX = xEmbeddings.expandedDimensions(axis: -1) / dimT
        posX = [
            posX[.ellipsis, .stride(from: 0, by: 2)].sin(),
            posX[.ellipsis, .stride(from: 1, by: 2)].cos(),
        ]
        .stacked(axis: 3)
        .flattened(start: 2)

        var posY = yEmbeddings.expandedDimensions(axis: -1) / dimT
        posY = [
            posY[.ellipsis, .stride(from: 0, by: 2)].sin(),
            posY[.ellipsis, .stride(from: 1, by: 2)].cos(),
        ]
        .stacked(axis: 3)
        .flattened(start: 2)

        var posW = wEmbeddings.expandedDimensions(axis: -1) / dimT
        posW = [
            posW[.ellipsis, .stride(from: 0, by: 2)].sin(),
            posW[.ellipsis, .stride(from: 1, by: 2)].cos(),
        ]
        .stacked(axis: 3)
        .flattened(start: 2)

        var posH = hEmbeddings.expandedDimensions(axis: -1) / dimT
        posH = [
            posH[.ellipsis, .stride(from: 0, by: 2)].sin(),
            posH[.ellipsis, .stride(from: 1, by: 2)].cos(),
        ]
        .stacked(axis: 3)
        .flattened(start: 2)

        return MLX.concatenated([posY, posX, posW, posH], axis: 2)
    }

    func callAsFunction(_ mask: MLXArray) -> MLXArray {
        var yEmbeddings = mask.cumsum(axis: 1)
        var xEmbeddings = mask.cumsum(axis: 2)
        if normalize {
            yEmbeddings = yEmbeddings / (yEmbeddings[0..., (yEmbeddings.shape[1] - 1)..., 0...] + 1e-6) * scale
            xEmbeddings = xEmbeddings / (xEmbeddings[0..., 0..., (xEmbeddings.shape[2] - 1)...] + 1e-6) * scale
        }

        var dimT = Array(0..<embeddingDimensions).asMLXArray(dtype: mask.dtype)
        dimT = temperature ** (2 * MLX.floorDivide(dimT, 2) / embeddingDimensions)

        var posY = yEmbeddings.expandedDimensions(axis: -1) / dimT
        posY = [
            posY[.ellipsis, .stride(from: 0, by: 2)].sin(),
            posY[.ellipsis, .stride(from: 1, by: 2)].cos(),
        ]
        .stacked(axis: 4)
        .flattened(start: 3)

        var posX = xEmbeddings.expandedDimensions(axis: -1) / dimT
        posX = [
            posX[.ellipsis, .stride(from: 0, by: 2)].sin(),
            posX[.ellipsis, .stride(from: 1, by: 2)].cos(),
        ]
        .stacked(axis: 4)
        .flattened(start: 3)

        return MLX.concatenated([posY, posX], axis: 3)
    }
}

final class Sam3ViTModel: Module {

    @ModuleInfo var embeddings: Sam3ViTEmbeddings
    @ModuleInfo(key: "layer_norm") var normalization: LayerNorm
    @ModuleInfo var layers: [UnaryLayer]

    init(_ config: Sam3ViTConfig) {
        _embeddings.wrappedValue = Sam3ViTEmbeddings(config)
        _normalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map {
            Sam3ViTLayer(config, config.globalAttentionIndices.contains($0) ? 0 : config.windowSize)
        }
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = embeddings(pixelValues)
        hidden = normalization(hidden)
        hidden = layers(hidden)
        return hidden
    }
}

final class Sam3ViTEmbeddings: Module {

    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: Conv2d
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    let patchSize: Int

    init(_ config: Sam3ViTConfig) {
        _patchEmbeddings.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: .init(config.patchSize),
            stride: .init(config.patchSize),
            bias: false
        )
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, 576, config.hiddenSize])
        patchSize = config.patchSize
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("patch_embeddings.projection", with: "patch_embeddings")
        }
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let (_, H, W, _) = pixelValues.shape4
        let hiddenSize = positionEmbeddings.shape[2]
        let pretrainSize = Int(pow(Double(positionEmbeddings.shape[1]), 0.5))

        var positionEmbeddings = positionEmbeddings.reshaped([1, pretrainSize, pretrainSize, hiddenSize]).movedAxis(source: 3, destination: 1)
        positionEmbeddings = MLX.tiled(positionEmbeddings, repetitions: [1, 1, H / patchSize / pretrainSize + 1, W / patchSize / pretrainSize + 1])
        positionEmbeddings = positionEmbeddings[.ellipsis, 0..<(H / patchSize), 0..<(W / patchSize)]
        positionEmbeddings = positionEmbeddings.movedAxis(source: 1, destination: 3)

        var patchEmbeddings = patchEmbeddings(pixelValues).flattened(start: 1, end: 2)
        patchEmbeddings = patchEmbeddings.reshaped(positionEmbeddings.shape)

        return patchEmbeddings + positionEmbeddings
    }
}

final class Sam3ViTLayer: Module, UnaryLayer {

    @ModuleInfo(key: "layer_norm1") var preNormalization: LayerNorm
    @ModuleInfo var rotaryEmbedding: Sam3ViTRotaryEmbedding
    @ModuleInfo var attention: Sam3ViTRoPEAttention
    @ModuleInfo(key: "layer_norm2") var postNormalization: LayerNorm
    @ModuleInfo var mlp: Sam3MLP

    let windowSize: Int

    init(_ config: Sam3ViTConfig, _ windowSize: Int) {
        let inputSize = config.imageSize / config.patchSize
        let rotaryInputSize = windowSize == 0 ? inputSize : windowSize
        let rotaryScale = Float(config.windowSize) / Float(rotaryInputSize)
        _preNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _rotaryEmbedding.wrappedValue = Sam3ViTRotaryEmbedding(config, rotaryInputSize, rotaryInputSize, rotaryScale)
        _attention.wrappedValue = Sam3ViTRoPEAttention(config)
        _postNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = Sam3MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self.windowSize = windowSize
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let (B, H, W, C) = hidden.shape4
        var hidden = hidden
        var residual = hidden
        hidden = preNormalization(hidden)

        if windowSize > 0 {  // TODO: Add padding to support more input sizes
            hidden =
                hidden
                .reshaped([B, H / windowSize, windowSize, W / windowSize, windowSize, C])
                .transposed(axes: [0, 1, 3, 2, 4, 5])
                .reshaped([-1, windowSize, windowSize, C])
        }

        hidden = attention(
            hidden,
            (
                rotaryEmbedding.embeddingsCos.asType(hidden.dtype),
                rotaryEmbedding.embeddingsSin.asType(hidden.dtype)
            )
        )

        if windowSize > 0 {
            hidden =
                hidden
                .reshaped([-1, H / windowSize, W / windowSize, windowSize, windowSize, C])
                .transposed(axes: [0, 1, 3, 2, 4, 5])
                .reshaped([B, H, W, C])
        }

        hidden += residual
        residual = hidden

        hidden = postNormalization(hidden)
        hidden = mlp(hidden)
        hidden += residual

        return hidden
    }
}

final class Sam3ViTRotaryEmbedding: Module {

    @ParameterInfo(key: "_embeddings_cos") var embeddingsCos: MLXArray
    @ParameterInfo(key: "_embeddings_sin") var embeddingsSin: MLXArray

    init(_ config: Sam3ViTConfig, _ xLim: Int, _ yLim: Int, _ scale: Float) {
        let dimensions = config.hiddenSize / config.numAttentionHeads
        let flattenedIndices = MLXArray(stride(from: 0, to: xLim * yLim, by: 1))
        let baseFreqs = 1 / (config.ropeTheta ** (MLXArray(stride(from: 0, to: dimensions, by: 4))[0..<dimensions / 4] / dimensions))
        let xFreqs = MLX.outer((flattenedIndices % xLim) * scale, baseFreqs)
        let yFreqs = MLX.outer(flattenedIndices.floorDivide(xLim) * scale, baseFreqs)
        var invFreq = MLX.concatenated([xFreqs, yFreqs], axis: -1)
        invFreq = MLX.repeated(invFreq, count: 2, axis: -1)
        _embeddingsCos.wrappedValue = invFreq.cos()
        _embeddingsSin.wrappedValue = invFreq.sin()
    }
}

final class Sam3ViTRoPEAttention: Module {

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outProjection: Linear

    let numHeads: Int

    init(_ config: Sam3ViTConfig) {
        _queryProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _outProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        numHeads = config.numAttentionHeads
    }

    func callAsFunction(_ hidden: MLXArray, _ positionEmbeddings: (MLXArray, MLXArray)) -> MLXArray {
        var queries = queryProjection(hidden)
        var keys = keyProjection(hidden)
        var values = valueProjection(hidden)

        let (B, W, H, C) = queries.shape4
        let L = W * H

        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

        let (cos, sin) = positionEmbeddings
        queries = (queries * cos) + (queries.rotatedPairwise() * sin)
        keys = (keys * cos) + (keys.rotatedPairwise() * sin)

        let scale = sqrt(1 / Float(queries.dim(-1)))
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: nil
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, H, W, C)

        return outProjection(output)
    }
}

final class Sam3Attention: Module {

    let numHeads: Int

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outProjection: Linear

    init(_ hiddenSize: Int, _ numAttentionHeads: Int) {
        self.numHeads = numAttentionHeads
        _queryProjection.wrappedValue = Linear(hiddenSize, hiddenSize)
        _keyProjection.wrappedValue = Linear(hiddenSize, hiddenSize)
        _valueProjection.wrappedValue = Linear(hiddenSize, hiddenSize)
        _outProjection.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXArray? = nil,
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

final class Sam3MLP: Module {

    @ModuleInfo(key: "fc1") var upProjection: Linear
    @ModuleInfo(key: "fc2") var downProjection: Linear
    @ModuleInfo var activation: UnaryLayer

    init(hiddenSize: Int, intermediateSize: Int, activation: UnaryLayer = GELU()) {
        _upProjection.wrappedValue = Linear(hiddenSize, intermediateSize)
        _downProjection.wrappedValue = Linear(intermediateSize, hiddenSize)
        self.activation = activation
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden
        hidden = upProjection(hidden)
        hidden = activation(hidden)
        hidden = downProjection(hidden)
        return hidden
    }
}

final class Sam3GeometryEncoder: Module {

    @ModuleInfo(key: "label_embed") var labelEmbedding: Embedding
    @ModuleInfo(key: "cls_embed") var classEmbedding: Embedding

    @ModuleInfo(key: "boxes_direct_project") var boxesDirectProjection: Linear
    @ModuleInfo(key: "boxes_pool_project") var boxesPoolingProjection: Conv2d
    @ModuleInfo(key: "boxes_pos_enc_project") var boxesPositionEncodingProjection: Linear

    @ModuleInfo(key: "final_proj") var finalProjection: Linear
    @ModuleInfo(key: "vision_layer_norm") var visionNormalization: LayerNorm
    @ModuleInfo(key: "prompt_layer_norm") var promptNormalization: LayerNorm
    @ModuleInfo(key: "output_layer_norm") var outputNormalization: LayerNorm

    @ModuleInfo var layers: [Sam3GeometryEncoderLayer]

    init(_ config: Sam3GeometryEncoderConfig) {
        _labelEmbedding.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.hiddenSize)
        _classEmbedding.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        _boxesDirectProjection.wrappedValue = Linear(4, config.hiddenSize)
        _boxesPoolingProjection.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: .init(config.roiSize)
        )
        _boxesPositionEncodingProjection.wrappedValue = Linear(config.hiddenSize + 2, config.hiddenSize)
        _finalProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        _visionNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _promptNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _outputNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.numLayers).map { _ in
            Sam3GeometryEncoderLayer(config)
        }
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        fatalError("Not implemented")
    }
}

final class Sam3GeometryEncoderLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: Sam3Attention
    @ModuleInfo(key: "cross_attn") var crossAttention: Sam3Attention
    @ModuleInfo(key: "layer_norm1") var preNormalization: LayerNorm
    @ModuleInfo(key: "layer_norm2") var postNormalization: LayerNorm
    @ModuleInfo(key: "layer_norm3") var outputNormalization: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: Sam3MLP

    init(_ config: Sam3GeometryEncoderConfig) {
        _attention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _crossAttention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _preNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _postNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _outputNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _mlp.wrappedValue = Sam3MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        fatalError("Not implemented")
    }
}

final class Sam3DetrEncoder: Module {

    typealias Output = (
        lastHiddenState: MLXArray,
        flattenedFeatures: MLXArray,
        flattenedPositionEmbeddings: MLXArray,
        spatialShapes: (Int, Int)
    )

    @ModuleInfo var layers: [Sam3DetrEncoderLayer]

    init(_ config: Sam3DetrEncoderConfig) {
        layers = (0..<config.numLayers).map { _ in
            Sam3DetrEncoderLayer(config)
        }
    }

    func callAsFunction(
        visionFeatures: MLXArray,
        visionPositionEmbeddings: MLXArray,
        textFeatures: MLXArray,
        textMask: MLXArray? = nil
    ) -> Output {
        let (_, H, W, _) = visionFeatures.shape4
        let flattenedFeatures = visionFeatures.flattened(start: 1, end: 2)
        let flattenedPositionEmbeddings = visionPositionEmbeddings.flattened(start: 1, end: 2)
        let textMask = MLX.tiled(textMask!, repetitions: [1, 1, H * W, 1]).asType(.bool)

        var hidden = flattenedFeatures
        for layer in layers {
            hidden = layer(
                textFeatures: textFeatures,
                textMask: textMask,
                visionFeatures: hidden,
                visionPositionEmbeddings: flattenedPositionEmbeddings,
            )
        }

        return (hidden, flattenedFeatures, flattenedPositionEmbeddings, (H, W))
    }
}

final class Sam3DetrEncoderLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: Sam3Attention
    @ModuleInfo(key: "cross_attn") var crossAttention: Sam3Attention
    @ModuleInfo(key: "layer_norm1") var preNormalization: LayerNorm
    @ModuleInfo(key: "layer_norm2") var postNormalization: LayerNorm
    @ModuleInfo(key: "layer_norm3") var outputNormalization: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: Sam3MLP

    init(_ config: Sam3DetrEncoderConfig) {
        _attention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _crossAttention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _preNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _postNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _outputNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _mlp.wrappedValue = Sam3MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize, activation: ReLU())
    }

    func callAsFunction(
        textFeatures: MLXArray,
        textMask: MLXArray?,
        visionFeatures: MLXArray,
        visionPositionEmbeddings: MLXArray
    ) -> MLXArray {
        var hidden = visionFeatures
        var residual = hidden

        hidden = preNormalization(hidden)
        hidden = attention(
            queries: hidden + visionPositionEmbeddings,
            keys: hidden + visionPositionEmbeddings,
            values: hidden,
            mask: nil
        )

        hidden += residual
        residual = hidden

        hidden = postNormalization(hidden)
        hidden = crossAttention(
            queries: hidden,
            keys: textFeatures,
            values: textFeatures,
            mask: textMask
        )

        hidden += residual
        residual = hidden

        hidden = outputNormalization(hidden)
        hidden = mlp(hidden)
        hidden += residual

        return hidden
    }
}

final class Sam3DetrDecoder: Module {

    typealias Output = (
        lastHiddenState: MLXArray,
        allHiddenStates: MLXArray,
        referenceBoxes: MLXArray,
        predBoxes: MLXArray,
        presenceLogits: MLXArray
    )

    @ModuleInfo var layers: [Sam3DetrDecoderLayer]
    @ModuleInfo(key: "output_layer_norm") var outputNormalization: LayerNorm
    @ModuleInfo(key: "box_head") var boxHead: Sam3DetrDecoderMLP
    @ModuleInfo(key: "query_embed") var queryEmbedding: Embedding
    @ModuleInfo(key: "reference_points") var referencePoints: Embedding
    @ModuleInfo(key: "presence_token") var presenceToken: Embedding
    @ModuleInfo(key: "presence_head") var presenceHead: Sam3DetrDecoderMLP
    @ModuleInfo(key: "presence_layer_norm") var presenceNormalization: LayerNorm
    @ModuleInfo(key: "ref_point_head") var refPointHead: Sam3DetrDecoderMLP
    @ModuleInfo(key: "box_rpb_embed_x") var boxRPBEmbeddingX: Sam3DetrDecoderMLP
    @ModuleInfo(key: "box_rpb_embed_y") var boxRPBEmbeddingY: Sam3DetrDecoderMLP
    @ModuleInfo(key: "_position_embedding") var positionEmbedding: Sam3SinePositionEmbedding

    init(_ config: Sam3DetrDecoderConfig) {
        layers = Array(repeating: config, count: config.numLayers).map(Sam3DetrDecoderLayer.init)
        _outputNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _boxHead.wrappedValue = Sam3DetrDecoderMLP(config.hiddenSize, config.hiddenSize, 4, 3)
        _queryEmbedding.wrappedValue = Embedding(embeddingCount: config.numQueries, dimensions: config.hiddenSize)
        _referencePoints.wrappedValue = Embedding(embeddingCount: config.numQueries, dimensions: 4)
        _presenceToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        _presenceHead.wrappedValue = Sam3DetrDecoderMLP(config.hiddenSize, config.hiddenSize, 1, 3)
        _presenceNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _refPointHead.wrappedValue = Sam3DetrDecoderMLP(config.hiddenSize * 2, config.hiddenSize, config.hiddenSize, 2)
        _boxRPBEmbeddingX.wrappedValue = Sam3DetrDecoderMLP(2, config.hiddenSize, config.numAttentionHeads, 2)
        _boxRPBEmbeddingY.wrappedValue = Sam3DetrDecoderMLP(2, config.hiddenSize, config.numAttentionHeads, 2)
        _positionEmbedding.wrappedValue = Sam3SinePositionEmbedding(embeddingDimensions: config.hiddenSize / 2, normalize: false)
    }

    func rpbMatrix(boxes: MLXArray, spatialShapes: (Int, Int)) -> MLXArray {
        let (height, width) = spatialShapes
        let components = boxes.split(parts: 4, axis: -1)
        let (cx, cy, w, h) = (components[0], components[1], components[2], components[3])
        let boxes = MLX.stacked([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis: -1).squeezed(axis: 2)
        let (B, Q, _) = boxes.shape3

        let coordsH = MLXArray(0..<height).asType(boxes.dtype) / height
        let coordsW = MLXArray(0..<width).asType(boxes.dtype) / width

        let scale = MLXArray(8).asType(boxes.dtype)
        let denominator = MLX.log2(scale).asType(boxes.dtype)

        var deltasY = (coordsH.reshaped([1, -1, 1]) - boxes.reshaped([-1, 1, 4])[0..., 0..., .stride(from: 1, to: 4, by: 2)]).reshaped([B, Q, -1, 2])
        var deltasYLog = deltasY * scale
        deltasYLog = MLX.sign(deltasYLog) * MLX.log2(MLX.abs(deltasYLog) + 1) / denominator
        deltasY = boxRPBEmbeddingY(deltasYLog)

        var deltasX = (coordsW.reshaped([1, -1, 1]) - boxes.reshaped([-1, 1, 4])[0..., 0..., .stride(from: 0, to: 3, by: 2)]).reshaped([B, Q, -1, 2])
        var deltasXLog = deltasX * scale
        deltasXLog = MLX.sign(deltasXLog) * MLX.log2(MLX.abs(deltasXLog) + 1) / denominator
        deltasX = boxRPBEmbeddingX(deltasXLog)

        var rpbMatrix = deltasY.expandedDimensions(axis: 3) + deltasX.expandedDimensions(axis: 2)
        rpbMatrix = rpbMatrix.flattened(start: 2, end: 3).transposed(axes: [0, 3, 1, 2])

        return rpbMatrix
    }

    func callAsFunction(
        visionFeatures: MLXArray,
        visionPositionEmbeddings: MLXArray,
        spatialShapes: (Int, Int),
        textFeatures: MLXArray,
        textMask: MLXArray? = nil
    ) -> Output {
        var hidden = [presenceToken.weight.expandedDimensions(axis: 0), queryEmbedding.weight.expandedDimensions(axis: 0)].concatenated(axis: 1)
        var referenceBoxes = referencePoints.weight.expandedDimensions(axis: 0).sigmoid()
        let textMask = MLX.tiled(textMask!, repetitions: [1, 1, 201, 1]).asType(.bool)

        var intermediateOutputs: [MLXArray] = []
        var intermediateBoxes: [MLXArray] = [referenceBoxes]
        var intermediatePresenceLogits: [MLXArray] = []

        for layer in layers {
            var queryPositionEncoding = positionEmbedding.encode(boxes: referenceBoxes)
            queryPositionEncoding = refPointHead(queryPositionEncoding)
            queryPositionEncoding = MLX.padded(queryPositionEncoding, widths: [0, IntOrPair((1, 0)), 0], mode: .constant, value: MLXArray(0))

            let rpbMatrix = rpbMatrix(boxes: referenceBoxes, spatialShapes: spatialShapes)
            let visionCrossAttentionMask = MLX.padded(rpbMatrix, widths: [0, 0, IntOrPair((1, 0)), 0], mode: .constant, value: MLXArray(0))

            hidden = layer(
                queries: hidden,
                queriesPositionEmbeddings: queryPositionEncoding,
                visionFeatures: visionFeatures,
                visionPositionEmbeddings: visionPositionEmbeddings,
                visionCrossAttentionMask: visionCrossAttentionMask,
                textFeatures: textFeatures,
                textMask: textMask
            )

            let referenceBoxesBeforeSigmoid = referenceBoxes.inverseSigmoid()
            let deltaBoxes = boxHead(outputNormalization(hidden[0..., 1...]))
            referenceBoxes = (deltaBoxes + referenceBoxesBeforeSigmoid).sigmoid()

            var presenceLogits = hidden[0..., 1...]
            presenceLogits = presenceNormalization(presenceLogits)
            presenceLogits = presenceHead(presenceLogits).squeezed(axis: -1)

            intermediateOutputs.append(outputNormalization(hidden[0..., 1...]))
            intermediateBoxes.append(referenceBoxes)
            intermediatePresenceLogits.append(presenceLogits)
        }

        var predBoxes = Array(intermediateBoxes.dropLast()).stacked().inverseSigmoid() + boxHead(intermediateOutputs.stacked())
        predBoxes = predBoxes.sigmoid()

        let components = predBoxes.split(parts: 4, axis: -1)
        let (cx, cy, w, h) = (components[0], components[1], components[2], components[3])
        predBoxes = MLX.stacked([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis: -1).squeezed(axis: 3)
        predBoxes = predBoxes[-1]

        return (
            intermediateOutputs.last!,
            intermediateOutputs.stacked(),
            referenceBoxes,
            predBoxes,
            intermediatePresenceLogits.stacked()
        )
    }
}

final class Sam3DetrDecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: Sam3Attention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNormalization: LayerNorm

    @ModuleInfo(key: "text_cross_attn") var textCrossAttention: Sam3Attention
    @ModuleInfo(key: "text_cross_attn_layer_norm") var textCrossAttentionNormalization: LayerNorm

    @ModuleInfo(key: "vision_cross_attn") var visionCrossAttention: Sam3Attention
    @ModuleInfo(key: "vision_cross_attn_layer_norm") var visionCrossAttentionNormalization: LayerNorm

    @ModuleInfo var mlp: Sam3MLP
    @ModuleInfo(key: "mlp_layer_norm") var mlpNormalization: LayerNorm

    init(_ config: Sam3DetrDecoderConfig) {
        _attention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _attentionNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _textCrossAttention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _textCrossAttentionNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _visionCrossAttention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _visionCrossAttentionNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _mlp.wrappedValue = Sam3MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize, activation: ReLU())
        _mlpNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
    }

    func callAsFunction(
        queries: MLXArray,
        queriesPositionEmbeddings: MLXArray,
        visionFeatures: MLXArray,
        visionPositionEmbeddings: MLXArray,
        visionCrossAttentionMask: MLXArray,
        textFeatures: MLXArray,
        textMask: MLXArray? = nil
    ) -> MLXArray {
        var hidden = queries
        var residual = hidden

        var queriesPositionEncoding = hidden + queriesPositionEmbeddings

        hidden = attention(
            queries: queriesPositionEncoding,
            keys: queriesPositionEncoding,
            values: hidden
        )

        hidden += residual
        hidden = attentionNormalization(hidden)
        residual = hidden

        queriesPositionEncoding = hidden + queriesPositionEmbeddings
        var keysPositionEncoding = visionFeatures + visionPositionEmbeddings
        hidden = textCrossAttention(
            queries: queriesPositionEncoding,
            keys: textFeatures,
            values: textFeatures,
            mask: textMask
        )

        hidden += residual
        hidden = textCrossAttentionNormalization(hidden)
        residual = hidden

        queriesPositionEncoding = hidden + queriesPositionEmbeddings
        keysPositionEncoding = visionFeatures + visionPositionEmbeddings
        hidden = visionCrossAttention(
            queries: queriesPositionEncoding,
            keys: keysPositionEncoding,
            values: visionFeatures,
            mask: visionCrossAttentionMask
        )

        hidden += residual
        hidden = visionCrossAttentionNormalization(hidden)
        residual = hidden

        hidden = mlp(hidden)
        hidden += residual
        hidden = mlpNormalization(hidden)

        return hidden
    }
}

final class Sam3DetrDecoderMLP: Module {

    @ModuleInfo var layers: [UnaryLayer]

    init(
        _ inputDimensions: Int,
        _ hiddenDimensions: Int,
        _ outputDimensions: Int,
        _ numLayers: Int
    ) {
        layers =
            [
                Linear(inputDimensions, hiddenDimensions)
            ]
            + (0..<numLayers - 2).map { _ in
                Linear(inputDimensions, hiddenDimensions)
            } + [
                Linear(hiddenDimensions, outputDimensions)
            ]
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing(/layer(\d)/) {
                "layers.\(Int($0.1)! - 1)"
            }
        }
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden
        for layer in layers.dropLast() {
            hidden = layer(hidden)
            hidden = relu(hidden)
        }

        if let last = layers.last {
            hidden = last(hidden)
        }

        return hidden
    }
}

final class Sam3MaskDecoder: Module {

    typealias Output = (
        predMasks: MLXArray,
        semanticSeg: MLXArray
    )

    @ModuleInfo(key: "pixel_decoder") var pixelDecoder: Sam3PixelDecoder
    @ModuleInfo(key: "mask_embedder") var maskEmbedder: Sam3MaskEmbedder
    @ModuleInfo(key: "instance_projection") var instanceProjection: Conv2d
    @ModuleInfo(key: "semantic_projection") var semanticProjection: Conv2d
    @ModuleInfo(key: "prompt_cross_attn") var promptCrossAttention: Sam3Attention
    @ModuleInfo(key: "prompt_cross_attn_norm") var promptCrossAttentionNormalization: LayerNorm

    init(_ config: Sam3MaskDecoderConfig) {
        _pixelDecoder.wrappedValue = Sam3PixelDecoder(config)
        _maskEmbedder.wrappedValue = Sam3MaskEmbedder(config)
        _instanceProjection.wrappedValue = Conv2d(inputChannels: config.hiddenSize, outputChannels: config.hiddenSize, kernelSize: 1)
        _semanticProjection.wrappedValue = Conv2d(inputChannels: config.hiddenSize, outputChannels: 1, kernelSize: 1)
        _promptCrossAttention.wrappedValue = Sam3Attention(config.hiddenSize, config.numAttentionHeads)
        _promptCrossAttentionNormalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
    }

    func callAsFunction(
        encoderHidden: MLXArray,
        decoderHidden: MLXArray,
        backboneFeatures: [MLXArray],
        spatialShapes: (Int, Int),
        textFeatures: MLXArray,
        textMask: MLXArray?
    ) -> Output {
        var hidden = encoderHidden
        var residual = hidden

        let textMask = MLX.tiled(textMask!, repetitions: [1, 1, encoderHidden.shape[1], 1]).asType(.bool)  // TODO
        hidden = promptCrossAttentionNormalization(hidden)
        hidden = promptCrossAttention(
            queries: hidden,
            keys: textFeatures,
            values: textFeatures,
            mask: textMask
        )

        hidden += residual
        residual = hidden

        let visionFeatures = Array(backboneFeatures.dropLast() + [hidden.reshaped([1, spatialShapes.0, spatialShapes.1, -1])])
        let pixelEmbeddings = pixelDecoder(features: visionFeatures)

        let instanceEmbeddings = instanceProjection(pixelEmbeddings)
        let maskEmbeddings = maskEmbedder(decoderHidden)

        let predMasks = MLX.einsum("bqc,bhwc->bqhw", maskEmbeddings, instanceEmbeddings)
        let semanticSeg = semanticProjection(pixelEmbeddings)

        return (predMasks, semanticSeg)
    }
}

final class Sam3PixelDecoder: Module {

    @ModuleInfo var layers: [Sam3PixelDecoderLayer]

    init(_ config: Sam3MaskDecoderConfig) {
        layers = (0..<config.numUpsamplingStages).map { _ in
            Sam3PixelDecoderLayer(config)
        }
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing(/conv_layers\.(\d)/) {
                "layers.\($0.1).projection"
            }.replacing(/norms\.(\d)/) {
                "layers.\($0.1).normalization"
            }
        }
    }

    func callAsFunction(features: [MLXArray]) -> MLXArray {
        var hidden = features.last!

        for (layer, feature) in zip(layers, features.dropLast().reversed()) {
            hidden = hidden.interpolate(size: Array(feature.shape.dropFirst().dropLast()), mode: .nearest)
            hidden += feature
            hidden = layer(hidden)
        }

        return hidden
    }
}

final class Sam3PixelDecoderLayer: Module {

    @ModuleInfo var projection: Conv2d
    @ModuleInfo var normalization: GroupNorm

    init(_ config: Sam3MaskDecoderConfig) {
        _projection.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        _normalization.wrappedValue = GroupNorm(groupCount: 8, dimensions: config.hiddenSize, pytorchCompatible: true)
    }

    func callAsFunction(_ feature: MLXArray) -> MLXArray {
        var hidden = feature
        hidden = projection(hidden)
        hidden = normalization(hidden)
        hidden = relu(hidden)
        return hidden
    }
}

final class Sam3MaskEmbedder: Module {

    @ModuleInfo var layers: [UnaryLayer]

    init(_ config: Sam3MaskDecoderConfig) {
        layers = [
            Linear(config.hiddenSize, config.hiddenSize),
            Linear(config.hiddenSize, config.hiddenSize),
            Linear(config.hiddenSize, config.hiddenSize),
        ]
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden
        for layer in layers.dropLast() {
            hidden = layer(hidden)
            hidden = relu(hidden)
        }

        if let last = layers.last {
            hidden = last(hidden)
        }

        return hidden
    }
}

final class Sam3DotProductScoring: Module {

    @ModuleInfo(key: "text_mlp") var textMLP: Sam3DetrDecoderMLP
    @ModuleInfo(key: "text_mlp_out_norm") var textMLPNormalization: LayerNorm
    @ModuleInfo(key: "text_proj") var textProjection: Linear
    @ModuleInfo(key: "query_proj") var queryProjection: Linear

    let scale: Float

    init(_ config: Sam3Config) {
        _textMLP.wrappedValue = Sam3DetrDecoderMLP(
            config.detrDecoderConfig.hiddenSize,
            config.detrDecoderConfig.intermediateSize,
            config.detrDecoderConfig.hiddenSize,
            2
        )
        _textMLPNormalization.wrappedValue = LayerNorm(dimensions: config.detrDecoderConfig.hiddenSize)
        _textProjection.wrappedValue = Linear(config.detrDecoderConfig.hiddenSize, config.detrDecoderConfig.hiddenSize)
        _queryProjection.wrappedValue = Linear(config.detrDecoderConfig.hiddenSize, config.detrDecoderConfig.hiddenSize)
        scale = 1.0 / sqrt(Float(config.detrDecoderConfig.hiddenSize))
    }

    func callAsFunction(
        decoderHiddenState: MLXArray,
        textFeatures: MLXArray,
        textMask: MLXArray
    ) -> MLXArray {

        var textFeatures = textFeatures
        let residual = textFeatures
        textFeatures = textMLP(textFeatures)
        textFeatures += residual
        textFeatures = textMLPNormalization(textFeatures)

        textFeatures = textFeatures * textMask.expandedDimensions(axes: [0, -1])
        textFeatures = textFeatures.sum(axis: 1) / textMask.sum()

        let projectedTextFeatures = textProjection(textFeatures).expandedDimensions(axes: [0, -1])
        let projectedQueries = queryProjection(decoderHiddenState)
        var scores = MLX.matmul(projectedQueries, projectedTextFeatures)
        scores = scores * scale
        scores = scores.squeezed(axis: -1)
        scores = scores[-1]

        return scores
    }
}
