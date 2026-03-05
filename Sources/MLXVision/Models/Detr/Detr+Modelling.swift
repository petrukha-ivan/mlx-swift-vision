//
//  Detr+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 02.11.2025.
//

import Foundation
import MLXNN
import MLX

final class DetrModelForObjectDetection: Module, Predictor {

    typealias Output = (
        logits: MLXArray,
        boxes: MLXArray
    )

    @ModuleInfo var model: DetrModel
    @ModuleInfo(key: "bbox_predictor") var bboxPredictor: DetrPredictionHead
    @ModuleInfo(key: "class_labels_classifier") var classLabelsClassifier: Linear

    init(_ config: DetrForObjectDetectionConfig) {
        _model.wrappedValue = DetrModel(config)
        _bboxPredictor.wrappedValue = DetrPredictionHead(inputDimensions: config.dimensions, hiddenDimensions: config.dimensions)
        _classLabelsClassifier.wrappedValue = Linear(config.dimensions, config.id2label.count + 1)
    }

    private lazy var _predict = MLX.compile { [unowned self] inputs in
        let pixelValues = inputs[0]
        let outputs = model(pixelValues).decoderOutputs
        let logits = classLabelsClassifier(outputs)
        let boxes = bboxPredictor(outputs).sigmoid().squeezed()
        return [logits, boxes]
    }

    func predict(_ input: ImageInput) throws -> Output {
        let outputs = _predict([input.pixelValues])
        return (outputs[0], outputs[1])
    }
}

final class DetrModelForImageSegmentation: Module, Predictor {

    typealias Output = (
        pixelValues: MLXArray,
        logits: MLXArray,
        segmentationMask: MLXArray
    )

    @ModuleInfo var model: DetrModel
    @ModuleInfo(key: "mask_head") var maskHead: DetrMaskHead
    @ModuleInfo(key: "bbox_attention") var bboxAttention: DetrMaskHeadAttention
    @ModuleInfo(key: "bbox_predictor") var bboxPredictor: DetrPredictionHead
    @ModuleInfo(key: "class_labels_classifier") var classLabelsClassifier: Linear

    init(_ config: DetrForObjectDetectionConfig) {
        model = DetrModel(config)
        _maskHead.wrappedValue = DetrMaskHead(config)
        _bboxAttention.wrappedValue = DetrMaskHeadAttention(config)
        _bboxPredictor.wrappedValue = DetrPredictionHead(inputDimensions: config.dimensions, hiddenDimensions: config.dimensions)
        _classLabelsClassifier.wrappedValue = Linear(config.dimensions, config.id2label.count + 1)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("detr.", with: "")
        }
    }

    private lazy var _predict = MLX.compile { [unowned self] inputs in
        let pixelValues = inputs[0]
        let outputs = model(pixelValues)
        let features = outputs.features
        let featureMask = outputs.featureMask
        let (B, H, W) = featureMask.shape3
        let logits = classLabelsClassifier(outputs.decoderOutputs)
        let bboxMask = bboxAttention(queries: outputs.decoderOutputs, keys: outputs.encoderOutputs.reshaped(B, H, W, -1), mask: featureMask)
        let fpnFeatures = [features[2], features[1], features[0]]
        let segMask = maskHead(outputs.projectedFeatures, bboxAttention: bboxMask, fpns: fpnFeatures).squeezed()
        return [logits, segMask]
    }

    func predict(_ input: ImageInput) throws -> Output {
        let outputs = _predict([input.pixelValues])
        return (input.pixelValues, outputs[0], outputs[1])
    }
}

final class DetrModel: Module {

    typealias Output = (
        decoderOutputs: MLXArray,
        encoderOutputs: MLXArray,
        features: [MLXArray],
        featureMask: MLXArray,
        projectedFeatures: MLXArray
    )

    let backbone: DetrConvModel
    let encoder: DetrEncoder
    let decoder: DetrDecoder

    @ModuleInfo(key: "input_projection") var inputProjection: Conv2d
    @ModuleInfo(key: "query_position_embeddings") var queryPositionEmbeddings: Embedding
    @ModuleInfo(key: "_query_indices") var queryIndices: MLXArray

    init(_ config: DetrConfig) {
        backbone = DetrConvModel(config)
        encoder = DetrEncoder(config)
        decoder = DetrDecoder(config)
        _inputProjection.wrappedValue = Conv2d(inputChannels: 2048, outputChannels: config.dimensions, kernelSize: 1)
        _queryPositionEmbeddings.wrappedValue = Embedding(embeddingCount: config.queriesCount, dimensions: config.dimensions)
        _queryIndices.wrappedValue = Array(0..<config.queriesCount).asMLXArray(dtype: .int32)
    }

    func callAsFunction(_ pixelValues: MLXArray, _ pixelMask: MLXArray? = nil) -> Output {
        let (B, H, W, _) = pixelValues.shape4
        let (features, featureMask, positionEncoding) = backbone.features(pixelValues, pixelMask ?? MLXArray.ones([B, H, W], dtype: .bool))

        let projectedFeatures = inputProjection(features.last!)
        let flattenedFeatures = projectedFeatures.flattened(start: 1, end: 2)
        let flattenedFeatureMask = featureMask.flattened(start: 1, end: 2)
        let flattenedPositionEncoding = positionEncoding.flattened(start: 1, end: 2)
        let encoderOutputs = encoder(
            flattenedFeatures,
            flattenedFeatureMask,
            flattenedPositionEncoding
        )

        var queryPositions = queryPositionEmbeddings(queryIndices).expandedDimensions(axis: 0)
        queryPositions = MLX.repeated(queryPositions, count: B, axis: 0)

        let decoderOutputs = decoder(
            hiddenStates: .zeros(like: queryPositions),
            objectQueries: queryPositions,
            encoderOutputs: encoderOutputs,
            encoderMask: flattenedFeatureMask,
            encoderPositions: flattenedPositionEncoding
        )

        return (
            decoderOutputs: decoderOutputs,
            encoderOutputs: encoderOutputs,
            features: features,
            featureMask: featureMask,
            projectedFeatures: projectedFeatures
        )
    }
}

final class DetrConvModel: Module {

    typealias FeaturesOutput = (
        features: [MLXArray],
        featureMask: MLXArray,
        positionEncoding: MLXArray
    )

    @ModuleInfo(key: "conv_encoder") var convEncoder: DetrConvEncoder
    @ModuleInfo(key: "position_embedding") var positionEmbedding: DetrPositionEmbedding

    init(_ config: DetrConfig) {
        _convEncoder.wrappedValue = DetrConvEncoder(config)
        switch config.positionEmbeddingType {
        case .sine:
            _positionEmbedding.wrappedValue = DetrSinePositionEmbedding(embeddingDimensions: config.dimensions / 2)
        case .learned:
            _positionEmbedding.wrappedValue = DetrLearnedPositionEmbedding(
                embeddingDimensions: config.dimensions / 2,
                maxSize: config.positionEmbeddingMaxSize
            )
        }
    }

    func features(_ pixelValues: MLXArray, _ pixelMask: MLXArray) -> FeaturesOutput {
        let (features, mask) = convEncoder(pixelValues, pixelMask)
        let positionEncoding = positionEmbedding(features.last!, mask.last!)
        return (features, mask.last!, positionEncoding)
    }
}

final class DetrConvEncoder: Module {

    typealias Output = (
        features: [MLXArray],
        masks: [MLXArray]
    )

    let model: ResNetModel

    init(_ config: DetrConfig) {
        model = ResNetModel(ResNetConfig.all[config.backbone]!)
    }

    func callAsFunction(_ pixelValues: MLXArray, _ pixelMask: MLXArray) -> Output {
        let features = model.features(pixelValues)
        let masks = features.map {
            pixelMask.expandedDimensions(axis: -1)
                .interpolate(size: $0.shape.dropFirst().dropLast())
                .squeezed(axis: -1)
        }
        return (features, masks)
    }
}

final class DetrEncoder: Module {

    let layers: [DetrEncoderLayer]

    init(_ config: DetrConfig) {
        layers = (0..<config.encoderLayersCount).map { _ in
            DetrEncoderLayer(config)
        }
    }

    func callAsFunction(_ inputEmbeddings: MLXArray, _ attentionMask: MLXArray, _ positionEncoding: MLXArray) -> MLXArray {
        layers.reduce(inputEmbeddings) { hiddenStates, layer in
            layer(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                positionEncoding: positionEncoding
            )
        }
    }
}

final class DetrEncoderLayer: Module {

    let activation: UnaryLayer

    @ModuleInfo(key: "self_attn") var attention: DetrAttention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNormalization: LayerNorm

    @ModuleInfo(key: "fc1") var feedforwardInput: Linear
    @ModuleInfo(key: "fc2") var feedforwardOutput: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNormalization: LayerNorm

    init(_ config: DetrConfig) {
        activation = ReLU()
        _attention.wrappedValue = DetrAttention(config)
        _attentionNormalization.wrappedValue = LayerNorm(dimensions: config.dimensions)
        _feedforwardInput.wrappedValue = Linear(config.dimensions, config.encoderFeedforwardDimensions)
        _feedforwardOutput.wrappedValue = Linear(config.encoderFeedforwardDimensions, config.dimensions)
        _finalNormalization.wrappedValue = LayerNorm(dimensions: config.dimensions)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray,
        positionEncoding: MLXArray
    ) -> MLXArray {
        var residual = hiddenStates
        let queries = hiddenStates + positionEncoding
        let keys = hiddenStates + positionEncoding
        var hiddenStates = attention(
            queries: queries,
            keys: keys,
            values: hiddenStates,
            mask: attentionMask
        )

        hiddenStates = residual + hiddenStates
        hiddenStates = attentionNormalization(hiddenStates)
        residual = hiddenStates

        hiddenStates = feedforwardInput(hiddenStates)
        hiddenStates = activation(hiddenStates)
        hiddenStates = feedforwardOutput(hiddenStates)

        hiddenStates = residual + hiddenStates
        hiddenStates = finalNormalization(hiddenStates)

        return hiddenStates
    }
}

final class DetrDecoder: Module {

    let layers: [DetrDecoderLayer]

    @ModuleInfo(key: "layernorm") var normalization: LayerNorm

    init(_ config: DetrConfig) {
        layers = (0..<config.decoderLayers).map { _ in DetrDecoderLayer(config) }
        _normalization.wrappedValue = LayerNorm(dimensions: config.dimensions)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        objectQueries: MLXArray,
        encoderOutputs: MLXArray,
        encoderMask: MLXArray,
        encoderPositions: MLXArray
    ) -> MLXArray {
        normalization(
            layers.reduce(hiddenStates) { hiddenStates, layer in
                layer(
                    hiddenStates: hiddenStates,
                    objectQueries: objectQueries,
                    encoderOutputs: encoderOutputs,
                    encoderMask: encoderMask,
                    encoderPositions: encoderPositions
                )
            }
        )
    }
}

final class DetrDecoderLayer: Module {

    let activation: UnaryLayer

    @ModuleInfo(key: "self_attn") var attention: DetrAttention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNormalization: LayerNorm

    @ModuleInfo(key: "encoder_attn") var encoderAttention: DetrAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttentionNormalization: LayerNorm

    @ModuleInfo(key: "fc1") var feedforwardInput: Linear
    @ModuleInfo(key: "fc2") var feedforwardOutput: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNormalization: LayerNorm

    init(_ config: DetrConfig) {
        activation = ReLU()
        _attention.wrappedValue = DetrAttention(config)
        _attentionNormalization.wrappedValue = LayerNorm(dimensions: config.dimensions)
        _encoderAttention.wrappedValue = DetrAttention(config)
        _encoderAttentionNormalization.wrappedValue = LayerNorm(dimensions: config.dimensions)
        _feedforwardInput.wrappedValue = Linear(config.dimensions, config.decoderFeedforwardDimensions)
        _feedforwardOutput.wrappedValue = Linear(config.decoderFeedforwardDimensions, config.dimensions)
        _finalNormalization.wrappedValue = LayerNorm(dimensions: config.dimensions)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        objectQueries: MLXArray,
        encoderOutputs: MLXArray,
        encoderMask: MLXArray,
        encoderPositions: MLXArray
    ) -> MLXArray {
        var residual = hiddenStates
        let queries = hiddenStates + objectQueries
        let keys = hiddenStates + objectQueries
        var hiddenStates = attention(
            queries: queries,
            keys: keys,
            values: hiddenStates
        )

        hiddenStates = residual + hiddenStates
        hiddenStates = attentionNormalization(hiddenStates)
        residual = hiddenStates

        let queriesCross = hiddenStates + objectQueries
        let keysCross = encoderOutputs + encoderPositions
        hiddenStates = encoderAttention(
            queries: queriesCross,
            keys: keysCross,
            values: encoderOutputs,
            mask: encoderMask
        )

        hiddenStates = residual + hiddenStates
        hiddenStates = encoderAttentionNormalization(hiddenStates)
        residual = hiddenStates

        hiddenStates = feedforwardInput(hiddenStates)
        hiddenStates = activation(hiddenStates)
        hiddenStates = feedforwardOutput(hiddenStates)

        hiddenStates = residual + hiddenStates
        hiddenStates = finalNormalization(hiddenStates)

        return hiddenStates
    }
}

final class DetrAttention: Module {

    let numHeads: Int

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    init(_ config: DetrConfig) {
        numHeads = config.decoderAttentionHeads
        _queryProjection.wrappedValue = Linear(config.dimensions, config.dimensions)
        _keyProjection.wrappedValue = Linear(config.dimensions, config.dimensions)
        _valueProjection.wrappedValue = Linear(config.dimensions, config.dimensions)
        _outProjection.wrappedValue = Linear(config.dimensions, config.dimensions)
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

protocol DetrPositionEmbedding {
    func callAsFunction(_ pixelValues: MLXArray, _ pixelMask: MLXArray) -> MLXArray
}

final class DetrSinePositionEmbedding: Module, DetrPositionEmbedding {

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

    func callAsFunction(_ pixelValues: MLXArray, _ pixelMask: MLXArray) -> MLXArray {
        var yEmbeddings = pixelMask.cumsum(axis: 1).asType(pixelValues.dtype)
        var xEmbeddings = pixelMask.cumsum(axis: 2).asType(pixelValues.dtype)
        if normalize {
            yEmbeddings = yEmbeddings / (yEmbeddings[0..., (yEmbeddings.shape[1] - 1)..., 0...] + 1e-6) * scale
            xEmbeddings = xEmbeddings / (xEmbeddings[0..., 0..., (xEmbeddings.shape[2] - 1)...] + 1e-6) * scale
        }

        var dimT = Array(0..<embeddingDimensions).asMLXArray(dtype: pixelValues.dtype)
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

final class DetrLearnedPositionEmbedding: Module, DetrPositionEmbedding {

    @ModuleInfo(key: "row_embeddings") var rowEmbedding: Embedding
    @ModuleInfo(key: "column_embeddings") var columnEmbedding: Embedding

    let maxSize: Int

    init(embeddingDimensions: Int, maxSize: Int) {
        _rowEmbedding.wrappedValue = Embedding(embeddingCount: maxSize, dimensions: embeddingDimensions)
        _columnEmbedding.wrappedValue = Embedding(embeddingCount: maxSize, dimensions: embeddingDimensions)
        self.maxSize = maxSize
    }

    func callAsFunction(_ pixelValues: MLXArray, _ pixelMask: MLXArray) -> MLXArray {
        let (B, H, W) = pixelMask.shape3
        let rowIndices = MLXArray(0..<H)
        let colIndices = MLXArray(0..<W)

        var yEmbeddings = rowEmbedding(rowIndices).expandedDimensions(axis: 1)
        yEmbeddings = MLX.repeated(yEmbeddings, count: W, axis: 1)

        var xEmbeddings = columnEmbedding(colIndices).expandedDimensions(axis: 0)
        xEmbeddings = MLX.repeated(xEmbeddings, count: H, axis: 0)

        var positionEmbeddings = MLX.concatenated([yEmbeddings, xEmbeddings], axis: 2)
        positionEmbeddings = positionEmbeddings.expandedDimensions(axis: 0)
        positionEmbeddings = MLX.repeated(positionEmbeddings, count: B, axis: 0)

        return positionEmbeddings
    }
}

final class DetrPredictionHead: Module, UnaryLayer {

    let layers: [UnaryLayer]
    let relu = ReLU()

    init(
        inputDimensions: Int,
        hiddenDimensions: Int,
        outputDimensions: Int = 4,
        layersCount: Int = 3
    ) {
        layers = zip(
            [inputDimensions] + (1..<layersCount).map({ _ in hiddenDimensions }),
            (1..<layersCount).map({ _ in hiddenDimensions }) + [outputDimensions]
        ).map {
            Linear($0.0, $0.1)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.last!(
            layers.dropLast().reduce(x) {
                relu($1($0))
            }
        )
    }
}

final class DetrMaskHead: Module {

    @ModuleInfo(key: "stem_layers") var stemLayers: [DetrMaskHeadStemLayer]
    @ModuleInfo(key: "fpn_layers") var fpnLayers: [DetrMaskHeadFPNLayer]
    @ModuleInfo(key: "out_lay") var outProjection: Conv2d

    init(_ config: DetrConfig) {
        let dim = config.dimensions + config.decoderAttentionHeads
        let contextDim = config.dimensions
        let hiddenSizes = ResNetConfig.all[config.backbone]!.hiddenSizes
        let fpnChannels = Array(hiddenSizes.reversed().suffix(3))

        _stemLayers.wrappedValue = [
            DetrMaskHeadStemLayer(inputChannels: dim, outputChannels: dim),
            DetrMaskHeadStemLayer(inputChannels: dim, outputChannels: contextDim / 2),
        ]

        _fpnLayers.wrappedValue = [
            DetrMaskHeadFPNLayer(fpnChannels: fpnChannels[0], inputChannels: contextDim / 2, outputChannels: contextDim / 4),
            DetrMaskHeadFPNLayer(fpnChannels: fpnChannels[1], inputChannels: contextDim / 4, outputChannels: contextDim / 8),
            DetrMaskHeadFPNLayer(fpnChannels: fpnChannels[2], inputChannels: contextDim / 8, outputChannels: contextDim / 16),
        ]

        _outProjection.wrappedValue = Conv2d(
            inputChannels: config.dimensions / 16,
            outputChannels: 1,
            kernelSize: 3,
            padding: 1
        )
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing(/lay([12]).([a-z_]+)/) {
                "stem_layers.\(Int($0.1)! - 1).convolution.\($0.2)"
            }
            .replacing(/gn([12]).([a-z_]+)/) {
                "stem_layers.\(Int($0.1)! - 1).normalization.\($0.2)"
            }
            .replacing(/lay([345]).([a-z_]+)/) {
                "fpn_layers.\(Int($0.1)! - 3).convolution.\($0.2)"
            }
            .replacing(/gn([345]).([a-z_]+)/) {
                "fpn_layers.\(Int($0.1)! - 3).normalization.\($0.2)"
            }
            .replacing(/adapter(\d+).([a-z_]+)/) {
                "fpn_layers.\(Int($0.1)! - 1).adapter.\($0.2)"
            }
        }
    }

    func callAsFunction(_ x: MLXArray, bboxAttention: MLXArray, fpns: [MLXArray]) -> MLXArray {
        var x = MLX.concatenated([MLX.repeated(x, count: bboxAttention.shape[1], axis: 0), bboxAttention.flattened(start: 0, end: 1)], axis: -1)
        x = stemLayers.reduce(x, { $1($0) })

        for (i, fpnLayer) in fpnLayers.enumerated() {
            x = fpnLayer(x, fpnFeature: fpns[i])
        }

        x = outProjection(x)

        return x
    }
}

final class DetrMaskHeadStemLayer: Module, UnaryLayer {

    let convolution: Conv2d
    let normalization: GroupNorm
    let activation: UnaryLayer

    init(
        inputChannels: Int,
        outputChannels: Int,
        groups: Int = 8,
        kernelSize: Int = 3,
        stride: Int = 1,
        padding: Int = 1
    ) {
        convolution = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding)
        )

        normalization = GroupNorm(
            groupCount: min(groups, outputChannels),
            dimensions: outputChannels,
            pytorchCompatible: true
        )

        activation = ReLU()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convolution(x)
        y = normalization(y)
        y = activation(y)
        return y
    }
}

final class DetrMaskHeadFPNLayer: Module {

    let adapter: Conv2d
    let convolution: Conv2d
    let normalization: GroupNorm
    let activation: UnaryLayer

    init(
        fpnChannels: Int,
        inputChannels: Int,
        outputChannels: Int,
        groups: Int = 8,
        kernelSize: Int = 3,
        stride: Int = 1,
        padding: Int = 1,
    ) {
        adapter = Conv2d(
            inputChannels: fpnChannels,
            outputChannels: inputChannels,
            kernelSize: 1
        )

        convolution = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding)
        )

        normalization = GroupNorm(
            groupCount: min(groups, outputChannels),
            dimensions: outputChannels,
            pytorchCompatible: true
        )

        activation = ReLU()
    }

    func callAsFunction(_ x: MLXArray, fpnFeature: MLXArray) -> MLXArray {
        var projected = adapter(fpnFeature)
        projected = MLX.repeated(projected, count: x.shape[0] / projected.shape[0], axis: 0)
        projected = projected + x.interpolate(size: projected.shape.dropFirst().dropLast())

        var y = convolution(projected)
        y = normalization(y)
        y = activation(y)

        return y
    }
}

final class DetrMaskHeadAttention: Module {

    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_linear") var queryProjection: Linear
    @ModuleInfo(key: "k_linear") var keyProjection: Linear

    init(_ config: DetrConfig) {
        numHeads = config.decoderAttentionHeads
        headDim = config.dimensions / config.decoderAttentionHeads
        scale = pow(Float(config.dimensions) / Float(config.decoderAttentionHeads), -0.5)
        _queryProjection.wrappedValue = Linear(config.dimensions, config.dimensions)
        _keyProjection.wrappedValue = Linear(config.dimensions, config.dimensions)
    }

    func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        mask: MLXArray?
    ) -> MLXArray {
        var queries = queryProjection(queries)
        var keys = keyProjection(keys)

        let (B, Q, _) = queries.shape3
        let (_, H, W, _) = keys.shape4

        queries = queries.reshaped(B, Q, numHeads, headDim)
        keys = keys.reshaped(B, H, W, numHeads, headDim).transposed(0, 3, 4, 1, 2)

        var weights = MLX.einsum("bqnc,bnchw->bqnhw", queries * scale, keys)
        if var mask {
            mask = mask.expandedDimensions(axes: [0, 1])
            weights = MLX.which(mask, weights, MLX.full(weights.shape, values: -Float.infinity, type: Float16.self))
        }

        weights = weights.reshaped([B, Q, numHeads * H * W])
        weights = weights.softmax(axis: 2)
        weights = weights.reshaped([B, Q, numHeads, H, W]).transposed(0, 1, 3, 4, 2)

        return weights
    }
}
