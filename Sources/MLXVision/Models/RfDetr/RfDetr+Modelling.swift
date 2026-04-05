//
//  RfDetr+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 03.03.2026.
//

import Foundation
import MLX
import MLXNN

final class RfDetrModelForObjectDetection: Module, Predictor {

    typealias Output = (
        logits: MLXArray,
        probs: MLXArray,
        boxes: MLXArray
    )

    @ModuleInfo var model: RfDetrModel
    @ModuleInfo(key: "class_embed") var classifier: Linear
    @ModuleInfo(key: "bbox_embed") var bboxPredictionHead: RfDetrMLPPredictionHead

    init(_ config: RfDetrForObjectDetectionConfig) {
        _model.wrappedValue = RfDetrModel(config, numClasses: config.id2label.count)
        _classifier.wrappedValue = Linear(config.dModel, config.id2label.count)
        _bboxPredictionHead.wrappedValue = RfDetrMLPPredictionHead(config)
    }

    private lazy var _predict = MLX.compile { [unowned self] inputs in
        let outputs = model(inputs[0])
        let logits = classifier(outputs.hiddenStates)
        let probs = logits.softmax(axis: -1)
        let boxesDelta = bboxPredictionHead(outputs.hiddenStates)
        let boxes = refineBoxes(outputs.referencePoints, boxesDelta)
        return [logits, probs, boxes]
    }

    func predict(_ input: ImageInput) throws -> Output {
        let outputs = _predict([input.pixelValues])
        return (outputs[0], outputs[1], outputs[2])
    }
}

// Detection works fine, but there seems to be an issue with the segmentation weights.
// Compared with the previously used checkpoints, segmentation returns many incorrect results.
// Waiting for this PR to be merged: https://github.com/huggingface/transformers/pull/36895
final class RfDetrModelForInstanceSegmentation: Module, Predictor {

    typealias Output = (
        probs: MLXArray,
        boxes: MLXArray,
        segmentationMask: MLXArray
    )

    @ModuleInfo var model: RfDetrModel
    @ModuleInfo(key: "class_embed") var classifier: Linear
    @ModuleInfo(key: "bbox_embed") var bboxPredictionHead: RfDetrMLPPredictionHead
    @ModuleInfo(key: "segmentation_head") var segmentationHead: RfDetrSegmentationHead

    private let decoderLayers: Int

    init(_ config: RfDetrForObjectDetectionConfig) {
        _model.wrappedValue = RfDetrModel(config, numClasses: config.id2label.count)
        _classifier.wrappedValue = Linear(config.dModel, config.id2label.count)
        _bboxPredictionHead.wrappedValue = RfDetrMLPPredictionHead(config)
        _segmentationHead.wrappedValue = RfDetrSegmentationHead(config)
        decoderLayers = config.decoderLayers
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0
                .replacing("model.model", with: "model")
                .replacing("model.class_embed", with: "class_embed")
                .replacing("model.bbox_embed", with: "bbox_embed")
                .replacing("spatial_features_proj", with: "segmentation_head.spatial_features_proj")
                .replacing("query_features_block", with: "segmentation_head.query_features_block")
                .replacing("query_features_proj", with: "segmentation_head.query_features_proj")
                .replacing("segmentation_bias", with: "segmentation_head.bias")
                .replacing("blocks", with: "segmentation_head.blocks")
        }
    }

    private lazy var _predict = MLX.compile { [unowned self] inputs in
        let pixelValues = inputs[0]
        let outputs = model(pixelValues)

        let logits = classifier(outputs.hiddenStates)
        let probs = logits.sigmoid()

        let boxesDelta = bboxPredictionHead(outputs.hiddenStates)
        let boxes = refineBoxes(outputs.referencePoints, boxesDelta)

        let queryFeatures = outputs.intermediateHiddenStates.split(parts: decoderLayers, axis: 0).map { $0.squeezed(axis: 0) }
        let masks = segmentationHead(
            spatialFeatures: outputs.spatialFeatures,
            queryFeatures: queryFeatures,
            imageSize: (pixelValues.shape[1], pixelValues.shape[2])
        )

        let targetSize = Array(pixelValues.shape.dropFirst().dropLast())
        let interpolatedMask = masks.last!.squeezed().expandedDimensions(axis: -1)
            .interpolate(size: targetSize, mode: .linear(alignCorners: false))
            .squeezed()

        return [probs, boxes, interpolatedMask]
    }

    func predict(_ input: ImageInput) throws -> Output {
        let outputs = _predict([input.pixelValues])
        return (outputs[0], outputs[1], outputs[2])
    }
}

final class RfDetrModel: LwDetrModel {

    init(_ config: RfDetrConfig, numClasses: Int) {
        super.init(
            numQueries: config.numQueries,
            groupDetr: config.groupDetr,
            dModel: config.dModel,
            numClasses: numClasses,
            backbone: RfDetrConvEncoder(config),
            decoder: RfDetrDecoder(config),
            bboxHeads: (0..<config.groupDetr).map { _ in
                RfDetrMLPPredictionHead(config)
            }
        )
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0
                .replacing("backbone.0.", with: "backbone.")
                .replacing("transformer.enc_output.", with: "enc_output.")
                .replacing("transformer.transformer.enc_output_norm.", with: "enc_output_norm.")
                .replacing("transformer.enc_output_norm.", with: "enc_output_norm.")
                .replacing("transformer.enc_out_class_embed.", with: "enc_out_class_embed.")
                .replacing("transformer.enc_out_bbox_embed.", with: "enc_out_bbox_embed.")
                .replacing("transformer.decoder.", with: "decoder.")
                .replacing("refpoint_embed.", with: "reference_point_embed.")
        }
    }
}

final class RfDetrConvEncoder: LwDetrConvEncoder {

    init(_ config: RfDetrConfig) {
        super.init(
            backbone: RfDetrDinov2Backbone(config.backboneConfig),
            projector: RfDetrScaleProjector(config)
        )
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("encoder.encoder.", with: "backbone.")
        }
    }
}

final class RfDetrDinov2Backbone: Module, LwDetrConvEncoder.Backbone {

    @ModuleInfo var embeddings: RfDetrDinov2Embeddings
    @ModuleInfo var encoder: RfDetrDinov2Encoder
    @ModuleInfo(key: "layernorm") var normalization: LayerNorm

    private let outIndices: Set<Int>
    private let patchSize: Int
    private let numWindows: Int
    private let numRegisterTokens: Int

    init(_ config: RfDetrDinov2Config) {
        _embeddings.wrappedValue = RfDetrDinov2Embeddings(config)
        _encoder.wrappedValue = RfDetrDinov2Encoder(config)
        _normalization.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        outIndices = Set(config.outIndices.isEmpty ? [config.numHiddenLayers] : config.outIndices)
        patchSize = config.patchSize
        numWindows = config.numWindows
        numRegisterTokens = config.numRegisterTokens
    }

    func callAsFunction(_ pixelValues: MLXArray) -> [MLXArray] {
        let (_, height, width, _) = pixelValues.shape4

        let embeddingOutput = embeddings(pixelValues)
        let hiddenStates = encoder(embeddingOutput)

        let numHPatches = max(1, height / patchSize)
        let numWPatches = max(1, width / patchSize)

        var featureMaps: [MLXArray] = []
        for (stage, hiddenState) in hiddenStates.enumerated() {
            guard outIndices.contains(stage) else {
                continue
            }

            var hidden = normalization(hiddenState)
            hidden = hidden[0..., (1 + numRegisterTokens)..., 0...]

            if numWindows > 1 {
                hidden = unpartitionWindows(
                    hidden,
                    numWindows: numWindows,
                    numHPatches: numHPatches,
                    numWPatches: numWPatches
                )
            }

            hidden = hidden.reshaped(pixelValues.dim(0), numHPatches, numWPatches, -1)
            featureMaps.append(hidden)
        }

        return featureMaps
    }

    private func unpartitionWindows(_ hiddenStates: MLXArray, numWindows: Int, numHPatches: Int, numWPatches: Int) -> MLXArray {
        let (hiddenBatchSize, sequenceLength, channels) = hiddenStates.shape3
        let numWindowsSquared = numWindows * numWindows

        let numHPatchesPerWindow = numHPatches / numWindows
        let numWPatchesPerWindow = numWPatches / numWindows

        let hidden =
            hiddenStates
            .reshaped(hiddenBatchSize / numWindowsSquared, numWindowsSquared * sequenceLength, channels)
            .reshaped(hiddenBatchSize / numWindowsSquared, numWindows, numWindows, numHPatchesPerWindow, numWPatchesPerWindow, channels)
            .transposed(0, 1, 3, 2, 4, 5)

        return hidden
    }
}

final class RfDetrDinov2Embeddings: Module {

    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: RfDetrDinov2PatchEmbeddings
    @ParameterInfo(key: "cls_token") var classToken: MLXArray
    @ParameterInfo(key: "mask_token") var maskToken: MLXArray
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    private let patchSize: Int
    private let numWindows: Int

    init(_ config: RfDetrDinov2Config) {
        _classToken.wrappedValue = MLX.zeros([1, 1, config.hiddenSize])
        _maskToken.wrappedValue = MLX.zeros([1, config.hiddenSize])
        _patchEmbeddings.wrappedValue = RfDetrDinov2PatchEmbeddings(config)

        let patchesPerSide = max(1, config.imageSize / config.patchSize)
        let numPatches = patchesPerSide * patchesPerSide
        _positionEmbeddings.wrappedValue = MLX.zeros([1, numPatches + 1, config.hiddenSize])

        patchSize = config.patchSize
        numWindows = config.numWindows
    }

    func callAsFunction(_ pixelValues: MLXArray, _ boolMaskedPos: MLXArray? = nil) -> MLXArray {
        let (batchSize, height, width, _) = pixelValues.shape4
        var embeddings = patchEmbeddings(pixelValues)

        if let boolMaskedPos {
            embeddings = MLX.where(
                boolMaskedPos.expandedDimensions(axis: -1),
                maskToken.expandedDimensions(axis: 0),
                embeddings
            )
        }

        let classTokens = MLX.repeated(classToken, count: batchSize, axis: 0)
        embeddings = MLX.concatenated([classTokens, embeddings], axis: 1)
        embeddings += interpolatePositionEmbeddings(
            embeddings,
            height: height,
            width: width
        )

        if numWindows > 1 {
            embeddings = partitionWindows(
                embeddings,
                numWindows: numWindows,
                patchSize: patchSize,
                height: height,
                width: width
            )
        }

        return embeddings
    }

    private func interpolatePositionEmbeddings(_ embeddings: MLXArray, height: Int, width: Int) -> MLXArray {
        let numPatches = embeddings.shape[1] - 1
        let numPositions = positionEmbeddings.shape[1] - 1
        if numPatches == numPositions, height == width {
            return positionEmbeddings
        }

        let classPos = positionEmbeddings[0..., 0..<1, 0...]
        let dim = embeddings.shape[2]
        let newHeight = max(1, height / patchSize)
        let newWidth = max(1, width / patchSize)
        let sqrtNumPositions = Int(Double(numPositions).squareRoot())

        let patchPos = positionEmbeddings[0..., 1..., 0...]
            .reshaped(1, sqrtNumPositions, sqrtNumPositions, dim)
            .interpolate(size: [newHeight, newWidth], mode: .linear(alignCorners: false))
            .reshaped(1, newHeight * newWidth, dim)

        return MLX.concatenated([classPos, patchPos], axis: 1)
    }

    private func partitionWindows(
        _ embeddings: MLXArray,
        numWindows: Int,
        patchSize: Int,
        height: Int,
        width: Int
    ) -> MLXArray {
        let batchSize = embeddings.shape[0]
        let numHPatches = height / patchSize
        let numWPatches = width / patchSize
        let numWPatchesPerWindow = numWPatches / numWindows
        let numHPatchesPerWindow = numHPatches / numWindows

        let classToken = MLX.repeated(
            embeddings[0..., 0..<1, 0...],
            count: numWindows * numWindows,
            axis: 0
        )

        let pixelTokens = embeddings[0..., 1..., 0...]
            .reshaped(batchSize, numHPatches, numWPatches, -1)
            .reshaped(batchSize, numWindows, numWPatchesPerWindow, numWindows, numHPatchesPerWindow, -1)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batchSize * numWindows * numWindows, numHPatchesPerWindow * numWPatchesPerWindow, -1)

        return MLX.concatenated([classToken, pixelTokens], axis: 1)
    }
}

final class RfDetrDinov2PatchEmbeddings: Module {

    @ModuleInfo var projection: Conv2d

    init(_ config: RfDetrDinov2Config) {
        _projection.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize)
        )
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let projected = projection(pixelValues)
        return projected.flattened(start: 1, end: 2)
    }
}

final class RfDetrDinov2Encoder: Module {

    @ModuleInfo(key: "layer") var layers: [RfDetrDinov2Layer]

    init(_ config: RfDetrDinov2Config) {
        let outputStages = Set(config.outIndices)
        let maxStage = max(config.numHiddenLayers, outputStages.max() ?? config.numHiddenLayers)

        let windowAttentionLayers: Set<Int>
        if config.windowBlockIndices.isEmpty {
            var all = Set(0...maxStage)
            for stage in outputStages {
                all.remove(stage)
            }
            windowAttentionLayers = all
        } else {
            windowAttentionLayers = Set(config.windowBlockIndices)
        }

        _layers.wrappedValue = (0..<config.numHiddenLayers).map { index in
            RfDetrDinov2Layer(config, layerIndex: index, windowAttentionLayers: windowAttentionLayers)
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> [MLXArray] {
        var states: [MLXArray] = [hiddenStates]
        var hidden = hiddenStates

        for layer in layers {
            hidden = layer(hidden)
            states.append(hidden)
        }

        return states
    }
}

final class RfDetrDinov2Layer: Module {

    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attention: RfDetrDinov2Attention
    @ModuleInfo(key: "layer_scale1") var layerScale1: RfDetrDinov2LayerScale

    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: RfDetrDinov2MLP
    @ModuleInfo(key: "layer_scale2") var layerScale2: RfDetrDinov2LayerScale

    private let numWindows: Int
    private let usesGlobalAttention: Bool

    init(_ config: RfDetrDinov2Config, layerIndex: Int, windowAttentionLayers: Set<Int>) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attention.wrappedValue = RfDetrDinov2Attention(config)
        _layerScale1.wrappedValue = RfDetrDinov2LayerScale(config)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = RfDetrDinov2MLP(config)
        _layerScale2.wrappedValue = RfDetrDinov2LayerScale(config)
        numWindows = config.numWindows
        usesGlobalAttention = !windowAttentionLayers.contains(layerIndex)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let shortcut = hiddenStates

        var hidden = hiddenStates
        if usesGlobalAttention {
            hidden = mergeWindowTokens(hidden, numWindows: numWindows)
        }

        let selfAttentionOutput = attention(norm1(hidden))
        var attentionOutput = selfAttentionOutput

        if usesGlobalAttention {
            attentionOutput = splitWindowTokens(
                hidden,
                selfAttentionOutput,
                numWindows: numWindows
            )
        }

        attentionOutput = layerScale1(attentionOutput)
        var output = shortcut + attentionOutput

        var layerOutput = norm2(output)
        layerOutput = mlp(layerOutput)
        layerOutput = layerScale2(layerOutput)

        output = output + layerOutput
        return output
    }

    private func mergeWindowTokens(_ hiddenStates: MLXArray, numWindows: Int) -> MLXArray {
        let (batchSize, sequenceLength, channels) = hiddenStates.shape3
        let numWindowsSquared = numWindows * numWindows
        return hiddenStates.reshaped(batchSize / numWindowsSquared, numWindowsSquared * sequenceLength, channels)
    }

    private func splitWindowTokens(_ hiddenStates: MLXArray, _ selfAttentionOutput: MLXArray, numWindows: Int) -> MLXArray {
        let (batchSize, sequenceLength, channels) = hiddenStates.shape3
        let numWindowsSquared = numWindows * numWindows
        return selfAttentionOutput.reshaped(batchSize * numWindowsSquared, sequenceLength / numWindowsSquared, channels)
    }
}

final class RfDetrDinov2LayerScale: Module, UnaryLayer {

    @ParameterInfo(key: "lambda1") var scale: MLXArray

    init(_ config: RfDetrDinov2Config) {
        _scale.wrappedValue = MLX.ones([config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * scale
    }
}

final class RfDetrDinov2Attention: Module, UnaryLayer {

    @ModuleInfo var attention: RfDetrDinov2SelfAttention
    @ModuleInfo var output: RfDetrDinov2SelfOutput

    init(_ config: RfDetrDinov2Config) {
        _attention.wrappedValue = RfDetrDinov2SelfAttention(config)
        _output.wrappedValue = RfDetrDinov2SelfOutput(config)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        output(attention(hiddenStates))
    }
}

final class RfDetrDinov2SelfAttention: Module, UnaryLayer {

    let numHeads: Int

    @ModuleInfo(key: "query") var queryProjection: Linear
    @ModuleInfo(key: "key") var keyProjection: Linear
    @ModuleInfo(key: "value") var valueProjection: Linear

    init(_ config: RfDetrDinov2Config) {
        numHeads = config.numAttentionHeads
        _queryProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var queries = queryProjection(hiddenStates)
        var keys = keyProjection(hiddenStates)
        var values = valueProjection(hiddenStates)

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

        return output
    }
}

final class RfDetrDinov2SelfOutput: Module {

    @ModuleInfo(key: "dense") var projection: Linear

    init(_ config: RfDetrDinov2Config) {
        _projection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        projection(hiddenStates)
    }
}

final class RfDetrDinov2MLP: Module, UnaryLayer {

    @ModuleInfo(key: "fc1") var expansionProjection: Linear
    @ModuleInfo(key: "fc2") var outputProjection: Linear

    private let activation: UnaryLayer

    init(_ config: RfDetrDinov2Config) {
        _expansionProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * config.mlpRatio)
        _outputProjection.wrappedValue = Linear(config.hiddenSize * config.mlpRatio, config.hiddenSize)
        activation = GELU(approximation: .none)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputProjection(activation(expansionProjection(x)))
    }
}

final class RfDetrScaleProjector: LwDetrScaleProjector {

    init(_ config: RfDetrConfig) {
        super.init(scaleLayers: [RfDetrScaleProjectorStage(config)])
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0
                .replacing("stages.0.0.", with: "scale_layers.0.projector_layer.")
                .replacing("stages.0.1.", with: "scale_layers.0.layer_norm.")
        }
    }
}

final class RfDetrScaleProjectorStage: LwDetrScaleProjectorStage {
    init(_ config: RfDetrConfig) {
        let projectorInputDim = config.backboneConfig.hiddenSize * max(1, config.backboneConfig.outIndices.count)
        super.init(
            projectorLayer: RfDetrC2FLayer(config, inputChannels: projectorInputDim),
            outputChannels: config.dModel,
            layerNormEps: config.layerNormEps
        )
    }
}

final class RfDetrC2FLayer: LwDetrC2FLayer {

    init(_ config: RfDetrConfig, inputChannels: Int) {
        let hiddenChannels = Int(Float(config.dModel) * config.hiddenExpansion)
        super.init(
            inputProjection: LwDetrConvNormLayer(
                inputChannels: inputChannels,
                outputChannels: hiddenChannels * 2,
                kernelSize: 1,
                stride: 1,
                normalization: LayerNorm(dimensions: hiddenChannels * 2, eps: config.layerNormEps),
                activation: config.activationFunction.layer
            ),
            outputProjection: LwDetrConvNormLayer(
                inputChannels: (2 + config.c2fNumBlocks) * hiddenChannels,
                outputChannels: config.dModel,
                kernelSize: 1,
                stride: 1,
                normalization: LayerNorm(dimensions: config.dModel, eps: config.layerNormEps),
                activation: config.activationFunction.layer
            ),
            bottlenecks: (0..<config.c2fNumBlocks).map { _ in
                LwDetrRepVggBlock(
                    channels: hiddenChannels,
                    normalization: { LayerNorm(dimensions: hiddenChannels, eps: config.layerNormEps) },
                    activation: { config.activationFunction.layer }
                )
            }
        )
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0
                .replacing("m.", with: "bottlenecks.")
                .replacing("cv1.", with: "conv1.")
                .replacing("cv2.", with: "conv2.")
                .replacing("bn.", with: "norm.")
        }
    }
}

final class RfDetrDepthwiseConvBlock: Module, UnaryLayer {

    @ModuleInfo(key: "depthwise_conv") var depthwiseConvolution: Conv2d
    @ModuleInfo(key: "layernorm") var normalization: LayerNorm
    @ModuleInfo(key: "pointwise_conv") var pointwiseProjection: Linear

    private let activation = GELU(approximation: .none)

    init(dimensions: Int) {
        _depthwiseConvolution.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions, kernelSize: 3, padding: 1, groups: dimensions)
        _normalization.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        _pointwiseProjection.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        hiddenStates + activation(pointwiseProjection(normalization(depthwiseConvolution(hiddenStates))))
    }
}

final class RfDetrSegmentationMLPBlock: Module, UnaryLayer {

    @ModuleInfo(key: "norm") var inputNormalization: LayerNorm
    @ModuleInfo(key: "fc1") var expansionProjection: Linear
    @ModuleInfo(key: "fc2") var outputProjection: Linear

    private let activation = GELU(approximation: .none)

    init(dimensions: Int) {
        _inputNormalization.wrappedValue = LayerNorm(dimensions: dimensions)
        _expansionProjection.wrappedValue = Linear(dimensions, dimensions * 4)
        _outputProjection.wrappedValue = Linear(dimensions * 4, dimensions)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0
                .replacing("mlp.fc1.", with: "fc1.")
                .replacing("mlp.fc2.", with: "fc2.")
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        hiddenStates + outputProjection(activation(expansionProjection(inputNormalization(hiddenStates))))
    }
}

final class RfDetrSegmentationHead: Module {

    @ModuleInfo var blocks: [RfDetrDepthwiseConvBlock]
    @ModuleInfo(key: "spatial_features_proj") var spatialFeaturesProjection: Conv2d
    @ModuleInfo(key: "query_features_block") var queryFeaturesBlock: RfDetrSegmentationMLPBlock
    @ModuleInfo(key: "query_features_proj") var queryFeaturesProjection: Linear
    @ParameterInfo var bias: MLXArray

    private let downsampleRatio: Int

    init(_ config: RfDetrConfig) {
        let bottleneckRatio = max(1, config.segmentationBottleneckRatio)
        let interactionDimensions = max(1, config.dModel / bottleneckRatio)
        _blocks.wrappedValue = (0..<config.decoderLayers).map { _ in RfDetrDepthwiseConvBlock(dimensions: config.dModel) }
        _spatialFeaturesProjection.wrappedValue = Conv2d(inputChannels: config.dModel, outputChannels: interactionDimensions, kernelSize: 1)
        _queryFeaturesBlock.wrappedValue = RfDetrSegmentationMLPBlock(dimensions: config.dModel)
        _queryFeaturesProjection.wrappedValue = Linear(config.dModel, interactionDimensions)
        _bias.wrappedValue = MLX.zeros([1])
        downsampleRatio = max(1, config.maskDownsampleRatio)
    }

    func callAsFunction(
        spatialFeatures: MLXArray,
        queryFeatures: [MLXArray],
        imageSize: (Int, Int)
    ) -> [MLXArray] {
        let targetHeight = max(1, imageSize.0 / downsampleRatio)
        let targetWidth = max(1, imageSize.1 / downsampleRatio)

        var spatial = spatialFeatures.interpolate(
            size: [targetHeight, targetWidth],
            mode: .linear(alignCorners: false)
        )

        var maskLogits: [MLXArray] = []
        for (block, queryFeature) in zip(blocks, queryFeatures) {
            spatial = block(spatial)
            let projectedSpatialFeatures = spatialFeaturesProjection(spatial)
            let projectedQueryFeatures = queryFeaturesProjection(queryFeaturesBlock(queryFeature))
            let mask = MLX.einsum("bhwc,bqc->bqhw", projectedSpatialFeatures, projectedQueryFeatures) + bias
            maskLogits.append(mask)
        }

        return maskLogits
    }
}

final class RfDetrDecoder: LwDetrDecoder {

    init(_ config: RfDetrConfig) {
        super.init(
            dModel: config.dModel,
            layers: (0..<config.decoderLayers).map { _ in RfDetrDecoderLayer(config) },
            refPointHead: RfDetrMLPPredictionHead(
                inputDimensions: 2 * config.dModel,
                hiddenDimensions: config.dModel,
                outputDimensions: config.dModel,
                layersCount: 2
            )
        )
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("norm.", with: "layernorm.")
        }
    }
}

final class RfDetrDecoderLayer: LwDetrDecoderLayer {

    init(_ config: RfDetrConfig) {
        super.init(
            dModel: config.dModel,
            selfAttention: RfDetrDecoderSelfAttention(config),
            crossAttention: RfDetrMultiscaleDeformableAttention(config),
            mlp: RfDetrDecoderMLP(config)
        )
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0
                .replacing("linear1.", with: "mlp.fc1.")
                .replacing("linear2.", with: "mlp.fc2.")
                .replacing("norm1.", with: "self_attn_layer_norm.")
                .replacing("norm2.", with: "cross_attn_layer_norm.")
                .replacing("norm3.", with: "layer_norm.")
                .replacing("out_proj.", with: "o_proj.")
        }
    }
}

final class RfDetrDecoderSelfAttention: LwDetrDecoderSelfAttention {

    init(_ config: RfDetrConfig) {
        super.init(
            dModel: config.dModel,
            numHeads: config.decoderSelfAttentionHeads,
            attentionBias: config.attentionBias
        )
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        .unflattened(
            parameters.flattened().flatMap { key, value in
                switch key {
                case "in_proj_weight":
                    zip(["q", "k", "v"], value.split(parts: 3, axis: 0)).map {
                        ("\($0)_proj.weight", $1)
                    }
                case "in_proj_bias":
                    zip(["q", "k", "v"], value.split(parts: 3, axis: 0)).map {
                        ("\($0)_proj.bias", $1)
                    }
                default:
                    [(key, value)]
                }
            }
        )
    }
}

final class RfDetrMultiscaleDeformableAttention: LwDetrMultiscaleDeformableAttention {
    init(_ config: RfDetrConfig) {
        super.init(
            dModel: config.dModel,
            nHeads: config.decoderCrossAttentionHeads,
            nPoints: config.decoderNPoints,
            nLevels: config.numFeatureLevels,
            attentionBias: config.attentionBias
        )
    }
}

final class RfDetrDecoderMLP: LwDetrDecoderMLP {
    init(_ config: RfDetrConfig) {
        super.init(
            dModel: config.dModel,
            decoderFfnDim: config.decoderFfnDim,
            activation: config.decoderActivationFunction.layer
        )
    }
}

final class RfDetrMLPPredictionHead: LwDetrMLPPredictionHead {
    convenience init(_ config: RfDetrConfig) {
        self.init(
            inputDimensions: config.dModel,
            hiddenDimensions: config.dModel,
            outputDimensions: 4,
            layersCount: 3
        )
    }
}
