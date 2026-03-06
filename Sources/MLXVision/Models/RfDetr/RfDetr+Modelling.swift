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
    @ModuleInfo(key: "class_embed") var classEmbed: Linear
    @ModuleInfo(key: "bbox_embed") var bboxEmbed: RfDetrMLPPredictionHead

    init(_ config: RfDetrForObjectDetectionConfig) {
        _model.wrappedValue = RfDetrModel(config, numClasses: config.id2label.count)
        _classEmbed.wrappedValue = Linear(config.dModel, config.id2label.count)
        _bboxEmbed.wrappedValue = RfDetrMLPPredictionHead(
            inputDimensions: config.dModel,
            hiddenDimensions: config.dModel,
            outputDimensions: 4,
            layersCount: 3
        )
    }

    private lazy var _predict = MLX.compile { [unowned self] inputs in
        let (hiddenStates, referencePoints) = model(inputs[0])
        let logits = classEmbed(hiddenStates)
        let probs = logits.softmax(axis: -1)
        let boxesDelta = bboxEmbed(hiddenStates)
        var boxes = refineBoxes(referencePoints, boxesDelta)
        return [logits, probs, boxes]
    }

    func predict(_ input: ImageInput) throws -> Output {
        let outputs = _predict([input.pixelValues])
        return (outputs[0], outputs[1], outputs[2])
    }
}

final class RfDetrModel: Module {

    typealias Output = (
        hiddenStates: MLXArray,
        referencePoints: MLXArray
    )

    private let numQueries: Int

    @ModuleInfo(key: "backbone") var backbone: RfDetrConvEncoder
    @ModuleInfo(key: "decoder") var decoder: RfDetrDecoder
    @ModuleInfo(key: "enc_output") var encOutput: [Linear]
    @ModuleInfo(key: "enc_output_norm") var encOutputNorm: [LayerNorm]
    @ModuleInfo(key: "enc_out_class_embed") var encOutClassEmbed: [Linear]
    @ModuleInfo(key: "enc_out_bbox_embed") var encOutBBoxEmbed: [RfDetrMLPPredictionHead]
    @ModuleInfo(key: "query_feat") var queryFeat: Embedding
    @ModuleInfo(key: "reference_point_embed") var referencePointEmbed: Embedding
    @ParameterInfo(key: "_query_indices") var queryIndices: MLXArray

    init(_ config: RfDetrConfig, numClasses: Int) {
        numQueries = config.numQueries
        _backbone.wrappedValue = RfDetrConvEncoder(config)
        _decoder.wrappedValue = RfDetrDecoder(config)
        _encOutput.wrappedValue = (0..<config.groupDetr).map { _ in
            Linear(config.dModel, config.dModel)
        }
        _encOutputNorm.wrappedValue = (0..<config.groupDetr).map { _ in
            LayerNorm(dimensions: config.dModel)
        }
        _encOutClassEmbed.wrappedValue = (0..<config.groupDetr).map { _ in
            Linear(config.dModel, numClasses)
        }
        _encOutBBoxEmbed.wrappedValue = (0..<config.groupDetr).map { _ in
            RfDetrMLPPredictionHead(
                inputDimensions: config.dModel,
                hiddenDimensions: config.dModel,
                outputDimensions: 4,
                layersCount: 3
            )
        }
        _queryFeat.wrappedValue = Embedding(
            embeddingCount: config.numQueries * config.groupDetr,
            dimensions: config.dModel
        )
        _referencePointEmbed.wrappedValue = Embedding(
            embeddingCount: config.numQueries * config.groupDetr,
            dimensions: 4
        )
        _queryIndices.wrappedValue = Array(0..<config.numQueries).asMLXArray(dtype: .int32)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> Output {
        let (projectedFeatures, featureMask) = backbone(pixelValues)

        let (batchSize, height, width, _) = projectedFeatures.shape4
        let sourceFlatten = projectedFeatures.flattened(start: 1, end: 2)
        let maskFlatten = featureMask.flattened(start: 1, end: 2)

        let (objectQueryEmbedding, outputProposals) = generateEncoderOutputProposals(
            encoderOutput: sourceFlatten,
            paddingMask: .!(maskFlatten.asType(DType.bool)),
            spatialShape: (height, width),
        )

        let topKCoords = generateTopKCoords(
            objectQueryEmbedding: objectQueryEmbedding,
            outputProposals: outputProposals,
            topK: numQueries
        )

        var referencePoints = referencePointEmbed(queryIndices).expandedDimensions(axis: 0)
        referencePoints = MLX.repeated(referencePoints, count: batchSize, axis: 0)

        let refinedReferencePoints = refineBoxes(topKCoords, referencePoints[0..., 0..<topKCoords.shape[1], 0...])
        referencePoints = MLX.concatenated([refinedReferencePoints, referencePoints[0..., topKCoords.shape[1]..., 0...]], axis: 1)

        var target = queryFeat(queryIndices).expandedDimensions(axis: 0)
        target = MLX.repeated(target, count: batchSize, axis: 0)

        let validRatios = MLXArray.ones([batchSize, 1, 2], dtype: sourceFlatten.dtype)
        let hiddenStates = decoder(
            inputsEmbeds: target,
            referencePoints: referencePoints,
            validRatios: validRatios,
            spatialShape: (height, width),
            encoderHiddenStates: sourceFlatten
        )

        return (hiddenStates, referencePoints)
    }

    private func generateTopKCoords(
        objectQueryEmbedding: MLXArray,
        outputProposals: MLXArray,
        topK: Int
    ) -> MLXArray {
        func sortedTopKIndices(_ scores: MLXArray, k: Int) -> MLXArray {
            let topK = max(1, min(k, scores.shape[1]))
            var indices = MLX.argPartition(-scores, kth: topK - 1, axis: 1)[0..., 0..<topK]
            let topKScores = MLX.takeAlong(scores, indices, axis: 1)
            let order = MLX.argSort(-topKScores, axis: 1)
            indices = MLX.takeAlong(indices, order, axis: 1)
            return indices
        }

        let objectQuery = encOutputNorm[0](encOutput[0](objectQueryEmbedding))
        let classLogits = encOutClassEmbed[0](objectQuery)
        let bboxDeltas = encOutBBoxEmbed[0](objectQuery)

        let refinedCoords = refineBoxes(outputProposals, bboxDeltas)

        let maxScores = classLogits.max(axis: -1)
        let topKIndices = sortedTopKIndices(maxScores, k: topK)

        let idx4 = MLX.repeated(topKIndices.expandedDimensions(axis: -1), count: 4, axis: -1)
        return MLX.takeAlong(refinedCoords, idx4, axis: 1)
    }

    private func generateEncoderOutputProposals(
        encoderOutput: MLXArray,
        paddingMask: MLXArray,
        spatialShape: (Int, Int)
    ) -> (MLXArray, MLXArray) {
        let (batchSize, _, _) = encoderOutput.shape3
        let (height, width) = spatialShape

        let gridY = MLXArray(0..<height).asType(encoderOutput.dtype)
        let gridX = MLXArray(0..<width).asType(encoderOutput.dtype)
        let meshGrid = MLX.meshGrid([gridY, gridX], indexing: .ij)
        let meshY = meshGrid[0]
        let meshX = meshGrid[1]

        var grid = MLX.stacked([meshX.flattened(), meshY.flattened()], axis: -1)
        grid = (grid + 0.5) / [Float(width), Float(height)].asMLXArray(dtype: encoderOutput.dtype)

        let widthHeight = MLX.ones(like: grid) * 0.05
        var proposals = MLX.concatenated([grid, widthHeight], axis: -1).expandedDimensions(axis: 0)
        proposals = MLX.repeated(proposals, count: batchSize, axis: 0)

        let infiniteProposals = MLX.full(proposals.shape, values: Float.infinity, type: Float.self)
        let proposalsValid = ((proposals .> 0.01) .&& (proposals .< 0.99)).min(axis: -1, keepDims: true)

        proposals = MLX.which(paddingMask.expandedDimensions(axis: -1), infiniteProposals, proposals)
        proposals = MLX.which(proposalsValid, proposals, infiniteProposals)

        var objectQuery = encoderOutput
        objectQuery = MLX.which(paddingMask.expandedDimensions(axis: -1), MLX.zeros(like: objectQuery), objectQuery)
        objectQuery = MLX.which(proposalsValid, objectQuery, MLX.zeros(like: objectQuery))

        return (objectQuery, proposals)
    }
}

final class RfDetrConvEncoder: Module {

    @ModuleInfo(key: "backbone") var backbone: RfDetrDinov2Backbone
    @ModuleInfo(key: "projector") var projector: RfDetrScaleProjector

    init(_ config: RfDetrConfig) {
        _backbone.wrappedValue = RfDetrDinov2Backbone(config.backboneConfig)
        _projector.wrappedValue = RfDetrScaleProjector(config)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> (MLXArray, MLXArray) {
        let features = backbone(pixelValues)
        let projected = projector(features)
        let (batchSize, height, width, _) = projected.shape4
        let mask = MLXArray.ones([batchSize, height, width], dtype: .bool)
        return (projected, mask)
    }
}

final class RfDetrDinov2Backbone: Module {

    @ModuleInfo(key: "embeddings") var embeddings: RfDetrDinov2Embeddings
    @ModuleInfo(key: "encoder") var encoder: RfDetrDinov2Encoder
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm

    private let outIndices: Set<Int>
    private let patchSize: Int
    private let numWindows: Int
    private let numRegisterTokens: Int

    init(_ config: RfDetrDinov2Config) {
        _embeddings.wrappedValue = RfDetrDinov2Embeddings(config)
        _encoder.wrappedValue = RfDetrDinov2Encoder(config)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
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

            var hidden = layerNorm(hiddenState)
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

    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "register_tokens") var registerTokens: MLXArray
    @ParameterInfo(key: "mask_token") var maskToken: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: RfDetrDinov2PatchEmbeddings
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    private let patchSize: Int
    private let numWindows: Int

    init(_ config: RfDetrDinov2Config) {
        _clsToken.wrappedValue = MLX.zeros([1, 1, config.hiddenSize])
        _registerTokens.wrappedValue = MLX.zeros([1, config.numRegisterTokens, config.hiddenSize])
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

        let clsTokens = MLX.repeated(clsToken, count: batchSize, axis: 0)
        embeddings = MLX.concatenated([clsTokens, embeddings], axis: 1)
        embeddings += interpolatePositionEmbeddings(
            embeddings,
            height: height,
            width: width
        )

        if registerTokens.shape[1] > 0 {
            let batchRegisterTokens = MLX.repeated(registerTokens, count: batchSize, axis: 0)
            embeddings = MLX.concatenated(
                [
                    embeddings[0..., 0..<1, 0...],
                    batchRegisterTokens,
                    embeddings[0..., 1..., 0...],
                ],
                axis: 1
            )
        }

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

        let clsToken = MLX.repeated(
            embeddings[0..., 0..<1, 0...],
            count: numWindows * numWindows,
            axis: 0
        )

        let pixelTokens = embeddings[0..., 1..., 0...]
            .reshaped(batchSize, numHPatches, numWPatches, -1)
            .reshaped(batchSize, numWindows, numWPatchesPerWindow, numWindows, numHPatchesPerWindow, -1)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batchSize * numWindows * numWindows, numHPatchesPerWindow * numWPatchesPerWindow, -1)

        return MLX.concatenated([clsToken, pixelTokens], axis: 1)
    }
}

final class RfDetrDinov2PatchEmbeddings: Module {

    @ModuleInfo(key: "projection") var projection: Conv2d

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
        let selectedStages = Set(config.outIndices)
        let maxStage = max(config.numHiddenLayers, selectedStages.max() ?? config.numHiddenLayers)

        let windowBlocks: Set<Int>
        if config.windowBlockIndexes.isEmpty {
            var all = Set(0...maxStage)
            for stage in selectedStages {
                all.remove(stage)
            }
            windowBlocks = all
        } else {
            windowBlocks = Set(config.windowBlockIndexes)
        }

        _layers.wrappedValue = (0..<config.numHiddenLayers).map { index in
            RfDetrDinov2Layer(config, layerIndex: index, windowBlocks: windowBlocks)
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

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: RfDetrDinov2Attention
    @ModuleInfo(key: "layer_scale1") var layerScale1: RfDetrDinov2LayerScale

    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: RfDetrDinov2MLP
    @ModuleInfo(key: "layer_scale2") var layerScale2: RfDetrDinov2LayerScale

    private let numWindows: Int
    private let globalAttention: Bool

    init(_ config: RfDetrDinov2Config, layerIndex: Int, windowBlocks: Set<Int>) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attention.wrappedValue = RfDetrDinov2Attention(config)
        _layerScale1.wrappedValue = RfDetrDinov2LayerScale(config)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = RfDetrDinov2MLP(config)
        _layerScale2.wrappedValue = RfDetrDinov2LayerScale(config)
        numWindows = config.numWindows
        globalAttention = !windowBlocks.contains(layerIndex)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let shortcut = hiddenStates

        var hidden = hiddenStates
        if globalAttention {
            hidden = mergeWindowTokens(hidden, numWindows: numWindows)
        }

        let selfAttentionOutput = attention(norm1(hidden))
        var attentionOutput = selfAttentionOutput

        if globalAttention {
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

    @ParameterInfo(key: "lambda1") var lambda1: MLXArray

    init(_ config: RfDetrDinov2Config) {
        _lambda1.wrappedValue = MLX.ones([config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * lambda1
    }
}

final class RfDetrDinov2Attention: Module, UnaryLayer {

    @ModuleInfo(key: "attention") var attention: RfDetrDinov2SelfAttention
    @ModuleInfo(key: "output") var output: RfDetrDinov2SelfOutput

    init(_ config: RfDetrDinov2Config) {
        _attention.wrappedValue = RfDetrDinov2SelfAttention(config)
        _output.wrappedValue = RfDetrDinov2SelfOutput(config)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        output(attention(hiddenStates), hiddenStates)
    }
}

final class RfDetrDinov2SelfAttention: Module, UnaryLayer {

    private let numHeads: Int

    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    init(_ config: RfDetrDinov2Config) {
        numHeads = config.numAttentionHeads
        _query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
        _value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batchSize, seqLength, _) = hiddenStates.shape3

        let queries = query(hiddenStates)
            .reshaped(batchSize, seqLength, numHeads, -1)
            .transposed(0, 2, 1, 3)
        let keys = key(hiddenStates)
            .reshaped(batchSize, seqLength, numHeads, -1)
            .transposed(0, 2, 1, 3)
        let values = value(hiddenStates)
            .reshaped(batchSize, seqLength, numHeads, -1)
            .transposed(0, 2, 1, 3)

        let scale = sqrt(1 / Float(queries.dim(-1)))
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, seqLength, -1)
    }
}

final class RfDetrDinov2SelfOutput: Module {

    @ModuleInfo(key: "dense") var dense: Linear

    init(_ config: RfDetrDinov2Config) {
        _dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ hiddenStates: MLXArray, _: MLXArray) -> MLXArray {
        dense(hiddenStates)
    }
}

final class RfDetrDinov2MLP: Module, UnaryLayer {

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    private let activation: UnaryLayer

    init(_ config: RfDetrDinov2Config) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * config.mlpRatio)
        _fc2.wrappedValue = Linear(config.hiddenSize * config.mlpRatio, config.hiddenSize)
        activation = GELU(approximation: .none)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(activation(fc1(x)))
    }
}

final class RfDetrScaleProjector: Module {

    @ModuleInfo(key: "scale_layers") var scaleLayers: [RfDetrScaleProjectorStage]

    init(_ config: RfDetrConfig) {
        _scaleLayers.wrappedValue = [RfDetrScaleProjectorStage(config)]
    }

    func callAsFunction(_ hiddenStates: [MLXArray]) -> MLXArray {
        let concatenated = MLX.concatenated(hiddenStates, axis: -1)
        return scaleLayers[0](concatenated)
    }
}

final class RfDetrScaleProjectorStage: Module {

    @ModuleInfo(key: "projector_layer") var projectorLayer: RfDetrC2FLayer
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: RfDetrConfig) {
        let projectorInputDim = config.backboneConfig.hiddenSize * max(1, config.backboneConfig.outIndices.count)
        _projectorLayer.wrappedValue = RfDetrC2FLayer(config, inputChannels: projectorInputDim)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        layerNorm(projectorLayer(hiddenStates))
    }
}

final class RfDetrC2FLayer: Module, UnaryLayer {

    private let hiddenChannels: Int

    @ModuleInfo(key: "conv1") var conv1: RfDetrConvNormLayer
    @ModuleInfo(key: "conv2") var conv2: RfDetrConvNormLayer
    @ModuleInfo var bottlenecks: [RfDetrRepVggBlock]

    init(_ config: RfDetrConfig, inputChannels: Int) {
        hiddenChannels = Int(Float(config.dModel) * config.hiddenExpansion)

        _conv1.wrappedValue = RfDetrConvNormLayer(
            inputChannels: inputChannels,
            outputChannels: hiddenChannels * 2,
            kernelSize: 1,
            stride: 1,
            eps: config.layerNormEps,
            activation: config.activationLayer
        )

        _conv2.wrappedValue = RfDetrConvNormLayer(
            inputChannels: (2 + config.c2fNumBlocks) * hiddenChannels,
            outputChannels: config.dModel,
            kernelSize: 1,
            stride: 1,
            eps: config.layerNormEps,
            activation: config.activationLayer
        )

        _bottlenecks.wrappedValue = (0..<config.c2fNumBlocks).map { _ in
            RfDetrRepVggBlock(config)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = conv1(x)
        let split = hidden.split(parts: 2, axis: -1)

        var allStates: [MLXArray] = [split[0], split[1]]
        var current = split[1]

        for bottleneck in bottlenecks {
            current = bottleneck(current)
            allStates.append(current)
        }

        return conv2(MLX.concatenated(allStates, axis: -1))
    }
}

final class RfDetrRepVggBlock: Module, UnaryLayer {

    @ModuleInfo(key: "conv1") var conv1: RfDetrConvNormLayer
    @ModuleInfo(key: "conv2") var conv2: RfDetrConvNormLayer

    init(_ config: RfDetrConfig) {
        let channels = Int(Float(config.dModel) * config.hiddenExpansion)

        _conv1.wrappedValue = RfDetrConvNormLayer(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            stride: 1,
            eps: config.layerNormEps,
            activation: config.activationLayer
        )

        _conv2.wrappedValue = RfDetrConvNormLayer(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            stride: 1,
            eps: config.layerNormEps,
            activation: config.activationLayer
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2(conv1(x))
    }
}

final class RfDetrConvNormLayer: Module, UnaryLayer {

    @ModuleInfo(key: "conv") var convolution: Conv2d
    @ModuleInfo(key: "norm") var normalization: LayerNorm

    private let activation: UnaryLayer

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        eps: Float,
        activation: UnaryLayer
    ) {
        _convolution.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(kernelSize / 2),
            bias: false
        )
        _normalization.wrappedValue = LayerNorm(dimensions: outputChannels, eps: eps)
        self.activation = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        activation(normalization(convolution(x)))
    }
}

final class RfDetrDecoder: Module {

    @ModuleInfo(key: "layers") var layers: [RfDetrDecoderLayer]
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm
    @ModuleInfo(key: "ref_point_head") var refPointHead: RfDetrMLPPredictionHead

    private let dModel: Int

    init(_ config: RfDetrConfig) {
        _layers.wrappedValue = (0..<config.decoderLayers).map { _ in
            RfDetrDecoderLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _refPointHead.wrappedValue = RfDetrMLPPredictionHead(
            inputDimensions: 2 * config.dModel,
            hiddenDimensions: config.dModel,
            outputDimensions: config.dModel,
            layersCount: 2
        )

        dModel = config.dModel
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        referencePoints: MLXArray,
        validRatios: MLXArray,
        spatialShape: (Int, Int),
        encoderHiddenStates: MLXArray
    ) -> MLXArray {
        func genSinePositionEmbeddings(_ positions: MLXArray, hiddenSize: Int) -> MLXArray {
            let scale: Float = 2 * .pi
            let dimension = hiddenSize / 2

            var dimT = MLXArray(0..<dimension).asType(.float32)
            dimT = 10_000 ** (2 * MLX.floorDivide(dimT, 2) / dimension)

            func encodeCoordinate(_ coordinate: MLXArray) -> MLXArray {
                let position = coordinate.expandedDimensions(axis: -1) * scale / dimT
                return MLX.stacked(
                    [
                        position[.ellipsis, .stride(from: 0, by: 2)].sin(),
                        position[.ellipsis, .stride(from: 1, by: 2)].cos(),
                    ],
                    axis: 3
                )
                .flattened(start: 2)
            }

            let posX = encodeCoordinate(positions[0..., 0..., 0])
            let posY = encodeCoordinate(positions[0..., 0..., 1])
            let posW = encodeCoordinate(positions[0..., 0..., 2])
            let posH = encodeCoordinate(positions[0..., 0..., 3])

            return MLX.concatenated([posY, posX, posW, posH], axis: -1).asType(positions.dtype)
        }

        var hiddenStates = inputsEmbeds

        var referenceInputs = referencePoints.expandedDimensions(axis: 2)
        referenceInputs *= MLX.concatenated([validRatios, validRatios], axis: -1).expandedDimensions(axis: 1)

        let querySineEmbed = genSinePositionEmbeddings(referenceInputs[0..., 0..., 0, 0...], hiddenSize: dModel)
        let queryPos = refPointHead(querySineEmbed)

        for layer in layers {
            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionEmbeddings: queryPos,
                referencePoints: referenceInputs,
                spatialShape: spatialShape,
                encoderHiddenStates: encoderHiddenStates
            )
        }

        return layerNorm(hiddenStates)
    }
}

final class RfDetrDecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttention: RfDetrDecoderSelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm

    @ModuleInfo(key: "cross_attn") var crossAttention: RfDetrMultiscaleDeformableAttention
    @ModuleInfo(key: "cross_attn_layer_norm") var crossAttentionLayerNorm: LayerNorm

    @ModuleInfo(key: "mlp") var mlp: RfDetrDecoderMLP
    @ModuleInfo(key: "layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: RfDetrConfig) {
        _selfAttention.wrappedValue = RfDetrDecoderSelfAttention(config)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _crossAttention.wrappedValue = RfDetrMultiscaleDeformableAttention(config)
        _crossAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _mlp.wrappedValue = RfDetrDecoderMLP(config)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        positionEmbeddings: MLXArray,
        referencePoints: MLXArray,
        spatialShape: (Int, Int),
        encoderHiddenStates: MLXArray
    ) -> MLXArray {
        var hidden = hiddenStates

        let selfAttentionOutput = selfAttention(
            hiddenStates,
            positionEmbeddings: positionEmbeddings
        )
        hidden = selfAttentionLayerNorm(hidden + selfAttentionOutput)

        let crossAttentionOutput = crossAttention(
            hiddenStates: hidden,
            positionEmbeddings: positionEmbeddings,
            referencePoints: referencePoints,
            encoderHiddenStates: encoderHiddenStates,
            spatialShape: spatialShape
        )
        hidden = crossAttentionLayerNorm(hidden + crossAttentionOutput)

        let mlpOutput = mlp(hidden)

        return finalLayerNorm(hidden + mlpOutput)
    }
}

final class RfDetrDecoderMLP: Module, UnaryLayer {

    @ModuleInfo(key: "fc1") var linear1: Linear
    @ModuleInfo(key: "fc2") var linear2: Linear

    private let activation: UnaryLayer

    init(_ config: RfDetrConfig) {
        _linear1.wrappedValue = Linear(config.dModel, config.decoderFfnDim)
        _linear2.wrappedValue = Linear(config.decoderFfnDim, config.dModel)
        activation = config.decoderActivationLayer
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        linear2(activation(linear1(hiddenStates)))
    }
}

final class RfDetrDecoderSelfAttention: Module {

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outProjection: Linear

    private let numHeads: Int

    init(_ config: RfDetrConfig) {
        numHeads = config.decoderSelfAttentionHeads
        _queryProjection.wrappedValue = Linear(config.dModel, config.dModel, bias: config.attentionBias)
        _keyProjection.wrappedValue = Linear(config.dModel, config.dModel, bias: config.attentionBias)
        _valueProjection.wrappedValue = Linear(config.dModel, config.dModel, bias: config.attentionBias)
        _outProjection.wrappedValue = Linear(config.dModel, config.dModel, bias: config.attentionBias)
    }

    func callAsFunction(_ hiddenStates: MLXArray, positionEmbeddings: MLXArray? = nil) -> MLXArray {
        let originalHiddenStates = hiddenStates
        let hiddenWithPos = positionEmbeddings.map { hiddenStates + $0 } ?? hiddenStates

        let (batchSize, queryLength, _) = hiddenStates.shape3
        let keyLength = originalHiddenStates.shape[1]

        let queries = queryProjection(hiddenWithPos).reshaped(batchSize, queryLength, numHeads, -1).transposed(0, 2, 1, 3)
        let keys = keyProjection(hiddenWithPos).reshaped(batchSize, keyLength, numHeads, -1).transposed(0, 2, 1, 3)
        let values = valueProjection(originalHiddenStates).reshaped(batchSize, keyLength, numHeads, -1).transposed(0, 2, 1, 3)

        let scale = sqrt(1 / Float(queries.dim(-1)))
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, queryLength, -1)

        return outProjection(output)
    }
}

final class RfDetrMultiscaleDeformableAttention: Module {

    @ModuleInfo(key: "sampling_offsets") var samplingOffsets: Linear
    @ModuleInfo(key: "attention_weights") var attentionWeights: Linear
    @ModuleInfo(key: "value_proj") var valueProjection: Linear
    @ModuleInfo(key: "output_proj") var outputProjection: Linear

    private let nHeads: Int
    private let nPoints: Int
    private let nLevels: Int
    private let dModel: Int

    init(_ config: RfDetrConfig) {
        nHeads = config.decoderCrossAttentionHeads
        nPoints = config.decoderNPoints
        nLevels = config.numFeatureLevels
        dModel = config.dModel
        _samplingOffsets.wrappedValue = Linear(config.dModel, nHeads * nLevels * nPoints * 2, bias: config.attentionBias)
        _attentionWeights.wrappedValue = Linear(config.dModel, nHeads * nLevels * nPoints, bias: config.attentionBias)
        _valueProjection.wrappedValue = Linear(config.dModel, config.dModel, bias: config.attentionBias)
        _outputProjection.wrappedValue = Linear(config.dModel, config.dModel, bias: config.attentionBias)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        positionEmbeddings: MLXArray,
        referencePoints: MLXArray,
        encoderHiddenStates: MLXArray,
        spatialShape: (Int, Int)
    ) -> MLXArray {
        let query = hiddenStates + positionEmbeddings
        let (batchSize, queryCount, _) = query.shape3
        let headDim = dModel / nHeads

        let value = valueProjection(encoderHiddenStates)
            .reshaped(batchSize, encoderHiddenStates.shape[1], nHeads, headDim)

        let offsets = samplingOffsets(query)
            .reshaped(batchSize, queryCount, nHeads, nLevels, nPoints, 2)

        let weights = attentionWeights(query)
            .reshaped(batchSize, queryCount, nHeads, nLevels * nPoints)
            .softmax(axis: -1)
            .reshaped(batchSize, queryCount, nHeads, nLevels, nPoints)

        let referenceXY = referencePoints[0..., 0..., 0..., 0..<2]
        let referenceWH = referencePoints[0..., 0..., 0..., 2...]

        let samplingLocations =
            referenceXY.expandedDimensions(axis: 2).expandedDimensions(axis: 2)
            + offsets / Float(nPoints) * referenceWH.expandedDimensions(axis: 2).expandedDimensions(axis: 2) * 0.5

        let (height, width) = spatialShape
        let valueMap =
            value
            .reshaped(batchSize, height, width, nHeads, headDim)
            .transposed(0, 3, 1, 2, 4)
            .reshaped(batchSize * nHeads, height, width, headDim)

        let levelSamplingLocations = samplingLocations[0..., 0..., 0..., 0, 0...]
            .transposed(0, 2, 1, 3, 4)
            .reshaped(batchSize * nHeads, queryCount, nPoints, 2)

        let sampledValues = bilinearSample(
            featureMap: valueMap,
            samplingLocations: levelSamplingLocations
        )

        let levelAttentionWeights = weights[0..., 0..., 0..., 0, 0...]
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize * nHeads, queryCount, nPoints)

        let output = (sampledValues * levelAttentionWeights.expandedDimensions(axis: -1))
            .sum(axis: 2)
            .reshaped(batchSize, nHeads, queryCount, headDim)
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, queryCount, dModel)

        return outputProjection(output)
    }

    private func bilinearSample(featureMap: MLXArray, samplingLocations: MLXArray) -> MLXArray {
        let (_, height, width, channels) = featureMap.shape4

        let samplingX = samplingLocations[0..., 0..., 0..., 0] * Float(width) - 0.5
        let samplingY = samplingLocations[0..., 0..., 0..., 1] * Float(height) - 0.5

        let x0 = MLX.floor(samplingX).asType(.int32)
        let y0 = MLX.floor(samplingY).asType(.int32)
        let x1 = x0 + 1
        let y1 = y0 + 1

        let x0f = x0.asType(samplingX.dtype)
        let y0f = y0.asType(samplingY.dtype)

        let fx = samplingX - x0f
        let fy = samplingY - y0f

        var wa = (1 - fx) * (1 - fy)
        var wb = (1 - fx) * fy
        var wc = fx * (1 - fy)
        var wd = fx * fy

        let valid00 = (x0 .>= 0) .&& (x0 .< width) .&& (y0 .>= 0) .&& (y0 .< height)
        let valid01 = (x0 .>= 0) .&& (x0 .< width) .&& (y1 .>= 0) .&& (y1 .< height)
        let valid10 = (x1 .>= 0) .&& (x1 .< width) .&& (y0 .>= 0) .&& (y0 .< height)
        let valid11 = (x1 .>= 0) .&& (x1 .< width) .&& (y1 .>= 0) .&& (y1 .< height)

        wa = MLX.where(valid00, wa, MLX.zeros(like: wa))
        wb = MLX.where(valid01, wb, MLX.zeros(like: wb))
        wc = MLX.where(valid10, wc, MLX.zeros(like: wc))
        wd = MLX.where(valid11, wd, MLX.zeros(like: wd))

        let x0c = MLX.clip(x0, min: 0, max: width - 1)
        let y0c = MLX.clip(y0, min: 0, max: height - 1)
        let x1c = MLX.clip(x1, min: 0, max: width - 1)
        let y1c = MLX.clip(y1, min: 0, max: height - 1)

        let flattenedFeatures = featureMap.reshaped(featureMap.shape[0], height * width, channels)

        let idx00 = (y0c * width + x0c).asType(.int32)
        let idx01 = (y1c * width + x0c).asType(.int32)
        let idx10 = (y0c * width + x1c).asType(.int32)
        let idx11 = (y1c * width + x1c).asType(.int32)

        let spatialCount = MLXArray(height * width, dtype: .int32)
        let batchOffsets = (MLXArray(0..<flattenedFeatures.shape[0]).asType(.int32) * spatialCount)
            .reshaped(-1, 1, 1)

        let flattenedGlobalFeatures = flattenedFeatures.reshaped(-1, channels)

        let v00 = MLX.take(flattenedGlobalFeatures, idx00 + batchOffsets, axis: 0)
        let v01 = MLX.take(flattenedGlobalFeatures, idx01 + batchOffsets, axis: 0)
        let v10 = MLX.take(flattenedGlobalFeatures, idx10 + batchOffsets, axis: 0)
        let v11 = MLX.take(flattenedGlobalFeatures, idx11 + batchOffsets, axis: 0)

        return wa.expandedDimensions(axis: -1) * v00
            + wb.expandedDimensions(axis: -1) * v01
            + wc.expandedDimensions(axis: -1) * v10
            + wd.expandedDimensions(axis: -1) * v11
    }
}

final class RfDetrMLPPredictionHead: Module, UnaryLayer {

    @ModuleInfo(key: "layers") var layers: [Linear]

    private let activation = ReLU()

    init(
        inputDimensions: Int,
        hiddenDimensions: Int,
        outputDimensions: Int,
        layersCount: Int
    ) {
        _layers.wrappedValue = zip(
            [inputDimensions] + (1..<layersCount).map { _ in hiddenDimensions },
            (1..<layersCount).map { _ in hiddenDimensions } + [outputDimensions]
        ).map {
            Linear($0.0, $0.1)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.last!(
            layers.dropLast().reduce(x) {
                activation($1($0))
            }
        )
    }
}

private extension RfDetrConfig {

    var activationLayer: UnaryLayer {
        switch activationFunction {
        case "gelu":
            GELU(approximation: .none)
        case "silu":
            SiLU()
        default:
            ReLU()
        }
    }

    var decoderActivationLayer: UnaryLayer {
        switch decoderActivationFunction {
        case "gelu":
            GELU(approximation: .none)
        case "silu":
            SiLU()
        default:
            ReLU()
        }
    }
}

private func refineBoxes(_ referencePoints: MLXArray, _ deltas: MLXArray) -> MLXArray {
    let cxcy = deltas[0..., 0..., 0..<2] * referencePoints[0..., 0..., 2...] + referencePoints[0..., 0..., 0..<2]
    let widthHeight = deltas[0..., 0..., 2...].exp() * referencePoints[0..., 0..., 2...]
    return MLX.concatenated([cxcy, widthHeight], axis: -1)
}
