//
//  RtDetr+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.03.2026.
//

import Foundation
import MLX
import MLXNN

final class RtDetrV2ModelForObjectDetection: Module, Predictor {

    typealias Output = (
        logits: MLXArray,
        probs: MLXArray,
        boxes: MLXArray
    )

    @ModuleInfo var model: RtDetrV2Model

    init(_ config: RtDetrV2ForObjectDetectionConfig) {
        _model.wrappedValue = RtDetrV2Model(config, numLabels: config.id2label.count)
    }

    private lazy var _predict = MLX.compile { [unowned self] inputs in
        let output = model(inputs[0])
        let probs = output.logits.softmax(axis: -1)
        return [output.logits, probs, output.boxes]
    }

    func predict(_ input: ImageInput) throws -> Output {
        let outputs = _predict([input.pixelValues])
        return (outputs[0], outputs[1], outputs[2])
    }
}

final class RtDetrV2Model: Module {

    typealias Output = (
        hiddenStates: MLXArray,
        logits: MLXArray,
        boxes: MLXArray
    )

    @ModuleInfo(key: "backbone") var backbone: RtDetrV2ConvEncoder
    @ModuleInfo(key: "encoder_input_proj") var encoderInputProj: [[UnaryLayer]]
    @ModuleInfo(key: "encoder") var encoder: RtDetrV2HybridEncoder

    @ModuleInfo(key: "denoising_class_embed") var denoisingClassEmbed: Embedding

    @ModuleInfo(key: "enc_output") var encOutput: [UnaryLayer]
    @ModuleInfo(key: "enc_score_head") var encScoreHead: Linear
    @ModuleInfo(key: "enc_bbox_head") var encBBoxHead: RtDetrV2MLPPredictionHead

    @ModuleInfo(key: "decoder_input_proj") var decoderInputProj: [[UnaryLayer]]
    @ModuleInfo(key: "decoder") var decoder: RtDetrV2Decoder

    private let numQueries: Int

    init(_ config: RtDetrV2Config, numLabels: Int) {
        _backbone.wrappedValue = RtDetrV2ConvEncoder(config)

        _encoderInputProj.wrappedValue = config.encoderInChannels.map {
            [
                Conv2d(
                    inputChannels: $0,
                    outputChannels: config.encoderHiddenDim,
                    kernelSize: 1,
                    stride: IntOrPair(1),
                    padding: IntOrPair(0),
                    bias: false
                ),
                BatchNorm(featureCount: config.encoderHiddenDim, eps: config.batchNormEps),
            ]
        }

        _encoder.wrappedValue = RtDetrV2HybridEncoder(config)

        _denoisingClassEmbed.wrappedValue = Embedding(
            embeddingCount: numLabels + 1,
            dimensions: config.dModel
        )

        _encOutput.wrappedValue = [
            Linear(config.dModel, config.dModel),
            LayerNorm(dimensions: config.dModel, eps: config.layerNormEps),
        ]
        _encScoreHead.wrappedValue = Linear(config.dModel, numLabels)
        _encBBoxHead.wrappedValue = RtDetrV2MLPPredictionHead(
            inputDimensions: config.dModel,
            hiddenDimensions: config.dModel,
            outputDimensions: 4,
            layersCount: 3
        )

        var decoderProjectionLayers: [[UnaryLayer]] = config.decoderInChannels.map {
            [
                Conv2d(
                    inputChannels: $0,
                    outputChannels: config.dModel,
                    kernelSize: 1,
                    stride: IntOrPair(1),
                    padding: IntOrPair(0),
                    bias: false
                ),
                BatchNorm(featureCount: config.dModel, eps: config.batchNormEps),
            ]
        }

        if config.numFeatureLevels > decoderProjectionLayers.count, let last = config.decoderInChannels.last {
            var inChannels = last
            for _ in decoderProjectionLayers.count..<config.numFeatureLevels {
                decoderProjectionLayers.append(
                    [
                        Conv2d(
                            inputChannels: inChannels,
                            outputChannels: config.dModel,
                            kernelSize: IntOrPair(3),
                            stride: IntOrPair(2),
                            padding: IntOrPair(1),
                            bias: false
                        ),
                        BatchNorm(featureCount: config.dModel, eps: config.batchNormEps),
                    ]
                )
                inChannels = config.dModel
            }
        }

        _decoderInputProj.wrappedValue = decoderProjectionLayers
        _decoder.wrappedValue = RtDetrV2Decoder(config, numLabels: numLabels)

        numQueries = config.numQueries
    }

    func callAsFunction(_ pixelValues: MLXArray) -> Output {
        let (batchSize, height, width, _) = pixelValues.shape4
        let pixelMask = MLXArray.ones([batchSize, height, width], dtype: .bool)

        let features = backbone(pixelValues, pixelMask)
        var projectedFeatures: [MLXArray] = []
        for (level, (featureMap, _)) in features.enumerated() {
            let projection = encoderInputProj[level]
            projectedFeatures.append(projection[1](projection[0](featureMap)))
        }

        let encoderFeatureMaps = encoder(projectedFeatures)

        var sources: [MLXArray] = []
        for (level, source) in encoderFeatureMaps.enumerated() {
            let projection = decoderInputProj[level]
            sources.append(projection[1](projection[0](source)))
        }

        if sources.count < decoderInputProj.count {
            var source = encoderFeatureMaps.last!
            for level in sources.count..<decoderInputProj.count {
                let projection = decoderInputProj[level]
                source = projection[1](projection[0](source))
                sources.append(source)
            }
        }

        var sourceFlatten: [MLXArray] = []
        var spatialShapesList: [(Int, Int)] = []
        for source in sources {
            let (_, h, w, _) = source.shape4
            spatialShapesList.append((h, w))
            sourceFlatten.append(source.flattened(start: 1, end: 2))
        }

        let flattenedSource = MLX.concatenated(sourceFlatten, axis: 1)
        let spatialShapes =
            spatialShapesList
            .flatMap { [$0.0, $0.1] }
            .asMLXArray(dtype: .int32)
            .reshaped(spatialShapesList.count, 2)

        let (anchors, validMask) = generateAnchors(spatialShapesList, dtype: flattenedSource.dtype)

        let memory = validMask.asType(flattenedSource.dtype) * flattenedSource
        let outputMemory = encOutput[1](encOutput[0](memory))

        let encOutputsClass = encScoreHead(outputMemory)
        let encOutputsCoordLogits = encBBoxHead(outputMemory) + anchors

        let maxScores = encOutputsClass.max(axis: -1)
        let topKIndices = sortedTopKIndices(maxScores, k: numQueries)

        let index4 = MLX.repeated(topKIndices.expandedDimensions(axis: -1), count: 4, axis: -1)
        let referencePointsUnact = MLX.takeAlong(encOutputsCoordLogits, index4, axis: 1)

        let indexD = MLX.repeated(topKIndices.expandedDimensions(axis: -1), count: outputMemory.shape[2], axis: -1)
        let target = MLX.takeAlong(outputMemory, indexD, axis: 1)

        let decoderOutput = decoder(
            inputsEmbeds: target,
            encoderHiddenStates: flattenedSource,
            referencePoints: referencePointsUnact,
            spatialShapes: spatialShapes,
            spatialShapesList: spatialShapesList
        )

        return (
            hiddenStates: decoderOutput.hiddenStates,
            logits: decoderOutput.logits,
            boxes: decoderOutput.boxes
        )
    }

    private func generateAnchors(
        _ spatialShapesList: [(Int, Int)],
        dtype: DType,
        gridSize: Float = 0.05,
        eps: Float = 1e-2
    ) -> (MLXArray, MLXArray) {
        var anchors: [MLXArray] = []

        for (level, (height, width)) in spatialShapesList.enumerated() {
            let gridY = MLXArray(0..<height).asType(dtype)
            let gridX = MLXArray(0..<width).asType(dtype)
            let meshGrid = MLX.meshGrid([gridY, gridX], indexing: .ij)
            let meshY = meshGrid[0]
            let meshX = meshGrid[1]

            var grid = MLX.stacked([meshX, meshY], axis: -1)
            grid = grid.expandedDimensions(axis: 0) + 0.5
            grid[0..., 0..., 0..., 0] = grid[0..., 0..., 0..., 0] / Float(width)
            grid[0..., 0..., 0..., 1] = grid[0..., 0..., 0..., 1] / Float(height)

            let widthHeight = MLX.ones(like: grid) * (gridSize * pow(2, Float(level)))
            let proposal = MLX.concatenated([grid, widthHeight], axis: -1).reshaped(1, height * width, 4)
            anchors.append(proposal)
        }

        var anchorsArray = MLX.concatenated(anchors, axis: 1)
        let validMask = ((anchorsArray .> eps) .&& (anchorsArray .< (1 - eps))).min(axis: -1, keepDims: true)
        anchorsArray = MLX.log(anchorsArray / (1 - anchorsArray))

        let inf = MLX.full(anchorsArray.shape, values: Float.infinity, type: Float.self)
        anchorsArray = MLX.where(validMask, anchorsArray, inf)

        return (anchorsArray, validMask)
    }

    private func sortedTopKIndices(_ scores: MLXArray, k: Int) -> MLXArray {
        let topK = max(1, min(k, scores.shape[1]))
        var indices = MLX.argPartition(-scores, kth: topK - 1, axis: 1)[0..., 0..<topK]
        let topKScores = MLX.takeAlong(scores, indices, axis: 1)
        let order = MLX.argSort(-topKScores, axis: 1)
        indices = MLX.takeAlong(indices, order, axis: 1)
        return indices
    }
}

final class RtDetrV2ConvEncoder: Module {

    @ModuleInfo(key: "model") var model: RtDetrResNetBackbone

    init(_ config: RtDetrV2Config) {
        _model.wrappedValue = RtDetrResNetBackbone(config.backboneConfig)
    }

    func callAsFunction(_ pixelValues: MLXArray, _ pixelMask: MLXArray) -> [(MLXArray, MLXArray)] {
        let features = model(pixelValues)
        return features.map {
            let mask = pixelMask.expandedDimensions(axis: -1)
                .asType(pixelValues.dtype)
                .interpolate(size: [$0.shape[1], $0.shape[2]])
                .squeezed(axis: -1)
                .asType(.bool)
            return ($0, mask)
        }
    }
}

final class RtDetrResNetBackbone: Module {

    @ModuleInfo(key: "embedder") var embedder: RtDetrResNetEmbeddings
    @ModuleInfo(key: "encoder") var encoder: RtDetrResNetEncoder

    private let outIndices: Set<Int>

    init(_ config: RtDetrResNetConfig) {
        _embedder.wrappedValue = RtDetrResNetEmbeddings(config)
        _encoder.wrappedValue = RtDetrResNetEncoder(config)
        outIndices = Set(config.outIndices)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> [MLXArray] {
        let embeddingOutput = embedder(pixelValues)
        let hiddenStates = encoder(embeddingOutput)

        var featureMaps: [MLXArray] = []
        for (index, hiddenState) in hiddenStates.enumerated() where outIndices.contains(index) {
            featureMaps.append(hiddenState)
        }

        return featureMaps
    }
}

final class RtDetrResNetEmbeddings: Module {

    @ModuleInfo(key: "embedder") var embedder: [RtDetrResNetConvLayer]

    let pooler: PaddedMaxPool2d

    init(_ config: RtDetrResNetConfig) {
        _embedder.wrappedValue = [
            RtDetrResNetConvLayer(
                inputChannels: config.numChannels,
                outputChannels: config.embeddingSize / 2,
                kernelSize: 3,
                stride: 2,
                activation: config.activationLayer
            ),
            RtDetrResNetConvLayer(
                inputChannels: config.embeddingSize / 2,
                outputChannels: config.embeddingSize / 2,
                kernelSize: 3,
                stride: 1,
                activation: config.activationLayer
            ),
            RtDetrResNetConvLayer(
                inputChannels: config.embeddingSize / 2,
                outputChannels: config.embeddingSize,
                kernelSize: 3,
                stride: 1,
                activation: config.activationLayer
            ),
        ]

        pooler = PaddedMaxPool2d(kernelSize: 3, stride: 2, padding: 1)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let embedded = embedder.reduce(pixelValues) { $1($0) }
        return pooler(embedded)
    }
}

final class RtDetrResNetEncoder: Module {

    @ModuleInfo(key: "stages") var stages: [RtDetrResNetStage]

    init(_ config: RtDetrResNetConfig) {
        var builtStages: [RtDetrResNetStage] = [
            RtDetrResNetStage(
                config,
                inputChannels: config.embeddingSize,
                outputChannels: config.hiddenSizes[0],
                stride: config.downsampleInFirstStage ? 2 : 1,
                depth: config.depths[0]
            )
        ]

        let pairs = zip(config.hiddenSizes, config.hiddenSizes.dropFirst())
        for ((inputChannels, outputChannels), depth) in zip(pairs, config.depths.dropFirst()) {
            builtStages.append(
                RtDetrResNetStage(
                    config,
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    stride: 2,
                    depth: depth
                )
            )
        }

        _stages.wrappedValue = builtStages
    }

    func callAsFunction(_ hiddenState: MLXArray) -> [MLXArray] {
        var hiddenStates: [MLXArray] = [hiddenState]
        var current = hiddenState
        for stage in stages {
            current = stage(current)
            hiddenStates.append(current)
        }
        return hiddenStates
    }
}

final class RtDetrResNetStage: Module, UnaryLayer {

    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    init(
        _ config: RtDetrResNetConfig,
        inputChannels: Int,
        outputChannels: Int,
        stride: Int,
        depth: Int
    ) {
        var stageLayers: [UnaryLayer] = []

        if config.layerType == "basic" {
            stageLayers.append(
                RtDetrResNetBasicLayer(
                    config,
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    stride: stride,
                    shouldApplyShortcut: true
                )
            )

            if depth > 1 {
                for _ in 0..<(depth - 1) {
                    stageLayers.append(
                        RtDetrResNetBasicLayer(
                            config,
                            inputChannels: outputChannels,
                            outputChannels: outputChannels,
                            stride: 1,
                            shouldApplyShortcut: false
                        )
                    )
                }
            }
        } else {
            stageLayers.append(
                RtDetrResNetBottleNeckLayer(
                    config,
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    stride: stride
                )
            )

            if depth > 1 {
                for _ in 0..<(depth - 1) {
                    stageLayers.append(
                        RtDetrResNetBottleNeckLayer(
                            config,
                            inputChannels: outputChannels,
                            outputChannels: outputChannels,
                            stride: 1
                        )
                    )
                }
            }
        }

        _layers.wrappedValue = stageLayers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.reduce(x) { $1($0) }
    }
}

final class RtDetrResNetBasicLayer: Module, UnaryLayer {

    let shortcut: UnaryLayer
    let shortcutPool: AvgPool2d?

    @ModuleInfo(key: "layer") var layer: [RtDetrResNetConvLayer]

    private let activation: UnaryLayer

    init(
        _ config: RtDetrResNetConfig,
        inputChannels: Int,
        outputChannels: Int,
        stride: Int,
        shouldApplyShortcut: Bool
    ) {
        activation = config.activationLayer

        if shouldApplyShortcut {
            if inputChannels != outputChannels {
                if stride == 2 {
                    shortcut = RtDetrResNetShortCut(
                        inputChannels: inputChannels,
                        outputChannels: outputChannels,
                        stride: 1
                    )
                    shortcutPool = AvgPool2d(kernelSize: 2, stride: 2)
                } else {
                    shortcut = RtDetrResNetShortCut(
                        inputChannels: inputChannels,
                        outputChannels: outputChannels,
                        stride: 1
                    )
                    shortcutPool = nil
                }
            } else {
                shortcut = RtDetrResNetShortCut(
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    stride: stride
                )
                shortcutPool = nil
            }
        } else {
            shortcut = Identity()
            shortcutPool = nil
        }

        _layer.wrappedValue = [
            RtDetrResNetConvLayer(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: 3,
                stride: stride,
                activation: config.activationLayer
            ),
            RtDetrResNetConvLayer(
                inputChannels: outputChannels,
                outputChannels: outputChannels,
                kernelSize: 3,
                stride: 1,
                activation: Identity()
            ),
        ]
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        guard shortcutPool != nil else {
            return parameters
        }

        return parameters.renameKeys {
            $0.replacing(/^shortcut\.1\./, with: "shortcut.")
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let residualInput = shortcutPool.map { $0(hiddenStates) } ?? hiddenStates
        let residual = shortcut(residualInput)
        let output = layer.reduce(hiddenStates) { $1($0) } + residual
        return activation(output)
    }
}

final class RtDetrResNetBottleNeckLayer: Module, UnaryLayer {

    let shortcut: UnaryLayer
    let shortcutPool: AvgPool2d?

    @ModuleInfo(key: "layer") var layer: [RtDetrResNetConvLayer]

    private let activation: UnaryLayer

    init(
        _ config: RtDetrResNetConfig,
        inputChannels: Int,
        outputChannels: Int,
        stride: Int
    ) {
        let reduction = 4
        let reducedChannels = outputChannels / reduction
        let shouldApplyShortcut = inputChannels != outputChannels || stride != 1
        activation = config.activationLayer

        if stride == 2 {
            shortcut =
                shouldApplyShortcut
                ? RtDetrResNetShortCut(inputChannels: inputChannels, outputChannels: outputChannels, stride: 1)
                : Identity()
            shortcutPool = shouldApplyShortcut ? AvgPool2d(kernelSize: 2, stride: 2) : nil
        } else {
            shortcut =
                shouldApplyShortcut
                ? RtDetrResNetShortCut(inputChannels: inputChannels, outputChannels: outputChannels, stride: stride)
                : Identity()
            shortcutPool = nil
        }

        _layer.wrappedValue = [
            RtDetrResNetConvLayer(
                inputChannels: inputChannels,
                outputChannels: reducedChannels,
                kernelSize: 1,
                stride: config.downsampleInBottleneck ? stride : 1,
                activation: config.activationLayer
            ),
            RtDetrResNetConvLayer(
                inputChannels: reducedChannels,
                outputChannels: reducedChannels,
                kernelSize: 3,
                stride: config.downsampleInBottleneck ? 1 : stride,
                activation: config.activationLayer
            ),
            RtDetrResNetConvLayer(
                inputChannels: reducedChannels,
                outputChannels: outputChannels,
                kernelSize: 1,
                stride: 1,
                activation: Identity()
            ),
        ]
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        guard shortcutPool != nil else {
            return parameters
        }

        return parameters.renameKeys {
            $0.replacing(/^shortcut\.1\./, with: "shortcut.")
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let residualInput = shortcutPool.map { $0(hiddenStates) } ?? hiddenStates
        let residual = shortcut(residualInput)
        let output = layer.reduce(hiddenStates) { $1($0) } + residual
        return activation(output)
    }
}

final class RtDetrResNetConvLayer: Module, UnaryLayer {

    @ModuleInfo(key: "convolution") var convolution: Conv2d
    @ModuleInfo(key: "normalization") var normalization: BatchNorm

    private let activation: UnaryLayer

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
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
        _normalization.wrappedValue = BatchNorm(featureCount: outputChannels)
        self.activation = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        activation(normalization(convolution(x)))
    }
}

final class RtDetrResNetShortCut: Module, UnaryLayer {

    @ModuleInfo(key: "convolution") var convolution: Conv2d
    @ModuleInfo(key: "normalization") var normalization: BatchNorm

    init(inputChannels: Int, outputChannels: Int, stride: Int) {
        _convolution.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 1,
            stride: IntOrPair(stride),
            bias: false
        )
        _normalization.wrappedValue = BatchNorm(featureCount: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        normalization(convolution(x))
    }
}

final class RtDetrV2ConvNormLayer: Module, UnaryLayer {

    @ModuleInfo(key: "conv") var convolution: Conv2d
    @ModuleInfo(key: "norm") var normalization: BatchNorm

    private let activation: UnaryLayer

    init(
        _ config: RtDetrV2Config,
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        activation: UnaryLayer? = nil
    ) {
        _convolution.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair((kernelSize - 1) / 2),
            bias: false
        )
        _normalization.wrappedValue = BatchNorm(featureCount: outputChannels, eps: config.batchNormEps)
        self.activation = activation ?? Identity()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        activation(normalization(convolution(hiddenStates)))
    }
}

final class RtDetrV2RepVggBlock: Module, UnaryLayer {

    @ModuleInfo(key: "conv1") var conv1: RtDetrV2ConvNormLayer
    @ModuleInfo(key: "conv2") var conv2: RtDetrV2ConvNormLayer

    private let activation: UnaryLayer

    init(_ config: RtDetrV2Config) {
        let hiddenChannels = Int(Float(config.encoderHiddenDim) * config.hiddenExpansion)
        _conv1.wrappedValue = RtDetrV2ConvNormLayer(
            config,
            inputChannels: hiddenChannels,
            outputChannels: hiddenChannels,
            kernelSize: 3,
            stride: 1
        )
        _conv2.wrappedValue = RtDetrV2ConvNormLayer(
            config,
            inputChannels: hiddenChannels,
            outputChannels: hiddenChannels,
            kernelSize: 1,
            stride: 1
        )
        activation = config.activationLayer
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        activation(conv1(x) + conv2(x))
    }
}

final class RtDetrV2CSPRepLayer: Module, UnaryLayer {

    @ModuleInfo(key: "conv1") var conv1: RtDetrV2ConvNormLayer
    @ModuleInfo(key: "conv2") var conv2: RtDetrV2ConvNormLayer
    @ModuleInfo(key: "bottlenecks") var bottlenecks: [RtDetrV2RepVggBlock]
    @ModuleInfo(key: "conv3") var conv3: UnaryLayer

    init(_ config: RtDetrV2Config) {
        let hiddenChannels = Int(Float(config.encoderHiddenDim) * config.hiddenExpansion)
        let activation = config.activationLayer

        _conv1.wrappedValue = RtDetrV2ConvNormLayer(
            config,
            inputChannels: config.encoderHiddenDim * 2,
            outputChannels: hiddenChannels,
            kernelSize: 1,
            stride: 1,
            activation: activation
        )

        _conv2.wrappedValue = RtDetrV2ConvNormLayer(
            config,
            inputChannels: config.encoderHiddenDim * 2,
            outputChannels: hiddenChannels,
            kernelSize: 1,
            stride: 1,
            activation: activation
        )

        _bottlenecks.wrappedValue = (0..<3).map { _ in
            RtDetrV2RepVggBlock(config)
        }

        if hiddenChannels == config.encoderHiddenDim {
            _conv3.wrappedValue = Identity()
        } else {
            _conv3.wrappedValue = RtDetrV2ConvNormLayer(
                config,
                inputChannels: hiddenChannels,
                outputChannels: config.encoderHiddenDim,
                kernelSize: 1,
                stride: 1,
                activation: activation
            )
        }
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hiddenStates1 = conv1(hiddenStates)
        for bottleneck in bottlenecks {
            hiddenStates1 = bottleneck(hiddenStates1)
        }

        let hiddenStates2 = conv2(hiddenStates)
        return conv3(hiddenStates1 + hiddenStates2)
    }
}

final class RtDetrV2HybridEncoder: Module {

    @ModuleInfo(key: "encoder") var encoder: [RtDetrV2AIFILayer]
    @ModuleInfo(key: "lateral_convs") var lateralConvs: [RtDetrV2ConvNormLayer]
    @ModuleInfo(key: "fpn_blocks") var fpnBlocks: [RtDetrV2CSPRepLayer]
    @ModuleInfo(key: "downsample_convs") var downsampleConvs: [RtDetrV2ConvNormLayer]
    @ModuleInfo(key: "pan_blocks") var panBlocks: [RtDetrV2CSPRepLayer]

    private let encodeProjLayers: [Int]
    private let numFpnStages: Int
    private let numPanStages: Int

    init(_ config: RtDetrV2Config) {
        encodeProjLayers = config.encodeProjLayers
        numFpnStages = max(0, config.encoderInChannels.count - 1)
        numPanStages = max(0, config.encoderInChannels.count - 1)

        _encoder.wrappedValue = (0..<encodeProjLayers.count).map { _ in
            RtDetrV2AIFILayer(config)
        }

        _lateralConvs.wrappedValue = (0..<numFpnStages).map { _ in
            RtDetrV2ConvNormLayer(
                config,
                inputChannels: config.encoderHiddenDim,
                outputChannels: config.encoderHiddenDim,
                kernelSize: 1,
                stride: 1,
                activation: config.activationLayer
            )
        }

        _fpnBlocks.wrappedValue = (0..<numFpnStages).map { _ in
            RtDetrV2CSPRepLayer(config)
        }

        _downsampleConvs.wrappedValue = (0..<numPanStages).map { _ in
            RtDetrV2ConvNormLayer(
                config,
                inputChannels: config.encoderHiddenDim,
                outputChannels: config.encoderHiddenDim,
                kernelSize: 3,
                stride: 2,
                activation: config.activationLayer
            )
        }

        _panBlocks.wrappedValue = (0..<numPanStages).map { _ in
            RtDetrV2CSPRepLayer(config)
        }
    }

    func callAsFunction(_ inputsEmbeds: [MLXArray]) -> [MLXArray] {
        var featureMaps = inputsEmbeds

        if !encoder.isEmpty {
            for (index, encodedIndex) in encodeProjLayers.enumerated() {
                featureMaps[encodedIndex] = encoder[index](featureMaps[encodedIndex])
            }
        }

        var fpnFeatureMaps: [MLXArray] = [featureMaps.last!]

        for (index, (lateralConv, fpnBlock)) in zip(lateralConvs, fpnBlocks).enumerated() {
            let backboneFeature = featureMaps[numFpnStages - index - 1]
            var topFeature = fpnFeatureMaps.last!
            topFeature = lateralConv(topFeature)
            fpnFeatureMaps[fpnFeatureMaps.count - 1] = topFeature

            let upsampled = topFeature.interpolate(
                size: [backboneFeature.shape[1], backboneFeature.shape[2]],
                mode: .nearest
            )
            let fused = MLX.concatenated([upsampled, backboneFeature], axis: -1)
            let newFeature = fpnBlock(fused)
            fpnFeatureMaps.append(newFeature)
        }

        fpnFeatureMaps.reverse()

        var panFeatureMaps: [MLXArray] = [fpnFeatureMaps[0]]
        for (index, (downsampleConv, panBlock)) in zip(downsampleConvs, panBlocks).enumerated() {
            let topPanFeature = panFeatureMaps.last!
            let fpnFeature = fpnFeatureMaps[index + 1]
            let downsampled = downsampleConv(topPanFeature)
            let fused = MLX.concatenated([downsampled, fpnFeature], axis: -1)
            let newPanFeature = panBlock(fused)
            panFeatureMaps.append(newPanFeature)
        }

        return panFeatureMaps
    }
}

final class RtDetrV2AIFILayer: Module, UnaryLayer {

    @ModuleInfo(key: "layers") var layers: [RtDetrV2EncoderLayer]

    private let hiddenDim: Int
    private let temperature: Float

    init(_ config: RtDetrV2Config) {
        _layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            RtDetrV2EncoderLayer(config)
        }
        hiddenDim = config.encoderHiddenDim
        temperature = config.positionalEncodingTemperature
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batchSize, height, width, _) = hiddenStates.shape4

        var flattened = hiddenStates.flattened(start: 1, end: 2)
        let posEmbed = positionEmbedding(width: width, height: height, dtype: hiddenStates.dtype)

        for layer in layers {
            flattened = layer(
                hiddenStates: flattened,
                attentionMask: .none,
                positionEmbeddings: posEmbed
            )
        }

        return flattened.reshaped(batchSize, height, width, hiddenDim)
    }

    private func positionEmbedding(width: Int, height: Int, dtype: DType) -> MLXArray {
        let gridW = MLXArray(0..<width).asType(dtype)
        let gridH = MLXArray(0..<height).asType(dtype)

        let mesh = MLX.meshGrid([gridW, gridH], indexing: .xy)
        let meshW = mesh[0]
        let meshH = mesh[1]

        let embeddingDim = hiddenDim / 4
        var omega = MLXArray(0..<embeddingDim).asType(dtype)
        omega = omega / Float(embeddingDim)
        omega = 1 / (temperature ** omega)

        let outW = meshW.flattened().expandedDimensions(axis: -1) * omega.expandedDimensions(axis: 0)
        let outH = meshH.flattened().expandedDimensions(axis: -1) * omega.expandedDimensions(axis: 0)

        return MLX.concatenated(
            [
                outH.sin(),
                outH.cos(),
                outW.sin(),
                outW.cos(),
            ],
            axis: -1
        )
        .expandedDimensions(axis: 0)
    }
}

final class RtDetrV2EncoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttention: RtDetrV2SelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm

    @ModuleInfo(key: "fc1") var linear1: Linear
    @ModuleInfo(key: "fc2") var linear2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    private let normalizeBefore: Bool
    private let activation: UnaryLayer

    init(_ config: RtDetrV2Config) {
        _selfAttention.wrappedValue = RtDetrV2SelfAttention(
            hiddenSize: config.encoderHiddenDim,
            numAttentionHeads: config.encoderAttentionHeads,
            bias: true
        )
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.encoderHiddenDim, eps: config.layerNormEps)

        _linear1.wrappedValue = Linear(config.encoderHiddenDim, config.encoderFfnDim)
        _linear2.wrappedValue = Linear(config.encoderFfnDim, config.encoderHiddenDim)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.encoderHiddenDim, eps: config.layerNormEps)

        normalizeBefore = config.normalizeBefore
        activation = config.encoderActivationLayer
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: MLXArray?
    ) -> MLXArray {
        var hidden = hiddenStates

        if normalizeBefore {
            hidden = selfAttentionLayerNorm(hidden)
        }

        let attentionOutput = selfAttention(
            hiddenStates: hidden,
            attentionMask: attentionMask,
            positionEmbeddings: positionEmbeddings
        )

        hidden = hiddenStates + attentionOutput
        if !normalizeBefore {
            hidden = selfAttentionLayerNorm(hidden)
        }

        if normalizeBefore {
            hidden = finalLayerNorm(hidden)
        }

        let residual = hidden
        hidden = linear1(hidden)
        hidden = activation(hidden)
        hidden = linear2(hidden)
        hidden = residual + hidden

        if !normalizeBefore {
            hidden = finalLayerNorm(hidden)
        }

        return hidden
    }
}

final class RtDetrV2SelfAttention: Module {

    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    private let numHeads: Int

    init(
        hiddenSize: Int,
        numAttentionHeads: Int,
        bias: Bool
    ) {
        numHeads = numAttentionHeads
        _keyProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
        _valueProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
        _queryProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
        _outProjection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: MLXArray?
    ) -> MLXArray {
        let queryKeyInput = positionEmbeddings.map { hiddenStates + $0 } ?? hiddenStates

        let (batchSize, sequenceLength, _) = hiddenStates.shape3

        let queries = queryProjection(queryKeyInput)
            .reshaped(batchSize, sequenceLength, numHeads, -1)
            .transposed(0, 2, 1, 3)

        let keys = keyProjection(queryKeyInput)
            .reshaped(batchSize, sequenceLength, numHeads, -1)
            .transposed(0, 2, 1, 3)

        let values = valueProjection(hiddenStates)
            .reshaped(batchSize, sequenceLength, numHeads, -1)
            .transposed(0, 2, 1, 3)

        let scale = sqrt(1 / Float(queries.dim(-1)))
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return outProjection(output)
    }
}

final class RtDetrV2Decoder: Module {

    typealias Output = (
        hiddenStates: MLXArray,
        logits: MLXArray,
        boxes: MLXArray
    )

    @ModuleInfo(key: "layers") var layers: [RtDetrV2DecoderLayer]
    @ModuleInfo(key: "query_pos_head") var queryPosHead: RtDetrV2MLPPredictionHead

    // TODO: Extract shared decoder head wiring across RT-DETR/LW-DETR/RF-DETR.
    @ModuleInfo(key: "bbox_embed") var bboxEmbed: [RtDetrV2MLPPredictionHead]
    @ModuleInfo(key: "class_embed") var classEmbed: [Linear]

    init(_ config: RtDetrV2Config, numLabels: Int) {
        _layers.wrappedValue = (0..<config.decoderLayers).map { _ in
            RtDetrV2DecoderLayer(config)
        }

        _queryPosHead.wrappedValue = RtDetrV2MLPPredictionHead(
            inputDimensions: 4,
            hiddenDimensions: 2 * config.dModel,
            outputDimensions: config.dModel,
            layersCount: 2
        )

        _bboxEmbed.wrappedValue = (0..<config.decoderLayers).map { _ in
            RtDetrV2MLPPredictionHead(
                inputDimensions: config.dModel,
                hiddenDimensions: config.dModel,
                outputDimensions: 4,
                layersCount: 3
            )
        }

        _classEmbed.wrappedValue = (0..<config.decoderLayers).map { _ in
            Linear(config.dModel, numLabels)
        }
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        encoderHiddenStates: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: MLXArray,
        spatialShapesList: [(Int, Int)]
    ) -> Output {
        var hiddenStates = inputsEmbeds
        var currentReferencePoints = referencePoints.sigmoid()

        var lastLogits = classEmbed[0](hiddenStates)
        var lastBoxes = currentReferencePoints

        for (index, layer) in layers.enumerated() {
            let referencePointsInput = currentReferencePoints.expandedDimensions(axis: 2)
            let queryPosition = queryPosHead(currentReferencePoints)

            hiddenStates = layer(
                hiddenStates: hiddenStates,
                positionEmbeddings: queryPosition,
                referencePoints: referencePointsInput,
                spatialShapes: spatialShapes,
                spatialShapesList: spatialShapesList,
                encoderHiddenStates: encoderHiddenStates
            )

            let predictedCorners = bboxEmbed[index](hiddenStates)
            currentReferencePoints = (predictedCorners + currentReferencePoints.inverseSigmoid()).sigmoid()

            lastLogits = classEmbed[index](hiddenStates)
            lastBoxes = currentReferencePoints
        }

        return (
            hiddenStates: hiddenStates,
            logits: lastLogits,
            boxes: lastBoxes
        )
    }
}

final class RtDetrV2DecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttention: RtDetrV2SelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm

    @ModuleInfo(key: "encoder_attn") var encoderAttention: RtDetrV2MultiscaleDeformableAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttentionLayerNorm: LayerNorm

    @ModuleInfo(key: "fc1") var linear1: Linear
    @ModuleInfo(key: "fc2") var linear2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    private let activation: UnaryLayer

    init(_ config: RtDetrV2Config) {
        _selfAttention.wrappedValue = RtDetrV2SelfAttention(
            hiddenSize: config.dModel,
            numAttentionHeads: config.decoderAttentionHeads,
            bias: true
        )
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)

        _encoderAttention.wrappedValue = RtDetrV2MultiscaleDeformableAttention(config)
        _encoderAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)

        _linear1.wrappedValue = Linear(config.dModel, config.decoderFfnDim)
        _linear2.wrappedValue = Linear(config.decoderFfnDim, config.dModel)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)

        activation = config.decoderActivationLayer
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        positionEmbeddings: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: MLXArray,
        spatialShapesList: [(Int, Int)],
        encoderHiddenStates: MLXArray
    ) -> MLXArray {
        var hidden = hiddenStates

        let selfAttentionOutput = selfAttention(
            hiddenStates: hidden,
            attentionMask: .none,
            positionEmbeddings: positionEmbeddings
        )
        hidden = selfAttentionLayerNorm(hidden + selfAttentionOutput)

        let crossAttentionOutput = encoderAttention(
            hiddenStates: hidden,
            encoderHiddenStates: encoderHiddenStates,
            positionEmbeddings: positionEmbeddings,
            referencePoints: referencePoints,
            spatialShapes: spatialShapes,
            spatialShapesList: spatialShapesList
        )
        hidden = encoderAttentionLayerNorm(hidden + crossAttentionOutput)

        let mlpOutput = linear2(activation(linear1(hidden)))
        hidden = finalLayerNorm(hidden + mlpOutput)

        return hidden
    }
}

final class RtDetrV2MultiscaleDeformableAttention: Module {

    @ModuleInfo(key: "sampling_offsets") var samplingOffsets: Linear
    @ModuleInfo(key: "attention_weights") var attentionWeights: Linear
    @ModuleInfo(key: "value_proj") var valueProjection: Linear
    @ModuleInfo(key: "output_proj") var outputProjection: Linear

    @ParameterInfo(key: "n_points_scale") var nPointsScale: MLXArray

    private let nHeads: Int
    private let nLevels: Int
    private let nPoints: Int
    private let dModel: Int
    private let offsetScale: Float
    private let method: String

    init(_ config: RtDetrV2Config) {
        let decoderHeads = config.decoderAttentionHeads
        let decoderLevels = config.decoderNLevels
        let decoderPoints = config.decoderNPoints

        nHeads = decoderHeads
        nLevels = decoderLevels
        nPoints = decoderPoints
        dModel = config.dModel
        offsetScale = config.decoderOffsetScale
        method = config.decoderMethod

        _samplingOffsets.wrappedValue = Linear(config.dModel, decoderHeads * decoderLevels * decoderPoints * 2)
        _attentionWeights.wrappedValue = Linear(config.dModel, decoderHeads * decoderLevels * decoderPoints)
        _valueProjection.wrappedValue = Linear(config.dModel, config.dModel)
        _outputProjection.wrappedValue = Linear(config.dModel, config.dModel)

        let nPointsScale = (0..<decoderLevels).flatMap { _ in
            Array(repeating: 1 / Float(decoderPoints), count: decoderPoints)
        }
        _nPointsScale.wrappedValue = nPointsScale.asMLXArray(dtype: .float32)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        positionEmbeddings: MLXArray,
        referencePoints: MLXArray,
        spatialShapes: MLXArray,
        spatialShapesList: [(Int, Int)]
    ) -> MLXArray {
        let hiddenWithPosition = hiddenStates + positionEmbeddings
        let (batchSize, queryLength, _) = hiddenWithPosition.shape3

        let value = valueProjection(encoderHiddenStates)
            .reshaped(batchSize, encoderHiddenStates.shape[1], nHeads, dModel / nHeads)

        let samplingOffsets = samplingOffsets(hiddenWithPosition)
            .reshaped(batchSize, queryLength, nHeads, nLevels * nPoints, 2)

        let attentionWeights = attentionWeights(hiddenWithPosition)
            .reshaped(batchSize, queryLength, nHeads, nLevels * nPoints)
            .softmax(axis: -1)

        let samplingLocations: MLXArray
        if referencePoints.shape[referencePoints.ndim - 1] == 4 {
            let scale = nPointsScale.asType(hiddenWithPosition.dtype)
                .reshaped(1, 1, 1, nLevels * nPoints, 1)
            let wh = referencePoints[0..., 0..., 0..., 2...]
                .expandedDimensions(axis: 3)
            let center = referencePoints[0..., 0..., 0..., 0..<2]
                .expandedDimensions(axis: 3)

            samplingLocations = center + (samplingOffsets * scale * wh * offsetScale)
        } else {
            let offsetNormalizer = MLX.stacked([spatialShapes[0..., 1], spatialShapes[0..., 0]], axis: -1)
                .asType(hiddenWithPosition.dtype)
                .reshaped(1, 1, 1, nLevels, 1, 2)

            let expandedOffsets = samplingOffsets.reshaped(batchSize, queryLength, nHeads, nLevels, nPoints, 2)
            samplingLocations =
                (referencePoints.expandedDimensions(axis: 2).expandedDimensions(axis: 4)
                + expandedOffsets / offsetNormalizer).reshaped(batchSize, queryLength, nHeads, nLevels * nPoints, 2)
        }

        let output = multiScaleDeformableAttention(
            value: value,
            spatialShapesList: spatialShapesList,
            samplingLocations: samplingLocations,
            attentionWeights: attentionWeights,
            method: method
        )

        return outputProjection(output)
    }

    private func multiScaleDeformableAttention(
        value: MLXArray,
        spatialShapesList: [(Int, Int)],
        samplingLocations: MLXArray,
        attentionWeights: MLXArray,
        method: String
    ) -> MLXArray {
        let (batchSize, _, _, headDim) = value.shape4
        let queryLength = samplingLocations.shape[1]
        let pointsPerLevel = nPoints

        var sampledByLevel: [MLXArray] = []
        var weightByLevel: [MLXArray] = []

        var offset = 0
        for (levelIndex, (height, width)) in spatialShapesList.enumerated() {
            let levelLength = height * width
            let levelValue = value[0..., offset..<(offset + levelLength), 0..., 0...]
            offset += levelLength

            let valueMap =
                levelValue
                .reshaped(batchSize, height, width, nHeads, headDim)
                .transposed(0, 3, 1, 2, 4)
                .reshaped(batchSize * nHeads, height, width, headDim)

            let start = levelIndex * pointsPerLevel
            let end = start + pointsPerLevel

            let levelLocations = samplingLocations[0..., 0..., 0..., start..<end, 0...]
                .transposed(0, 2, 1, 3, 4)
                .reshaped(batchSize * nHeads, queryLength, pointsPerLevel, 2)

            let sampled: MLXArray
            if method == "discrete" {
                sampled = discreteSample(featureMap: valueMap, samplingLocations: levelLocations)
            } else {
                sampled = bilinearSample(featureMap: valueMap, samplingLocations: levelLocations)
            }
            sampledByLevel.append(sampled)

            let levelWeights = attentionWeights[0..., 0..., 0..., start..<end]
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize * nHeads, queryLength, pointsPerLevel)
            weightByLevel.append(levelWeights)
        }

        let sampledValues = MLX.concatenated(sampledByLevel, axis: 2)
        let allWeights = MLX.concatenated(weightByLevel, axis: 2)

        return (sampledValues * allWeights.expandedDimensions(axis: -1))
            .sum(axis: 2)
            .reshaped(batchSize, nHeads, queryLength, headDim)
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, queryLength, dModel)
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

    private func discreteSample(featureMap: MLXArray, samplingLocations: MLXArray) -> MLXArray {
        let (_, height, width, channels) = featureMap.shape4

        let samplingCoordinates = (samplingLocations * [Float(width), Float(height)].asMLXArray(dtype: samplingLocations.dtype) + 0.5)
            .asType(.int32)

        let samplingX = MLX.clip(samplingCoordinates[0..., 0..., 0..., 0], min: 0, max: width - 1)
        let samplingY = MLX.clip(samplingCoordinates[0..., 0..., 0..., 1], min: 0, max: height - 1)

        let flattenedFeatures = featureMap.reshaped(featureMap.shape[0], height * width, channels)
        let flattenedGlobalFeatures = flattenedFeatures.reshaped(-1, channels)

        let samplingIndex = (samplingY * width + samplingX).asType(.int32)
        let spatialCount = MLXArray(height * width, dtype: .int32)
        let batchOffsets = (MLXArray(0..<featureMap.shape[0]).asType(.int32) * spatialCount)
            .reshaped(-1, 1, 1)

        return MLX.take(flattenedGlobalFeatures, samplingIndex + batchOffsets, axis: 0)
    }
}

final class RtDetrV2MLPPredictionHead: Module, UnaryLayer {

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

private extension RtDetrResNetConfig {

    var activationLayer: UnaryLayer {
        switch hiddenAct {
        case "gelu":
            GELU(approximation: .none)
        case "silu":
            SiLU()
        default:
            ReLU()
        }
    }
}

private extension RtDetrV2Config {

    var activationLayer: UnaryLayer {
        switch activationFunction {
        case "gelu":
            GELU(approximation: .none)
        case "relu":
            ReLU()
        default:
            SiLU()
        }
    }

    var encoderActivationLayer: UnaryLayer {
        switch encoderActivationFunction {
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
