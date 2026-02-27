//
//  EfficientNet+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import Foundation
import MLXNN
import MLX

final class EfficientNetModelForImageClassification: Module, Predictor {

    let efficientnet: EfficientNetModel
    let classifier: Linear

    init(_ config: EfficientNetForImageClassificationConfig) {
        efficientnet = EfficientNetModel(config)
        classifier = Linear(config.hiddenDim, config.id2label.count)
    }

    func predict(_ input: ImageInput) throws -> MLXArray {
        let output = efficientnet(input.pixelValues)
        let logits = classifier(output).squeezed()
        return logits
    }
}

final class EfficientNetModel: Module {

    let embeddings: EfficientNetEmbeddings
    let encoder: EfficientNetEncoder
    let pooler: UnaryLayer

    init(_ config: EfficientNetConfig) {
        embeddings = EfficientNetEmbeddings(config)
        encoder = EfficientNetEncoder(config)
        pooler =
            switch config.poolingType {
            case .mean:
                AdaptiveAvgPool2d(outputSize: 1)
            case .max:
                AdaptiveMaxPool2d(outputSize: 1)
            }
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let embeddingsOutput = embeddings(pixelValues)
        let encoderOutput = encoder(embeddingsOutput)
        let pooledOutput = pooler(encoderOutput).squeezed(axes: [1, 2])
        return pooledOutput
    }
}

final class EfficientNetEmbeddings: Module {

    @ModuleInfo var padding: Padding2d
    @ModuleInfo var convolution: Conv2d
    @ModuleInfo(key: "batchnorm") var normalization: BatchNorm
    @ModuleInfo var activation: UnaryLayer

    init(_ config: EfficientNetConfig) {
        padding = Padding2d(padding: IntOrPair((0, 1)), paddingValue: 0)
        _convolution.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.roundFilters(numChannels: 32),
            kernelSize: 3,
            stride: 2,
            bias: false
        )
        _normalization.wrappedValue = BatchNorm(
            featureCount: config.roundFilters(numChannels: 32),
            eps: config.batchNormEps,
            momentum: config.batchNormMomentum
        )
        _activation.wrappedValue = SiLU()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var features = padding(pixelValues)
        features = convolution(features)
        features = normalization(features)
        features = activation(features)
        return features
    }
}

final class EfficientNetEncoder: Module {

    @ModuleInfo var blocks: [EfficientNetBlock]
    @ModuleInfo(key: "top_conv") var convolution: Conv2d
    @ModuleInfo(key: "top_bn") var normalization: BatchNorm
    @ModuleInfo var activation: UnaryLayer

    init(_ config: EfficientNetConfig) {
        blocks = zip(
            config.inChannels,
            config.outChannels,
            config.strides,
            config.kernelSizes,
            config.expandRatios,
            config.numBlockRepeats
        ).flatMap { inDim, outDim, stride, kernelSize, expandRatio, numBlockRepeats in
            (0..<numBlockRepeats).map {
                EfficientNetBlock(
                    config,
                    config.roundFilters(numChannels: $0 > 0 ? outDim : inDim),
                    config.roundFilters(numChannels: outDim),
                    $0 > 0 ? 1 : stride,
                    kernelSize,
                    expandRatio,
                    $0 > 0
                )
            }
        }
        _convolution.wrappedValue = Conv2d(
            inputChannels: config.roundFilters(numChannels: config.outChannels.last!),
            outputChannels: config.roundFilters(numChannels: 1280),
            kernelSize: 1,
            bias: false
        )
        _normalization.wrappedValue = BatchNorm(
            featureCount: config.hiddenDim,
            eps: config.batchNormEps,
            momentum: config.batchNormMomentum
        )
        activation = SiLU()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = pixelValues

        for block in blocks {
            hidden = block(hidden)
        }

        hidden = convolution(hidden)
        hidden = normalization(hidden)
        hidden = activation(hidden)

        return hidden
    }
}

final class EfficientNetBlock: Module {

    let expansion: EfficientNetExpansionLayer?
    @ModuleInfo(key: "depthwise_conv") var depthwiseLayer: EfficientNetDepthwiseLayer
    @ModuleInfo(key: "squeeze_excite") var squeezeExcite: EfficientNetSqueezeExciteLayer
    let projection: EfficientNetFinalBlockLayer

    let applyResidual: Bool

    init(
        _ config: EfficientNetConfig,
        _ inDim: Int,
        _ outDim: Int,
        _ stride: Int,
        _ kernelSize: Int,
        _ expandRatio: Int,
        _ residual: Bool
    ) {
        applyResidual = residual
        let expand = expandRatio > 1
        let expandedChannels = inDim * expandRatio
        expansion = expand ? EfficientNetExpansionLayer(config, inDim, expandedChannels) : nil
        _depthwiseLayer.wrappedValue = EfficientNetDepthwiseLayer(
            config,
            expand ? expandedChannels : inDim,
            expand ? expandedChannels : inDim,
            stride,
            kernelSize
        )
        _squeezeExcite.wrappedValue = EfficientNetSqueezeExciteLayer(
            config,
            expand ? expandedChannels : inDim,
            Int(Float(inDim) * config.squeezeExpansionRatio),
            expand ? expandedChannels : inDim
        )
        projection = EfficientNetFinalBlockLayer(config, expand ? expandedChannels : inDim, outDim)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden
        let residual = hidden

        if let expansion {
            hidden = expansion(hidden)
        }

        hidden = depthwiseLayer(hidden)
        hidden = squeezeExcite(hidden)
        hidden = projection(hidden)

        if applyResidual {
            hidden += residual
        }

        return hidden
    }
}

final class EfficientNetExpansionLayer: Module {

    @ModuleInfo(key: "expand_conv") var convolution: Conv2d
    @ModuleInfo(key: "expand_bn") var normalization: BatchNorm
    @ModuleInfo var activation: UnaryLayer

    init(
        _ config: EfficientNetConfig,
        _ inputChannels: Int,
        _ outputChannels: Int,
    ) {
        _convolution.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 1, bias: false)
        _normalization.wrappedValue = BatchNorm(featureCount: outputChannels, eps: config.batchNormEps, momentum: config.batchNormMomentum)
        _activation.wrappedValue = SiLU()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden

        hidden = convolution(hidden)
        hidden = normalization(hidden)
        hidden = activation(hidden)

        return hidden
    }
}

final class EfficientNetDepthwiseLayer: Module {

    @ModuleInfo var padding: Padding2d?
    @ModuleInfo(key: "depthwise_conv") var convolution: Conv2d
    @ModuleInfo(key: "depthwise_norm") var normalization: BatchNorm
    @ModuleInfo var activation: UnaryLayer

    init(
        _ config: EfficientNetConfig,
        _ inputChannels: Int,
        _ outputChannels: Int,
        _ stride: Int,
        _ kernelSize: Int,
    ) {
        _padding.wrappedValue = stride == 2 ? Padding2d(padding: IntOrPair((kernelSize / 2 - 1, kernelSize / 2)), paddingValue: 0) : nil
        _convolution.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: .init(kernelSize),
            stride: .init(stride),
            padding: stride == 2 ? IntOrPair((0, 0)) : computeSamePadding(input: inputChannels, kernel: kernelSize, stride: stride),
            groups: inputChannels,
            bias: false
        )
        _normalization.wrappedValue = BatchNorm(featureCount: outputChannels, eps: config.batchNormEps, momentum: config.batchNormMomentum)
        _activation.wrappedValue = SiLU()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden

        if let padding {
            hidden = padding(hidden)
        }

        hidden = convolution(hidden)
        hidden = normalization(hidden)
        hidden = activation(hidden)

        return hidden
    }
}

final class EfficientNetSqueezeExciteLayer: Module {

    @ModuleInfo var squeeze: AdaptiveAvgPool2d
    @ModuleInfo var reduce: Conv2d
    @ModuleInfo var expand: Conv2d
    @ModuleInfo var activation: UnaryLayer

    init(
        _ config: EfficientNetConfig,
        _ inputChannels: Int,
        _ hiddenChannels: Int,
        _ outputChannels: Int
    ) {
        _squeeze.wrappedValue = AdaptiveAvgPool2d(outputSize: 1)
        _reduce.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: hiddenChannels, kernelSize: 1)
        _expand.wrappedValue = Conv2d(inputChannels: hiddenChannels, outputChannels: outputChannels, kernelSize: 1)
        _activation.wrappedValue = SiLU()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden
        let residual = hidden

        hidden = squeeze(hidden)
        hidden = reduce(hidden)
        hidden = activation(hidden)

        hidden = expand(hidden)
        hidden = sigmoid(hidden)
        hidden *= residual

        return hidden
    }
}

final class EfficientNetFinalBlockLayer: Module {

    @ModuleInfo(key: "project_conv") var convolution: Conv2d
    @ModuleInfo(key: "project_bn") var normalization: BatchNorm

    init(
        _ config: EfficientNetConfig,
        _ inputChannels: Int,
        _ outputChannels: Int
    ) {
        _convolution.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 1, bias: false)
        _normalization.wrappedValue = BatchNorm(featureCount: outputChannels, eps: config.batchNormEps, momentum: config.batchNormMomentum)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var hidden = hidden
        hidden = convolution(hidden)
        hidden = normalization(hidden)
        return hidden
    }
}

private extension EfficientNetConfig {
    func roundFilters(numChannels: Int) -> Int {
        let divisor = Int(depthDivisor)
        let scaled = Float(numChannels) * widthCoefficient
        let divisorD = Float(divisor)
        let newDim = max(divisor, Int((scaled + divisorD / 2.0) / divisorD) * divisor)
        if Float(newDim) < 0.9 * scaled {
            return newDim + Int(divisor)
        } else {
            return newDim
        }
    }
}

private func computeSamePadding(input: Int, kernel: Int, stride: Int = 1, dilation: Int = 1) -> IntOrPair {
    let kernel = (kernel - 1) * dilation + 1
    let outSize = Int(ceil(Double(input) / Double(stride)))
    let total = max(0, (outSize - 1) * stride + kernel - input)
    let before = total / 2
    let after = total - before
    return IntOrPair((before, after))
}
