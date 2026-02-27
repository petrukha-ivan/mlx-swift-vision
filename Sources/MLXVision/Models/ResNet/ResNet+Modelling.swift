//
//  ResNet+Modelling.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 02.11.2025.
//

import MLX
import MLXNN
import ReerCodable

final class ResNetModelForImageClassification: Module, Predictor {

    @ModuleInfo var resnet: ResNetModel
    @ModuleInfo var classifier: Linear

    init(_ config: ResNetForImageClassificationConfig) {
        resnet = ResNetModel(config)
        classifier = Linear(config.hiddenSizes.last!, config.id2label.count)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing("classifier.1", with: "classifier")
        }
    }

    func predict(_ input: ImageInput) throws -> MLXArray {
        let hidden = resnet(input.pixelValues)
        let logits = classifier(hidden).squeezed()
        return logits
    }
}

final class ResNetModel: Module {

    let embedder: ResNetEmbedder
    let encoder: ResNetEncoder
    let pooler: AdaptiveAvgPool2d

    init(_ config: ResNetConfig) {
        embedder = ResNetEmbedder(config)
        encoder = ResNetEncoder(config)
        pooler = AdaptiveAvgPool2d(outputSize: 1)
    }

    override func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        parameters.renameKeys {
            $0.replacing(/^conv1\.([a-z_]+)$/) {
                "embedder.embedder.convolution.\($0.1)"
            }
            .replacing(/^bn1\.([a-z_]+)$/) {
                "embedder.embedder.normalization.\($0.1)"
            }
            .replacing(/^layer(\d+)\.(\d+)\.conv(\d+)\.([a-z_]+)$/) {
                "encoder.stages.\(Int($0.1)! - 1).layers.\($0.2).layers.\(Int($0.3)! - 1).convolution.\($0.4)"
            }
            .replacing(/^layer(\d+)\.(\d+)\.bn(\d+)\.([a-z_]+)$/) {
                "encoder.stages.\(Int($0.1)! - 1).layers.\($0.2).layers.\(Int($0.3)! - 1).normalization.\($0.4)"
            }
            .replacing(/^layer(\d+)\.(\d+)\.downsample\.0\.([a-z_]+)$/) {
                "encoder.stages.\(Int($0.1)! - 1).layers.\($0.2).shortcut.convolution.\($0.3)"
            }
            .replacing(/^layer(\d+)\.(\d+)\.downsample\.1\.([a-z_]+)$/) {
                "encoder.stages.\(Int($0.1)! - 1).layers.\($0.2).shortcut.normalization.\($0.3)"
            }
            .replacing("layer.", with: "layers.")
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let embeddings = embedder(x)
        let encodings = encoder(embeddings)
        let pooled = pooler(encodings)
        return pooled
    }

    func features(_ pixelValues: MLXArray) -> [MLXArray] {
        let embeddings = embedder(pixelValues)
        var features = [MLXArray]()
        for stage in encoder.stages {
            features.append(stage(features.last ?? embeddings))
        }
        return features
    }
}

final class ResNetEmbedder: Module {

    let embedder: ResNetConvLayer
    let pooler: PaddedMaxPool2d

    init(_ config: ResNetConfig) {
        embedder = ResNetConvLayer(inputChannels: config.numChannels, outputChannels: config.embeddingSize, kernelSize: 7, stride: 2)
        pooler = PaddedMaxPool2d(kernelSize: 3, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = embedder(x)
        y = pooler(y)
        return y
    }
}

final class ResNetEncoder: Module {

    let stages: [UnaryLayer]

    init(_ config: ResNetConfig) {
        stages =
            [
                ResNetStage(
                    config: config,
                    inputChannels: config.embeddingSize,
                    outputChannels: config.hiddenSizes[0],
                    stride: config.downsampleInFirstStage ? 2 : 1,
                    depth: config.depths[0]
                )
            ]
            + zip(zip(config.hiddenSizes, config.hiddenSizes.dropFirst()), config.depths.dropFirst()).map {
                ResNetStage(
                    config: config,
                    inputChannels: $0.0,
                    outputChannels: $0.1,
                    depth: $1
                )
            }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return stages(x)
    }
}

final class ResNetStage: Module, UnaryLayer {

    let layers: [UnaryLayer]

    init(
        config: ResNetConfig,
        inputChannels: Int,
        outputChannels: Int,
        stride: Int = 2,
        depth: Int = 2
    ) {
        layers =
            [
                ResNetBottleneck(
                    config: config,
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    stride: stride
                )
            ]
            + (0..<depth - 1).map { _ in
                ResNetBottleneck(
                    config: config,
                    inputChannels: outputChannels,
                    outputChannels: outputChannels
                )
            }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return layers(x)
    }
}

final class ResNetConvLayer: Module, UnaryLayer {

    let convolution: Conv2d
    let normalization: BatchNorm
    let activation: UnaryLayer

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        activation: UnaryLayer = ReLU()
    ) {
        convolution = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(kernelSize / 2),
            bias: false
        )
        normalization = BatchNorm(featureCount: outputChannels)
        self.activation = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = convolution(x)
        hidden = normalization(hidden)
        hidden = activation(hidden)
        return hidden
    }
}

final class ResNetBottleneck: Module, UnaryLayer {

    let shortcut: UnaryLayer
    let layers: [UnaryLayer]
    let activation: UnaryLayer

    init(
        config: ResNetConfig,
        inputChannels: Int,
        outputChannels: Int,
        stride: Int = 1,
        reduction: Int = 4
    ) {
        let reducedChannels = outputChannels / reduction
        let shouldApplyShortcut = (inputChannels != outputChannels) || (stride != 1)
        activation = ReLU()
        shortcut =
            shouldApplyShortcut ? ResNetShortcut(inputChannels: inputChannels, outputChannels: outputChannels, stride: stride) : Identity()
        layers = [
            ResNetConvLayer(
                inputChannels: inputChannels,
                outputChannels: reducedChannels,
                kernelSize: 1,
                stride: config.downsampleInBottleneck ? stride : 1
            ),
            ResNetConvLayer(
                inputChannels: reducedChannels,
                outputChannels: reducedChannels,
                stride: config.downsampleInBottleneck ? 1 : stride
            ),
            ResNetConvLayer(inputChannels: reducedChannels, outputChannels: outputChannels, kernelSize: 1, activation: Identity()),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = shortcut(x)
        var hidden = layers(x)
        hidden += residual
        hidden = activation(hidden)
        return hidden
    }
}

final class ResNetShortcut: Module, UnaryLayer {

    let convolution: Conv2d
    let normalization: BatchNorm

    init(
        inputChannels: Int,
        outputChannels: Int,
        stride: Int = 1,
    ) {
        convolution = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 1,
            stride: IntOrPair(stride),
            bias: false
        )
        normalization = BatchNorm(featureCount: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = convolution(x)
        hidden = normalization(hidden)
        return hidden
    }
}
