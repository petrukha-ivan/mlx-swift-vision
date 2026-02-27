//
//  EfficientNet+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import ReerCodable

@Codable
class EfficientNetConfig {

    @Codable
    enum PoolingType: String {
        case mean
        case max
    }

    @CodingKey("pooling_type")
    var poolingType: PoolingType

    @CodingKey("in_channels")
    var inChannels: [Int]

    @CodingKey("out_channels")
    var outChannels: [Int]

    @CodingKey("num_channels")
    var numChannels: Int

    @CodingKey("num_block_repeats")
    var numBlockRepeats: [Int]

    @CodingKey("kernel_sizes")
    var kernelSizes: [Int]

    @CodingKey("expand_ratios")
    var expandRatios: [Int]

    @CodingKey("strides")
    var strides: [Int]

    @CodingKey("hidden_dim")
    var hiddenDim: Int

    @CodingKey("width_coefficient")
    var widthCoefficient: Float

    @CodingKey("depth_coefficient")
    var depthCoefficient: Float

    @CodingKey("depth_divisor")
    var depthDivisor: Float

    @CodingKey("squeeze_expansion_ratio")
    var squeezeExpansionRatio: Float

    @CodingKey("batch_norm_eps")
    var batchNormEps: Float

    @CodingKey("batch_norm_momentum")
    var batchNormMomentum: Float
}

@InheritedCodable
final class EfficientNetForImageClassificationConfig: EfficientNetConfig {

    @CodingKey("label2id")
    var label2id: [String: Int]

    @CodingKey("id2label")
    var id2label: [Int: String]
}
