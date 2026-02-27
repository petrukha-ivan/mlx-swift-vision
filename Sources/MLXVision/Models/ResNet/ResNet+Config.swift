//
//  ResNet+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.11.2025.
//

import ReerCodable

@Codable
class ResNetConfig {

    @CodingKey("num_channels")
    var numChannels: Int = 3

    @CodingKey("embedding_size")
    var embeddingSize: Int = 64

    @CodingKey("hidden_sizes")
    var hiddenSizes: [Int] = [256, 512, 1024, 2048]

    @CodingKey("depths")
    var depths: [Int] = [3, 4, 6, 3]

    @CodingKey("downsample_in_first_stage")
    var downsampleInFirstStage: Bool = false

    @CodingKey("downsample_in_bottleneck")
    var downsampleInBottleneck: Bool = false
}

@InheritedCodable
final class ResNetForImageClassificationConfig: ResNetConfig {

    @CodingKey("label2id")
    var label2id: [String: Int]

    @CodingKey("id2label")
    var id2label: [Int: String]
}

extension ResNetConfig {

    static var all: [String: ResNetConfig] {
        return [
            "resnet50": ResNetConfig(depths: [3, 4, 6, 3]),
            "resnet101": ResNetConfig(depths: [3, 4, 23, 3]),
            "resnet152": ResNetConfig(depths: [3, 8, 36, 3]),
        ]
    }
}
