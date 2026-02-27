//
//  Detr+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.11.2025.
//

import ReerCodable

@Codable
class DetrConfig {

    @Codable
    enum PositionEmbeddingType: String {
        case sine
        case learned
    }

    @CodingKey("backbone")
    var backbone: String

    @CodingKey("d_model")
    var dimensions: Int

    @CodingKey("encoder_layers")
    var encoderLayersCount: Int

    @CodingKey("encoder_ffn_dim")
    var encoderFeedforwardDimensions: Int

    @CodingKey("decoder_layers")
    var decoderLayers: Int

    @CodingKey("decoder_ffn_dim")
    var decoderFeedforwardDimensions: Int

    @CodingKey("decoder_attention_heads")
    var decoderAttentionHeads: Int

    @CodingKey("num_queries")
    var queriesCount: Int

    @CodingKey("position_embedding_type")
    var positionEmbeddingType: PositionEmbeddingType = .sine

    @CodingKey("position_embedding_max_size")
    var positionEmbeddingMaxSize: Int = 50
}

@InheritedCodable
final class DetrForObjectDetectionConfig: DetrConfig {

    @CodingKey("id2label")
    var id2label: [Int: String]
}
