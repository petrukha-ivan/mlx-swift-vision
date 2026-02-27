//
//  CLIP+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import ReerCodable

@Codable
struct CLIPConfig {

    @CodingKey("text_config")
    var textConfig: CLIPTextConfig

    @CodingKey("vision_config")
    var visionConfig: CLIPVisionConfig

    @CodingKey("logit_scale_init_value")
    var logitScaleInitValue: Float

    @CodingKey("projection_dim")
    var projectionDim: Int
}

@Codable
struct CLIPTextConfig: CLIPSharedConfig {

    @CodingKey("vocab_size")
    var vocabSize: Int

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("intermediate_size")
    var intermediateSize: Int

    @CodingKey("num_hidden_layers")
    var numHiddenLayers: Int

    @CodingKey("max_position_embeddings")
    var maxPositionEmbeddings: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int

    @CodingKey("projection_dim")
    var projectionDim: Int

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float
}

@Codable
struct CLIPVisionConfig: CLIPSharedConfig {

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("intermediate_size")
    var intermediateSize: Int

    @CodingKey("num_hidden_layers")
    var numHiddenLayers: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int

    @CodingKey("num_channels")
    var numChannels: Int = 3

    @CodingKey("patch_size")
    var patchSize: Int

    @CodingKey("image_size")
    var imageSize: Int

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float
}

protocol CLIPSharedConfig {
    var hiddenSize: Int { get }
    var intermediateSize: Int { get }
    var numHiddenLayers: Int { get }
    var numAttentionHeads: Int { get }
    var layerNormEps: Float { get }
}
