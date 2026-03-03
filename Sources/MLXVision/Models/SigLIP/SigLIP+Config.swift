//
//  SigLIP+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import ReerCodable

@Codable
struct SigLIPConfig {

    @CodingKey("text_config")
    var textConfig: SigLIPTextConfig

    @CodingKey("vision_config")
    var visionConfig: SigLIPVisionConfig
}

@Codable
struct SigLIPTextConfig: CLIPSharedConfig {

    @CodingKey("vocab_size")
    var vocabSize: Int = 32_000

    @CodingKey("hidden_size")
    var hiddenSize: Int = 768

    @CodingKey("intermediate_size")
    var intermediateSize: Int = 3_072

    @CodingKey("num_hidden_layers")
    var numHiddenLayers: Int = 12

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int = 12

    @CodingKey("max_position_embeddings")
    var maxPositionEmbeddings: Int = 64

    @CodingKey("projection_size")
    var projectionSize: Int?

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float = 1e-6
}

@Codable
struct SigLIPVisionConfig: CLIPSharedConfig {

    @CodingKey("hidden_size")
    var hiddenSize: Int = 768

    @CodingKey("intermediate_size")
    var intermediateSize: Int = 3_072

    @CodingKey("num_hidden_layers")
    var numHiddenLayers: Int = 12

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int = 12

    @CodingKey("num_channels")
    var numChannels: Int = 3

    @CodingKey("patch_size")
    var patchSize: Int = 16

    @CodingKey("image_size")
    var imageSize: Int = 224

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float = 1e-6
}
