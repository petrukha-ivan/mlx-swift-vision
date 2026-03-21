//
//  Sam3+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import ReerCodable

@Codable
@CodingContainer("detector_config")
struct Sam3Config {

    @CodingKey("vision_config")
    var visionConfig: Sam3VisionConfig

    @CodingKey("text_config")
    var textConfig: CLIPTextConfig

    @CodingKey("geometry_encoder_config")
    var geometryEncoderConfig: Sam3GeometryEncoderConfig

    @CodingKey("detr_encoder_config")
    var detrEncoderConfig: Sam3DetrEncoderConfig

    @CodingKey("detr_decoder_config")
    var detrDecoderConfig: Sam3DetrDecoderConfig

    @CodingKey("mask_decoder_config")
    var maskDecoderConfig: Sam3MaskDecoderConfig
}

@Codable
struct Sam3VisionConfig {

    @CodingKey("backbone_config")
    var backboneConfig: Sam3ViTConfig

    @CodingKey("scale_factors")
    var scaleFactors: [Float]

    @CodingKey("fpn_hidden_size")
    var fpnHiddenSize: Int
}

@Codable
struct Sam3ViTConfig {

    @CodingKey("num_channels")
    var numChannels: Int

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("intermediate_size")
    var intermediateSize: Int

    @CodingKey("num_hidden_layers")
    var numHiddenLayers: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int

    @CodingKey("patch_size")
    var patchSize: Int

    @CodingKey("image_size")
    var imageSize: Int

    @CodingKey("pretrain_image_size")
    var pretrainImageSize: Int

    @CodingKey("window_size")
    var windowSize: Int

    @CodingKey("global_attn_indexes")
    var globalAttnIndices: [Int]

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float

    @CodingKey("rope_theta")
    var ropeTheta: Float
}

@Codable
struct Sam3GeometryEncoderConfig {

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("intermediate_size")
    var intermediateSize: Int

    @CodingKey("roi_size")
    var roiSize: Int

    @CodingKey("num_layers")
    var numLayers: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int
}

@Codable
struct Sam3DetrEncoderConfig {

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("intermediate_size")
    var intermediateSize: Int

    @CodingKey("num_layers")
    var numLayers: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int
}

@Codable
struct Sam3DetrDecoderConfig {

    @CodingKey("num_layers")
    var numLayers: Int

    @CodingKey("num_queries")
    var numQueries: Int

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("intermediate_size")
    var intermediateSize: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int
}

@Codable
struct Sam3MaskDecoderConfig {

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int

    @CodingKey("num_upsampling_stages")
    var numUpsamplingStages: Int
}

extension Sam3Config {
    func with(overrides: ModelOverrides) -> Sam3Config {
        guard let inputSize = overrides.inputSize else {
            return self
        }

        let width = Int(inputSize.width)
        let height = Int(inputSize.height)
        precondition(width == height)
        precondition(width % (visionConfig.backboneConfig.patchSize * visionConfig.backboneConfig.windowSize) == 0)

        var config = self
        config.visionConfig.backboneConfig.imageSize = width
        return config
    }
}
