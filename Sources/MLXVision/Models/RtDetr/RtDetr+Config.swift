//
//  RtDetr+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.03.2026.
//

import ReerCodable

@Codable
class RtDetrResNetConfig {

    @CodingKey("num_channels")
    var numChannels: Int = 3

    @CodingKey("embedding_size")
    var embeddingSize: Int = 64

    @CodingKey("hidden_sizes")
    var hiddenSizes: [Int] = [256, 512, 1024, 2048]

    @CodingKey("depths")
    var depths: [Int] = [3, 4, 6, 3]

    @CodingKey("layer_type")
    var layerType: String = "bottleneck"

    @CodingKey("hidden_act")
    var hiddenAct: ActivationType = .relu

    @CodingKey("downsample_in_first_stage")
    var downsampleInFirstStage: Bool = false

    @CodingKey("downsample_in_bottleneck")
    var downsampleInBottleneck: Bool = false

    @CodingKey("out_indices")
    var outIndices: [Int] = [2, 3, 4]
}

@Codable
class RtDetrV2Config {

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float = 1e-5

    @CodingKey("batch_norm_eps")
    var batchNormEps: Float = 1e-5

    @CodingKey("backbone_config")
    var backboneConfig: RtDetrResNetConfig

    @CodingKey("freeze_backbone_batch_norms")
    var freezeBackboneBatchNorms: Bool = true

    @CodingKey("encoder_hidden_dim")
    var encoderHiddenDim: Int = 256

    @CodingKey("encoder_in_channels")
    var encoderInChannels: [Int] = [512, 1024, 2048]

    @CodingKey("feat_strides")
    var featStrides: [Int] = [8, 16, 32]

    @CodingKey("encoder_layers")
    var encoderLayers: Int = 1

    @CodingKey("encoder_ffn_dim")
    var encoderFfnDim: Int = 1024

    @CodingKey("encoder_attention_heads")
    var encoderAttentionHeads: Int = 8

    @CodingKey("encode_proj_layers")
    var encodeProjLayers: [Int] = [2]

    @CodingKey("positional_encoding_temperature")
    var positionalEncodingTemperature: Float = 10_000

    @CodingKey("encoder_activation_function")
    var encoderActivationFunction: ActivationType = .gelu

    @CodingKey("activation_function")
    var activationFunction: ActivationType = .silu

    @CodingKey("eval_size")
    var evalSize: [Int]? = nil

    @CodingKey("normalize_before")
    var normalizeBefore: Bool = false

    @CodingKey("hidden_expansion")
    var hiddenExpansion: Float = 1

    @CodingKey("d_model")
    var dModel: Int = 256

    @CodingKey("num_queries")
    var numQueries: Int = 300

    @CodingKey("decoder_in_channels")
    var decoderInChannels: [Int] = [256, 256, 256]

    @CodingKey("decoder_ffn_dim")
    var decoderFfnDim: Int = 1024

    @CodingKey("num_feature_levels")
    var numFeatureLevels: Int = 3

    @CodingKey("decoder_n_points")
    var decoderNPoints: Int = 4

    @CodingKey("decoder_layers")
    var decoderLayers: Int = 6

    @CodingKey("decoder_attention_heads")
    var decoderAttentionHeads: Int = 8

    @CodingKey("decoder_activation_function")
    var decoderActivationFunction: ActivationType = .relu

    @CodingKey("num_denoising")
    var numDenoising: Int = 100

    @CodingKey("label_noise_ratio")
    var labelNoiseRatio: Float = 0.5

    @CodingKey("box_noise_scale")
    var boxNoiseScale: Float = 1

    @CodingKey("learn_initial_query")
    var learnInitialQuery: Bool = false

    @CodingKey("anchor_image_size")
    var anchorImageSize: [Int]? = nil

    @CodingKey("with_box_refine")
    var withBoxRefine: Bool = true

    @CodingKey("decoder_n_levels")
    var decoderNLevels: Int = 3

    @CodingKey("decoder_offset_scale")
    var decoderOffsetScale: Float = 0.5

    @CodingKey("decoder_method")
    var decoderMethod: String = "default"
}

@InheritedCodable
final class RtDetrV2ForObjectDetectionConfig: RtDetrV2Config {

    @CodingKey("id2label")
    var id2label: [Int: String]
}
