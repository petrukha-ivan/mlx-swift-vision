//
//  LwDetr+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.03.2026.
//

import ReerCodable

@Codable
class LwDetrVitConfig {

    @CodingKey("hidden_size")
    var hiddenSize: Int

    @CodingKey("num_hidden_layers")
    var numHiddenLayers: Int

    @CodingKey("num_attention_heads")
    var numAttentionHeads: Int = 12

    @CodingKey("mlp_ratio")
    var mlpRatio: Int = 4

    @CodingKey("hidden_act")
    var hiddenAct: String = "gelu"

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float = 1e-6

    @CodingKey("image_size")
    var imageSize: Int = 224

    @CodingKey("pretrain_image_size")
    var pretrainImageSize: Int = 224

    @CodingKey("patch_size")
    var patchSize: Int = 14

    @CodingKey("num_channels")
    var numChannels: Int = 3

    @CodingKey("qkv_bias")
    var qkvBias: Bool = true

    @CodingKey("use_mask_token")
    var useMaskToken: Bool = true

    @CodingKey("num_register_tokens")
    var numRegisterTokens: Int = 0

    @CodingKey("use_absolute_position_embeddings")
    var useAbsolutePositionEmbeddings: Bool = true

    @CodingKey("num_windows")
    var numWindows: Int = 4

    @CodingKey("num_windows_side")
    var numWindowsSide: Int = 0

    @CodingKey("out_indices")
    var outIndices: [Int] = []

    @CodingKey("window_block_indices")
    var windowBlockIndices: [Int] = []
}

@Codable
class LwDetrConfig {

    @CodingKey("backbone_config")
    var backboneConfig: LwDetrVitConfig

    @CodingKey("num_queries")
    var numQueries: Int

    @CodingKey("hidden_expansion")
    var hiddenExpansion: Float = 0.5

    @CodingKey("c2f_num_blocks")
    var c2fNumBlocks: Int = 3

    @CodingKey("activation_function")
    var activationFunction: String = "silu"

    @CodingKey("layer_norm_eps")
    var layerNormEps: Float = 1e-5

    @CodingKey("batch_norm_eps")
    var batchNormEps: Float = 1e-5

    @CodingKey("d_model")
    var dModel: Int = 256

    @CodingKey("decoder_ffn_dim")
    var decoderFfnDim: Int = 2_048

    @CodingKey("decoder_n_points")
    var decoderNPoints: Int = 4

    @CodingKey("decoder_layers")
    var decoderLayers: Int = 3

    @CodingKey("decoder_self_attention_heads")
    var decoderSelfAttentionHeads: Int = 8

    @CodingKey("decoder_cross_attention_heads")
    var decoderCrossAttentionHeads: Int = 16

    @CodingKey("decoder_activation_function")
    var decoderActivationFunction: String = "relu"

    @CodingKey("num_feature_levels")
    var numFeatureLevels: Int = 1

    @CodingKey("attention_bias")
    var attentionBias: Bool = true

    @CodingKey("group_detr")
    var groupDetr: Int = 13
}

@InheritedCodable
final class LwDetrForObjectDetectionConfig: LwDetrConfig {

    @CodingKey("id2label")
    var id2label: [Int: String]
}
