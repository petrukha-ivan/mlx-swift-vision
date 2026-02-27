//
//  ModelInputs.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import MLX

/// Canonical vision model input tensor bundle.
public struct ImageInput {

    /// Image tensor values, typically `[batch, height, width, channels]`.
    public let pixelValues: MLXArray
    /// Pixel mask tensor aligned with `pixelValues`.
    public let pixelMask: MLXArray

    /// Creates an image input bundle.
    public init(pixelValues: MLXArray, pixelMask: MLXArray) {
        self.pixelValues = pixelValues
        self.pixelMask = pixelMask
    }
}

/// Canonical text model input tensor bundle.
public struct TextInput {

    /// Token IDs for text input.
    public let textTokens: MLXArray
    /// Attention mask for text input.
    public let textMask: MLXArray

    /// Creates a text input bundle.
    public init(textTokens: MLXArray, textMask: MLXArray) {
        self.textTokens = textTokens
        self.textMask = textMask
    }
}

/// Multimodal model input tensor bundle for text-image tasks.
public struct MultimodalInput {

    /// Text tensors for the request.
    public let textInput: TextInput
    /// Image tensors for the request.
    public let imageInput: ImageInput

    /// Creates a multimodal input bundle.
    public init(textInput: TextInput, imageInput: ImageInput) {
        self.textInput = textInput
        self.imageInput = imageInput
    }
}
