//
//  Requests.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import CoreImage

/// Marker protocol for requests that contain an image payload.
public protocol ImageBasedRequest {
    /// Input image for the task.
    var image: CIImage { get }
}

/// Request for image classification.
public struct ImageClassificationRequest: ImageBasedRequest {

    /// Input image to classify.
    public let image: CIImage

    /// Creates an image classification request.
    public init(image: CIImage) {
        self.image = image
    }
}

/// Request for object detection.
public struct ObjectDetectionRequest: ImageBasedRequest {

    /// Input image to analyze.
    public let image: CIImage
    /// Minimum confidence score required to keep a detection.
    public let scoreThreshold: Float

    /// Creates an object detection request.
    public init(image: CIImage, scoreThreshold: Float = 0.5) {
        self.image = image
        self.scoreThreshold = scoreThreshold
    }
}

/// Request for image segmentation.
public struct InstanceSegmentationRequest: ImageBasedRequest {

    /// Input image to segment.
    public let image: CIImage
    /// Minimum confidence threshold for returned masks.
    public let scoreThreshold: Float

    /// Creates an image segmentation request.
    public init(image: CIImage, scoreThreshold: Float = 0.5) {
        self.image = image
        self.scoreThreshold = scoreThreshold
    }
}

/// Request for zero-shot image classification with candidate labels.
public struct ZeroShotClassificationRequest: ImageBasedRequest {

    /// Input image to classify.
    public let image: CIImage
    /// Candidate labels used as text prompts.
    public let labels: [String]

    /// Creates a zero-shot classification request.
    public init(image: CIImage, labels: [String]) {
        self.image = image
        self.labels = labels
    }
}

/// Request for zero-shot segmentation from a free-form prompt.
public struct ZeroShotSegmentationRequest: ImageBasedRequest {

    /// Input image to segment.
    public let image: CIImage
    /// Natural-language prompt that identifies the target concept.
    public let prompt: String
    /// Minimum confidence threshold for returned masks.
    public let scoreThreshold: Float
    /// Minimum confidence threshold for pixels of the mask.
    public let maskThreshold: Float

    /// Creates a zero-shot segmentation request.
    public init(
        image: CIImage,
        prompt: String,
        scoreThreshold: Float = 0.5,
        maskThreshold: Float = 0.5
    ) {
        self.image = image
        self.prompt = prompt
        self.scoreThreshold = scoreThreshold
        self.maskThreshold = maskThreshold
    }
}

/// Request for embedding extraction from text or image input.
public enum EmbeddingsExtractionRequest {
    /// Extract embeddings for a text prompt.
    case text(String)
    /// Extract embeddings for an image.
    case image(CIImage)
}
