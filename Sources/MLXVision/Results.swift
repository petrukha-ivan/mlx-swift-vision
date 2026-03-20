//
//  Results.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import CoreImage
import CoreGraphics

public protocol LabeledResult {
    var label: String { get }
}

public protocol ScoredResult {
    var score: Float { get }
}

public protocol BoxedResult {
    var bbox: CGRect { get }
}

public protocol MaskedResult {
    var mask: CIImage { get }
}

public extension Array where Element: ScoredResult {
    func top(_ k: Int) -> Self {
        Array(sorted(by: { $0.score > $1.score }).prefix(k))
    }
}

/// Single classification prediction with label and confidence score.
public struct ClassificationResult: Sendable, Hashable, LabeledResult, ScoredResult {

    /// Predicted label.
    public let label: String
    /// Confidence score for the label.
    public let score: Float

    /// Creates a classification result entry.
    public init(label: String, score: Float) {
        self.label = label
        self.score = score
    }
}

/// Object detection result with normalized geometry and confidence metadata.
public struct ObjectDetectionResult: Sendable, Hashable, LabeledResult, ScoredResult, BoxedResult {

    /// Normalized bounding box.
    public let bbox: CGRect
    /// Predicted label.
    public let label: String
    /// Confidence score for the label.
    public let score: Float

    /// Creates an object detection result.
    public init(bbox: CGRect, label: String, score: Float) {
        self.bbox = bbox
        self.label = label
        self.score = score
    }
}

/// Image segmentation result with binary mask, normalized geometry, and metadata.
public struct InstanceSegmentationResult: LabeledResult, ScoredResult, MaskedResult, BoxedResult {

    /// Segmentation mask for the result.
    public let mask: CIImage
    /// Normalized bounding box.
    public let bbox: CGRect
    /// Predicted label.
    public let label: String
    /// Confidence score for the label.
    public let score: Float

    /// Creates an image segmentation result.
    public init(mask: CIImage, bbox: CGRect, label: String, score: Float) {
        self.mask = mask
        self.bbox = bbox
        self.label = label
        self.score = score
    }
}

/// Embeddings vector output.
public typealias Embeddings = [Float]
