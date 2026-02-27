//
//  Pipeline+Tasks.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.12.2025.
//

import Foundation

/// Marker protocol that binds an input type to an output type for model loading.
public protocol VisionTask {
    /// Task request input type.
    associatedtype Input
    /// Task response output type.
    associatedtype Output
}

/// Image classification task marker.
public enum ImageClassificationTask: VisionTask {
    /// Classification request input.
    public typealias Input = ImageClassificationRequest
    /// Ranked classification outputs.
    public typealias Output = [ClassificationResult]
}

/// Object detection task marker.
public enum ObjectDetectionTask: VisionTask {
    /// Detection request input.
    public typealias Input = ObjectDetectionRequest
    /// Detection outputs.
    public typealias Output = [ObjectDetectionResult]
}

/// Image segmentation task marker.
public enum ImageSegmentationTask: VisionTask {
    /// Segmentation request input.
    public typealias Input = ImageSegmentationRequest
    /// Segmentation outputs.
    public typealias Output = [ImageSegmentationResult]
}

/// Embeddings extraction task marker.
public enum EmbeddingsExtractionTask: VisionTask {
    /// Embeddings extraction request input.
    public typealias Input = EmbeddingsExtractionRequest
    /// Embeddings output.
    public typealias Output = Embeddings
}

/// Zero-shot classification task marker.
public enum ZeroShotClassificationTask: VisionTask {
    /// Zero-shot classification request input.
    public typealias Input = ZeroShotClassificationRequest
    /// Ranked classification outputs.
    public typealias Output = [ClassificationResult]
}

/// Zero-shot segmentation task marker.
public enum ZeroShotSegmentationTask: VisionTask {
    /// Zero-shot segmentation request input.
    public typealias Input = ZeroShotSegmentationRequest
    /// Segmentation outputs.
    public typealias Output = [ImageSegmentationResult]
}

/// Type-erased classification model pipeline.
public typealias AnyModelForImageClassification = AnyPipeline<ImageClassificationRequest, [ClassificationResult]>
/// Type-erased object detection model pipeline.
public typealias AnyModelForObjectDetection = AnyPipeline<ObjectDetectionRequest, [ObjectDetectionResult]>
/// Type-erased image segmentation model pipeline.
public typealias AnyModelForImageSegmentation = AnyPipeline<ImageSegmentationRequest, [ImageSegmentationResult]>
/// Type-erased zero-shot classification model pipeline.
public typealias AnyModelForZeroShotClassification = AnyPipeline<ZeroShotClassificationRequest, [ClassificationResult]>
/// Type-erased zero-shot segmentation model pipeline.
public typealias AnyModelForZeroShotSegmentation = AnyPipeline<ZeroShotSegmentationRequest, [ImageSegmentationResult]>
/// Type-erased embeddings extraction model pipeline.
public typealias AnyModelForEmbeddingsExtraction = AnyPipeline<EmbeddingsExtractionRequest, Embeddings>
