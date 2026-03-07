//
//  Annotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.03.2026.
//

import CoreImage

public protocol Annotator<Detection> {

    associatedtype Detection

    func annotate(image: CIImage, detections: [Detection]) -> CIImage
}

public struct ComposedAnnotator<Detection>: Annotator {

    private let annotators: [any Annotator<Detection>]

    public init(_ annotators: any Annotator<Detection>...) {
        self.annotators = annotators
    }

    public init(_ annotators: [any Annotator<Detection>]) {
        self.annotators = annotators
    }

    public func annotate(image: CIImage, detections: [Detection]) -> CIImage {
        annotators.reduce(image) { image, annotator in
            annotator.annotate(image: image, detections: detections)
        }
    }
}
