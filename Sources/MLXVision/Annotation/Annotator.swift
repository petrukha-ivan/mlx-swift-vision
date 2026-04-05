//
//  Annotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.03.2026.
//

import CoreImage

public protocol Annotator<Detection> {

    associatedtype Detection

    func overlay(for image: CIImage, detections: [Detection]) -> CIImage
}

public extension Annotator {
    func annotate(image: CIImage, detections: [Detection]) -> CIImage {
        overlay(for: image, detections: detections).composited(over: image)
    }
}

public struct ComposedAnnotator<Detection>: Annotator {

    private let annotators: [any Annotator<Detection>]

    public init(_ annotators: any Annotator<Detection>...) {
        self.annotators = annotators
    }

    public init(_ annotators: [any Annotator<Detection>]) {
        self.annotators = annotators
    }

    public func overlay(for image: CIImage, detections: [Detection]) -> CIImage {
        let transparent = CIImage(color: .clear).cropped(to: image.extent)
        return annotators.reduce(transparent) { overlay, annotator in
            annotator.overlay(for: image, detections: detections).composited(over: overlay)
        }
    }
}
