//
//  BoxAnnotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 23.11.2025.
//

import CoreImage
import CoreGraphics

public class BoxAnnotator<Detection: BoxedResult & LabeledResult>: Annotator {

    let lineWidth: CGFloat
    let strokeColor: ColorProvider

    public init(
        lineWidth: CGFloat = 8.0,
        strokeColor: ColorProvider = .auto
    ) {
        self.lineWidth = lineWidth
        self.strokeColor = strokeColor
    }

    public func overlay(for image: CIImage, detections: [Detection]) -> CIImage {
        let transparent = CIImage(color: .clear).cropped(to: image.extent)
        let canvasSize = CGSize(width: 1024, height: 1024)
        let canvasScale = CGAffineTransform(scaleX: canvasSize.width, y: canvasSize.height)

        let annotation = CGImage.render(size: canvasSize) { context in
            context.setLineWidth(lineWidth)
            for detection in detections {
                let strokeColor = strokeColor.color(for: detection.label, alpha: 1.0).resolvedCGColor
                let bbox = detection.bbox.applying(canvasScale)
                context.setStrokeColor(strokeColor)
                context.stroke(bbox)
            }
        }

        let overlay = annotation.map(CIImage.init)?
            .resized(size: image.extent.size)
            .transformed(
                by: CGAffineTransform(
                    translationX: image.extent.origin.x,
                    y: image.extent.origin.y
                )
            )

        return overlay ?? transparent
    }
}
