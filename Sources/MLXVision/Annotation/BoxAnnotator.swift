//
//  BoxAnnotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 23.11.2025.
//

import CoreImage
import CoreGraphics

public class BoxAnnotator<Detection: BoxedResult>: Annotator {

    let lineWidth: CGFloat
    let strokeColor: CGColor

    public init(
        lineWidth: CGFloat = 8.0,
        strokeColor: CGColor = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
    ) {
        self.lineWidth = lineWidth
        self.strokeColor = strokeColor
    }

    public func annotate(image: CIImage, detections: [Detection]) -> CIImage {
        let canvasSize = CGSize(width: 1024, height: 1024)
        let annotation = CGImage.render(size: canvasSize) { context in
            context.setLineWidth(lineWidth)
            context.setStrokeColor(strokeColor)
            for bbox in detections.map(\.bbox) {
                context.stroke(
                    CGRect(
                        x: CGFloat(bbox[0]) * canvasSize.width,
                        y: CGFloat(bbox[1]) * canvasSize.height,
                        width: CGFloat(bbox[2]) * canvasSize.width,
                        height: CGFloat(bbox[3]) * canvasSize.height
                    )
                )
            }
        }

        let filter = CIFilter.sourceOverCompositing()
        filter.backgroundImage = image
        filter.inputImage = annotation.map(CIImage.init)?
            .resized(size: image.extent.size)
            .transformed(
                by: CGAffineTransform(
                    translationX: image.extent.origin.x,
                    y: image.extent.origin.y
                )
            )

        return filter.outputImage ?? image
    }
}
