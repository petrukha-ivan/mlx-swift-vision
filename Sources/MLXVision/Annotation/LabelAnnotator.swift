//
//  LabelAnnotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 23.11.2025.
//

import CoreGraphics
import CoreImage
import CoreText

public class LabelAnnotator {

    let fontSize: CGFloat
    let textAttributes: [CFString: Any]

    public init(
        fontSize: CGFloat = 24,
        foregroundColor: CGColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1),
        backgroundColor: CGColor = CGColor(red: 0, green: 0, blue: 0, alpha: 1)
    ) {
        self.fontSize = fontSize
        self.textAttributes = [
            kCTFontAttributeName: CTFontCreateWithName("Helvetica" as CFString, fontSize, nil),
            kCTForegroundColorAttributeName: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
            kCTBackgroundColorAttributeName: CGColor(red: 0, green: 0, blue: 0, alpha: 1),
        ]
    }

    public func annotate(image: CIImage, detections: [ObjectDetectionResult]) -> CIImage {
        let canvasSize = CGSize(width: 1024, height: 1024)
        let annotation = CGImage.render(size: canvasSize) { context in
            context.textMatrix = .identity
            context.translateBy(x: 0, y: canvasSize.height)
            context.scaleBy(x: 1.0, y: -1.0)
            for detection in detections {
                let attributedString = CFAttributedStringCreate(kCFAllocatorDefault, detection.label as CFString, textAttributes as CFDictionary)!
                let textLine = CTLineCreateWithAttributedString(attributedString)
                let point = CGPoint(x: CGFloat(detection.bbox[0]) * canvasSize.width, y: CGFloat(detection.bbox[1]) * canvasSize.height)
                context.textPosition = CGPoint(x: point.x, y: canvasSize.height - point.y - fontSize / 2 - 6)
                CTLineDraw(textLine, context)
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
