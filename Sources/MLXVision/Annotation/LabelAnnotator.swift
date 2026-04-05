//
//  LabelAnnotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 23.11.2025.
//

import CoreGraphics
import CoreImage
import CoreText

public class LabelAnnotator<Detection: LabeledResult & BoxedResult>: Annotator {

    let fontSize: CGFloat
    let backgroundColor: ColorProvider
    let textAttributes: [CFString: Any]

    public init(
        fontSize: CGFloat = 24,
        foregroundColor: CGColor = CGColor(red: 1, green: 1, blue: 1, alpha: 1),
        backgroundColor: ColorProvider = .auto
    ) {
        self.fontSize = fontSize
        self.backgroundColor = backgroundColor
        self.textAttributes = [
            kCTFontAttributeName: CTFontCreateWithName("Helvetica" as CFString, fontSize, nil),
            kCTForegroundColorAttributeName: foregroundColor,
        ]
    }

    public func overlay(for image: CIImage, detections: [Detection]) -> CIImage {
        let transparent = CIImage(color: .clear).cropped(to: image.extent)
        let canvasSize = CGSize(width: 1024, height: 1024)
        let canvasScale = CGAffineTransform(scaleX: canvasSize.width, y: canvasSize.height)

        let annotation = CGImage.render(size: canvasSize) { context in
            context.textMatrix = .identity
            context.translateBy(x: 0, y: canvasSize.height)
            context.scaleBy(x: 1.0, y: -1.0)
            for detection in detections {
                let backgroundColor = backgroundColor.color(for: detection.label, alpha: 1.0).resolvedCGColor
                let textAttributes = textAttributes.merging([kCTBackgroundColorAttributeName: backgroundColor], uniquingKeysWith: { _, new in new })
                let attributedString = CFAttributedStringCreate(kCFAllocatorDefault, detection.label as CFString, textAttributes as CFDictionary)!
                let textLine = CTLineCreateWithAttributedString(attributedString)
                let point = detection.bbox.origin.applying(canvasScale)
                context.textPosition = CGPoint(x: point.x, y: canvasSize.height - point.y - fontSize / 2 - 6)
                CTLineDraw(textLine, context)
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
