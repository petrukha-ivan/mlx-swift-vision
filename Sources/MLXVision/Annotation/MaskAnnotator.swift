//
//  MaskAnnotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 23.11.2025.
//

import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics

public class MaskAnnotator<Detection: MaskedResult & LabeledResult>: Annotator {

    let alpha: CGFloat
    let color: ColorProvider

    public init(alpha: CGFloat = 0.8, color: ColorProvider = .auto) {
        self.alpha = alpha
        self.color = color
    }

    public func overlay(for image: CIImage, detections: [Detection]) -> CIImage {
        let transparent = CIImage(color: .clear).cropped(to: image.extent)
        var overlay = transparent

        for detection in detections {
            let color = color.color(for: detection.label, alpha: alpha)
            let filter = CIFilter.blendWithMask()
            filter.backgroundImage = transparent
            filter.inputImage = CIImage(color: color).cropped(to: image.extent)
            filter.maskImage = detection.mask
                .resized(size: image.extent.size, method: .nearest)
                .transformed(
                    by: CGAffineTransform(
                        translationX: image.extent.origin.x,
                        y: image.extent.origin.y
                    )
                )

            if let output = filter.outputImage {
                overlay = output.composited(over: overlay)
            }
        }

        return overlay
    }
}
