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

    public init(alpha: CGFloat = 0.8) {
        self.alpha = alpha
    }

    public func annotate(image: CIImage, detections: [Detection]) -> CIImage {
        var image = image
        for detection in detections {
            let hash = detection.label.hash
            let color = CIColor(
                red: CGFloat(hash & 255) / 255.0,
                green: CGFloat((hash >> 8) & 255) / 255.0,
                blue: CGFloat((hash >> 16) & 255) / 255.0,
                alpha: alpha
            )

            let filter = CIFilter.blendWithMask()
            filter.backgroundImage = image
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
                image = output
            }
        }

        return image
    }
}
