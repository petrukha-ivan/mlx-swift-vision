//
//  MaskAnnotator.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 23.11.2025.
//

import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics
import CoreText
import MLX

public class MaskAnnotator {

    let alpha: CGFloat

    public init(alpha: CGFloat = 0.8) {
        self.alpha = alpha
    }

    public func annotate(image: CIImage, detections: [ImageSegmentationResult]) -> CIImage {
        var image = image
        for detection in detections {
            let hash = detection.label.hash
            let mask = detection.mask
            let (height, width) = mask.shape2
            let rgba = MLX.ones([height, width, 4], dtype: .uint8) * 255
            rgba[.ellipsis, 0] = MLXArray(hash & 255)
            rgba[.ellipsis, 1] = MLXArray((hash >> 8) & 255)
            rgba[.ellipsis, 2] = MLXArray((hash >> 16) & 255)
            rgba[.ellipsis, 3] = MLX.which(mask, 255, 0)

            let rgbaData = rgba.asType(.uint8).asData()
            let annotationImage = CIImage(
                bitmapData: rgbaData.data,
                bytesPerRow: width * 4,
                size: CGSize(width: CGFloat(width), height: CGFloat(height)),
                format: .RGBA8,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
            )

            let filter = CIFilter.sourceOverCompositing()
            filter.backgroundImage = image
            filter.inputImage =
                annotationImage
                .resized(size: image.extent.size)
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
