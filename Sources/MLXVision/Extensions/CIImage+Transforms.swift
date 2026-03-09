//
//  CIImage+Transforms.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 09.03.2026.
//

import CoreImage

extension CIImage {

    func resizedNearest(scaleX: CGFloat, scaleY: CGFloat) -> CIImage {
        self.samplingNearest()
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }

    func resizedBilinear(scaleX: CGFloat, scaleY: CGFloat) -> CIImage {
        self.samplingLinear()
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }

    func resizedLanczos(scaleX: CGFloat, scaleY: CGFloat) -> CIImage {
        let filter = CIFilter.lanczosScaleTransform()
        filter.inputImage = self
        filter.scale = Float(scaleY)
        filter.aspectRatio = Float(scaleX / scaleY)
        return filter.outputImage!
    }

    func resizedBicubic(scaleX: CGFloat, scaleY: CGFloat) -> CIImage {
        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = self
        filter.scale = Float(scaleY)
        filter.aspectRatio = Float(scaleX / scaleY)
        return filter.outputImage!
    }

    func resized(size targetSize: CGSize, method: ResampleMethod = .bilinear) -> CIImage {
        let currentSize = extent.size
        let scaleX = targetSize.width / currentSize.width
        let scaleY = targetSize.height / currentSize.height

        let scaledImage: CIImage
        switch method {
        case .nearest:
            scaledImage = resizedNearest(scaleX: scaleX, scaleY: scaleY)
        case .bilinear:
            scaledImage = resizedBilinear(scaleX: scaleX, scaleY: scaleY)
        case .lanczos:
            scaledImage = resizedLanczos(scaleX: scaleX, scaleY: scaleY)
        case .bicubic:
            scaledImage = resizedBicubic(scaleX: scaleX, scaleY: scaleY)
        }

        let cropRect = CGRect(
            x: scaledImage.extent.midX.rounded(.toNearestOrEven) - targetSize.width / 2.0,
            y: scaledImage.extent.midY.rounded(.toNearestOrEven) - targetSize.height / 2.0,
            width: targetSize.width,
            height: targetSize.height
        )

        return scaledImage.cropped(to: cropRect)
    }
}

extension CIImage {

    public func centerCropped() -> CIImage {
        return centerCropped(
            CGSize(
                width: min(extent.width, extent.height),
                height: min(extent.width, extent.height)
            )
        )
    }

    public func centerCropped(_ size: CGSize) -> CIImage {
        let width = extent.width
        let height = extent.height

        let cropX = extent.origin.x + (width - size.width) / 2.0
        let cropY = extent.origin.y + (height - size.height) / 2.0

        return cropped(
            to: CGRect(
                x: max(extent.origin.x, cropX),
                y: max(extent.origin.y, cropY),
                width: min(size.width, width),
                height: min(size.height, height)
            )
        )
    }
}
