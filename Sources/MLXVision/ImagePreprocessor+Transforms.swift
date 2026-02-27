//
//  ImagePreprocessor+Transforms.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.11.2025.
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
        let currentSize = extent.integral.size
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
            x: scaledImage.extent.midX - targetSize.width / 2.0,
            y: scaledImage.extent.midY - targetSize.height / 2.0,
            width: targetSize.width,
            height: targetSize.height
        )

        return scaledImage.cropped(to: cropRect)
    }

    func resized(size: TargetSize, method: ResampleMethod) -> CIImage {
        func calculateTargetSize() -> CGSize {
            switch size {
            case .value(let value):
                return CGSize(
                    width: CGFloat(value),
                    height: CGFloat(value)
                )
            case .width(let width, let height):
                return CGSize(
                    width: CGFloat(width),
                    height: CGFloat(height)
                )
            case .shortestEdge(let shortestEdge, let longestEdge):
                guard let longestEdge else {
                    return CGSize(
                        width: CGFloat(shortestEdge),
                        height: CGFloat(shortestEdge)
                    )
                }

                let width = currentSize.width
                let height = currentSize.height
                let shortest = min(width, height)
                let longest = max(width, height)
                let scaleFromShortest = CGFloat(shortestEdge) / shortest
                let scaleFromLongest = CGFloat(longestEdge) / longest
                let scale = min(scaleFromShortest, scaleFromLongest)
                return CGSize(
                    width: CGFloat((width * scale).rounded(.toNearestOrEven)),
                    height: CGFloat((height * scale).rounded(.toNearestOrEven))
                )
            }
        }

        let currentSize = extent.integral.size
        let targetSize = calculateTargetSize()

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
            x: scaledImage.extent.midX - targetSize.width / 2.0,
            y: scaledImage.extent.midY - targetSize.height / 2.0,
            width: targetSize.width,
            height: targetSize.height
        )

        return scaledImage.cropped(to: cropRect)
    }

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

        return self.cropped(
            to: CGRect(
                x: max(extent.origin.x, cropX),
                y: max(extent.origin.y, cropY),
                width: min(size.width, width),
                height: min(size.height, height)
            )
        )
    }
}
