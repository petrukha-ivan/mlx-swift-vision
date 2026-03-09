//
//  ImagePreprocessor+Transforms.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.11.2025.
//

import CoreImage

extension CIImage {
    func resized(size targetSize: TargetSize, method: ResampleMethod) -> CIImage {
        let currentSize = extent.size
        let targetSize = targetSize.finalSize(for: currentSize)
        return resized(size: targetSize, method: method)
    }
}

extension TargetSize {
    func finalSize(for imageSize: CGSize) -> CGSize {
        switch self {
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

            let width = imageSize.width
            let height = imageSize.height
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
}
