//
//  TestImageResizing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 18.02.2026.
//

import Testing
import Foundation
import CoreImage
@testable import MLXVision

@Test(arguments: [
    (TargetSize.value(224), CGSize(width: 128, height: 128), CGSize(width: 224, height: 224)),
    (TargetSize.value(224), CGSize(width: 128, height: 256), CGSize(width: 224, height: 224)),
    (TargetSize.value(224), CGSize(width: 256, height: 256), CGSize(width: 224, height: 224)),
    (TargetSize.value(224), CGSize(width: 512, height: 512), CGSize(width: 224, height: 224)),
    (TargetSize.value(224), CGSize(width: 1024, height: 512), CGSize(width: 224, height: 224)),
    (TargetSize.value(224), CGSize(width: 512, height: 1024), CGSize(width: 224, height: 224)),
    (TargetSize.value(224), CGSize(width: 1080, height: 1920), CGSize(width: 224, height: 224)),
    (TargetSize.width(1008, height: 1008), CGSize(width: 128, height: 128), CGSize(width: 1008, height: 1008)),
    (TargetSize.width(1008, height: 1008), CGSize(width: 128, height: 256), CGSize(width: 1008, height: 1008)),
    (TargetSize.width(1008, height: 1008), CGSize(width: 256, height: 256), CGSize(width: 1008, height: 1008)),
    (TargetSize.width(1008, height: 1008), CGSize(width: 512, height: 512), CGSize(width: 1008, height: 1008)),
    (TargetSize.width(1008, height: 1008), CGSize(width: 1024, height: 512), CGSize(width: 1008, height: 1008)),
    (TargetSize.width(1008, height: 1008), CGSize(width: 512, height: 1024), CGSize(width: 1008, height: 1008)),
    (TargetSize.width(1008, height: 1008), CGSize(width: 1080, height: 1920), CGSize(width: 1008, height: 1008)),
    (TargetSize.shortestEdge(800, longestEdge: 1333), CGSize(width: 128, height: 128), CGSize(width: 800, height: 800)),
    (TargetSize.shortestEdge(800, longestEdge: 1333), CGSize(width: 128, height: 256), CGSize(width: 666, height: 1333)),
    (TargetSize.shortestEdge(800, longestEdge: 1333), CGSize(width: 256, height: 256), CGSize(width: 800, height: 800)),
    (TargetSize.shortestEdge(800, longestEdge: 1333), CGSize(width: 512, height: 512), CGSize(width: 800, height: 800)),
    (TargetSize.shortestEdge(800, longestEdge: 1333), CGSize(width: 1024, height: 512), CGSize(width: 1333, height: 666)),
    (TargetSize.shortestEdge(800, longestEdge: 1333), CGSize(width: 512, height: 1024), CGSize(width: 666, height: 1333)),
    (TargetSize.shortestEdge(800, longestEdge: 1333), CGSize(width: 1080, height: 1920), CGSize(width: 750, height: 1333)),
])
func imageResizingAlignedWithTransformers(targetSize: TargetSize, originalSize: CGSize, expectedSize: CGSize) async throws {
    let sampleImage = CIImage(color: .white).cropped(to: CGRect(origin: .zero, size: originalSize))
    let resizedImage = sampleImage.resized(size: targetSize, method: .bilinear)
    #expect(resizedImage.extent.height == expectedSize.height)
    #expect(resizedImage.extent.width == expectedSize.width)
}
