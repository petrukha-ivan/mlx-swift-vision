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

struct ImageSize: CustomStringConvertible, ExpressibleByArrayLiteral {

    let width: Int
    let height: Int

    init(arrayLiteral elements: Int...) {
        self.width = elements[0]
        self.height = elements[1]
    }

    var description: String {
        "\(width)x\(height)"
    }
}

@Test(arguments: [
    ("resnet", [128, 128] as ImageSize, [224, 224] as ImageSize),
    ("resnet", [128, 256] as ImageSize, [224, 224] as ImageSize),
    ("resnet", [256, 256] as ImageSize, [224, 224] as ImageSize),
    ("resnet", [512, 512] as ImageSize, [224, 224] as ImageSize),
    ("resnet", [1024, 512] as ImageSize, [224, 224] as ImageSize),
    ("resnet", [512, 1024] as ImageSize, [224, 224] as ImageSize),
    ("resnet", [1080, 1920] as ImageSize, [224, 224] as ImageSize),
    ("efficientnet", [128, 128] as ImageSize, [224, 224] as ImageSize),
    ("efficientnet", [128, 256] as ImageSize, [224, 224] as ImageSize),
    ("efficientnet", [256, 256] as ImageSize, [224, 224] as ImageSize),
    ("efficientnet", [512, 512] as ImageSize, [224, 224] as ImageSize),
    ("efficientnet", [1024, 512] as ImageSize, [224, 224] as ImageSize),
    ("efficientnet", [512, 1024] as ImageSize, [224, 224] as ImageSize),
    ("efficientnet", [1080, 1920] as ImageSize, [224, 224] as ImageSize),
    ("detr", [128, 128] as ImageSize, [800, 800] as ImageSize),
    ("detr", [128, 256] as ImageSize, [666, 1333] as ImageSize),
    ("detr", [256, 256] as ImageSize, [800, 800] as ImageSize),
    ("detr", [512, 512] as ImageSize, [800, 800] as ImageSize),
    ("detr", [1024, 512] as ImageSize, [1333, 666] as ImageSize),
    ("detr", [512, 1024] as ImageSize, [666, 1333] as ImageSize),
    ("detr", [1080, 1920] as ImageSize, [750, 1333] as ImageSize),
    ("clip", [128, 128] as ImageSize, [224, 224] as ImageSize),
    ("clip", [128, 256] as ImageSize, [224, 224] as ImageSize),
    ("clip", [256, 256] as ImageSize, [224, 224] as ImageSize),
    ("clip", [512, 512] as ImageSize, [224, 224] as ImageSize),
    ("clip", [1024, 512] as ImageSize, [224, 224] as ImageSize),
    ("clip", [512, 1024] as ImageSize, [224, 224] as ImageSize),
    ("clip", [1080, 1920] as ImageSize, [224, 224] as ImageSize),
    ("sam3", [128, 128] as ImageSize, [1008, 1008] as ImageSize),
    ("sam3", [128, 256] as ImageSize, [1008, 1008] as ImageSize),
    ("sam3", [256, 256] as ImageSize, [1008, 1008] as ImageSize),
    ("sam3", [512, 512] as ImageSize, [1008, 1008] as ImageSize),
    ("sam3", [1024, 512] as ImageSize, [1008, 1008] as ImageSize),
    ("sam3", [512, 1024] as ImageSize, [1008, 1008] as ImageSize),
    ("sam3", [1080, 1920] as ImageSize, [1008, 1008] as ImageSize),
])
func imageResizingAlignedWithTransformers(preprocessorPrefix: String, originalSize: ImageSize, expectedSize: ImageSize) async throws {
    let preprocessorConfigURL = try #require(Bundle.module.url(forResource: "\(preprocessorPrefix)_preprocessor_config", withExtension: "json"))
    let preprocessorConfig = try ImagePreprocessorConfig.decoded(from: preprocessorConfigURL)
    let preprocessor = ImagePreprocessor(preprocessorConfig)
    let sampleImage = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: originalSize.width, height: originalSize.height))
    let processedImage = try preprocessor.preprocess(image: sampleImage)
    let (_, height, width, _) = processedImage.pixelValues.shape4
    #expect(height == expectedSize.height)
    #expect(width == expectedSize.width)
}
