//
//  ImagePreprocessor.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 02.11.2025.
//

import CoreImage
import CoreImage.CIFilterBuiltins
import MLX

class ImagePreprocessor {

    let config: ImagePreprocessorConfig
    let overrides: ModelOverrides
    let context = CIContext()

    init(_ config: ImagePreprocessorConfig, overrides: ModelOverrides = ModelOverrides()) {
        self.config = config
        self.overrides = overrides
    }

    func preprocess(image: CIImage) throws -> ImageInput {
        let image =
            if let targetSize = overrides.inputSize {
                image.resized(size: targetSize, method: config.resample)
            } else {
                image.resized(size: config.targetSize, method: config.resample)
            }

        let size = image.extent.size
        let height = Int(size.height.rounded())
        let width = Int(size.width.rounded())

        let format = CIFormat.RGBA8
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

        let rowBytes = width * 4
        var bitmapData = Data(count: height * rowBytes)
        bitmapData.withUnsafeMutableBytes { buffer in
            context.render(image, toBitmap: buffer.baseAddress!, rowBytes: rowBytes, bounds: image.extent, format: format, colorSpace: colorSpace)
            context.clearCaches()
        }

        let pixels = MLXArray(bitmapData, [height, width, 4], dtype: .int8)
        let (pixelValues, pixelMask) = preprocess(pixels: pixels)

        return ImageInput(
            pixelValues: pixelValues,
            pixelMask: pixelMask
        )
    }

    private lazy var _preprocess = MLX.compile { [unowned self] inputs in
        let mean = config.imageMean.asMLXArray(dtype: .float32)
        let std = config.imageStd.asMLXArray(dtype: .float32)
        var pixelValues = inputs[0][0..., 0..., ..<3]
        pixelValues = pixelValues / 255.0
        pixelValues = (pixelValues - mean) / std
        pixelValues = pixelValues.expandedDimensions(axis: 0)

        var pixelMask = MLX.ones(like: pixelValues)
        if let dtype = overrides.modelDtype {
            pixelValues = pixelValues.asType(dtype)
            pixelMask = pixelMask.asType(dtype)
        }

        return [pixelValues, pixelMask]
    }

    func preprocess(pixels: MLXArray) -> (MLXArray, MLXArray) {
        let outputs = _preprocess([pixels])
        let pixelValues = outputs[0]
        let pixelMask = outputs[1]
        return (pixelValues, pixelMask)
    }
}
