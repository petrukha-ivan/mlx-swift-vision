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

        let format = CIFormat.RGBX8
        let rowBytes = width * 4 * 1

        let bitmap = MLXArray.zeros([height, width, 4], dtype: .uint8)
        var bitmapData = bitmap.asData(access: .noCopy).data
        bitmapData.withUnsafeMutableBytes { buffer in
            context.render(image, toBitmap: buffer.baseAddress!, rowBytes: rowBytes, bounds: image.extent, format: format, colorSpace: nil)
        }

        let outputs = _preprocess([bitmap])
        let pixelValues = outputs[0]
        let pixelMask = outputs[1]

        return ImageInput(
            pixelValues: pixelValues,
            pixelMask: pixelMask
        )
    }

    private lazy var _preprocess = MLX.compile { [unowned self] inputs in
        let mean = MLXArray(config.imageMean)
        let std = MLXArray(config.imageStd)
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
}
