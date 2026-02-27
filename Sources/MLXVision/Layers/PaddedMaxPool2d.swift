//
//  PaddedMaxPool2d.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.11.2025.
//

import MLXNN
import MLX

final class PaddedMaxPool2d: Module {

    let inner: MaxPool2d
    let padding: IntOrPair
    let paddingValue: ScalarOrArray

    init(kernelSize: IntOrPair, stride: IntOrPair, padding: IntOrPair) {
        self.padding = padding
        self.paddingValue = -Float.infinity
        self.inner = MaxPool2d(kernelSize: kernelSize, stride: stride)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let widths = [IntOrPair((0, 0)), padding, padding, IntOrPair((0, 0))]
        let value = paddingValue.asMLXArray(dtype: x.dtype)
        let padded = MLX.padded(x, widths: widths, value: value)
        return inner(padded)
    }
}
