//
//  Padding2d.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.12.2025.
//

import MLXNN
import MLX

final class Padding2d: Module {

    let padding: IntOrPair
    let paddingValue: ScalarOrArray

    init(padding: IntOrPair, paddingValue: ScalarOrArray) {
        self.padding = padding
        self.paddingValue = paddingValue
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let widths = [IntOrPair((0, 0)), padding, padding, IntOrPair((0, 0))]
        let value = paddingValue.asMLXArray(dtype: x.dtype)
        let padded = MLX.padded(x, widths: widths, value: value)
        return padded
    }
}
