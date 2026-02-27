//
//  AdaptiveAvgPool2d.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.11.2025.
//

import MLXNN
import MLX

final class AdaptiveAvgPool2d: Module, UnaryLayer {

    let outputSize: IntOrPair

    init(outputSize: IntOrPair) {
        self.outputSize = outputSize
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, H, W, C) = x.shape4
        let x = x.reshaped([B, outputSize.first, H / outputSize.first, outputSize.second, W / outputSize.second, C])
        return x.mean(axes: [2, 4])
    }
}

final class AdaptiveMaxPool2d: Module, UnaryLayer {

    let outputSize: IntOrPair

    init(outputSize: IntOrPair) {
        self.outputSize = outputSize
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, H, W, C) = x.shape4
        let x = x.reshaped([B, outputSize.first, H / outputSize.first, outputSize.second, W / outputSize.second, C])
        return x.max(axes: [2, 4])
    }
}
