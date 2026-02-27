//
//  Array+Callable.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 05.11.2025.
//

import MLX
import MLXNN

extension Array where Element == UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        reduce(x, { $1($0) })
    }
}
