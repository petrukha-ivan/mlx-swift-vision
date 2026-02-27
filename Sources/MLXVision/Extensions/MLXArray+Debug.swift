//
//  MLXArray+Debug.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 09.11.2025.
//

import MLX

extension MLXArray {
    func describe(precision: Int = 4) {
        let format = "%.\(precision)f"
        print("Dtype: \(dtype)")
        print("Shape: \(shape)")
        print("Min:   \(String(format: format, min().item(Float.self)))")
        print("Max:   \(String(format: format, max().item(Float.self)))")
        print("Mean:  \(String(format: format, mean().item(Float.self)))")
        print("Std:   \(String(format: format, std(self).item(Float.self)))")
    }
}
