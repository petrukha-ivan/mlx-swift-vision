//
//  MLXArray+Ops.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 09.11.2025.
//

import MLX
import MLXNN

extension Array<MLXArray> {

    func stacked(axis: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.stacked(self, axis: axis, stream: stream)
    }

    func concatenated(axis: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.concatenated(self, axis: axis, stream: stream)
    }
}

extension MLXArray {

    func sigmoid(stream: StreamOrDevice = .default) -> MLXArray {
        MLX.sigmoid(self, stream: stream)
    }

    func softmax(precise: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.softmax(self, precise: precise, stream: stream)
    }

    func softmax(axis: Int, precise: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.softmax(self, axis: axis, precise: precise, stream: stream)
    }

    func argmax(axis: Int = -1, keepDims: Bool = false, stream: StreamOrDevice = .default) -> MLXArray {
        MLX.argMax(self, axis: axis, keepDims: keepDims, stream: stream)
    }

    func interpolate(size: [Int], mode: Upsample.Mode = .nearest) -> MLXArray {
        let N = shape.dropFirst().dropLast()
        let scales = zip(size, N).map { Float($0) / Float($1) }
        return Upsample(scaleFactor: .array(scales), mode: mode)(self)
    }

    func norm(pow: Int = 2, axis: Int = -1, keepDims: Bool = true) -> MLXArray {
        let squared = MLX.pow(self, pow)
        let sum = MLX.sum(squared, axis: axis, keepDims: keepDims)
        let norm = MLX.pow(sum, 0.5)
        return norm
    }

    func inverseSigmoid(eps: Float = 1e-3) -> MLXArray {
        let x = MLX.clip(self, min: 0, max: 1)
        let x1 = MLX.clip(x, min: eps)
        let x2 = MLX.clip(1 - x, min: eps)
        return MLX.log(x1 / x2)
    }

    func rotatedPairwise() -> MLXArray {
        var x = reshaped(shape.dropLast() + [-1, 2])
        var (x1, x2) = x.split(axis: -1)
        x1 = x1.squeezed()
        x2 = x2.squeezed()
        x = MLX.stacked([-x2, x1], axis: -1)
        x = x.flattened(start: -2)
        return x
    }
}

extension MLXArray {

    func split(axis: Int = 0, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray, MLXArray) {
        let pieces = self.split(parts: 3, axis: axis, stream: stream)
        return (pieces[0], pieces[1], pieces[2])
    }

    func split(axis: Int = 0, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let pieces = self.split(parts: 4, axis: axis, stream: stream)
        return (pieces[0], pieces[1], pieces[2], pieces[3])
    }
}
