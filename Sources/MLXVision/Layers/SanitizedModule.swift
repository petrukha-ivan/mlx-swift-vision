//
//  SanitizedModule.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 08.11.2025.
//

import MLX
import MLXNN

class Module: MLXNN.Module {

    func sanitize(parameters: ModuleParameters) -> ModuleParameters {
        return parameters
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let parameters = sanitize(parameters: parameters)
        return try super.update(
            parameters: parameters,
            verify: verify,
            path: path,
            modulePath: modulePath
        )
    }
}

final class BatchNorm: MLXNN.BatchNorm {

    @discardableResult
    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let parameters = parameters.filterKeys { $0 != "num_batches_tracked" }
        return try super.update(
            parameters: parameters,
            verify: verify,
            path: path,
            modulePath: modulePath
        )
    }
}

final class Conv2d: MLXNN.Conv2d {

    @discardableResult
    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let parameters = parameters.mapValues { $0 == "weight" ? $1.movedAxis(source: 1, destination: 3) : $1 }
        return try super.update(
            parameters: parameters,
            verify: verify,
            path: path,
            modulePath: modulePath
        )
    }
}

final class ConvTransposed2d: MLXNN.ConvTransposed2d {

    @discardableResult
    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let parameters = parameters.mapValues { $0 == "weight" ? $1.movedAxis(source: 0, destination: 3) : $1 }
        return try super.update(
            parameters: parameters,
            verify: verify,
            path: path,
            modulePath: modulePath
        )
    }
}

extension ModuleParameters {

    func renameKeys(_ transform: (Key) -> Key) -> Self {
        ModuleParameters.unflattened(self.flattened().map { (transform($0.0), $0.1) })
    }

    func filterKeys(_ isIncluded: (Key) -> Bool) -> Self {
        ModuleParameters.unflattened(self.flattened().filter({ isIncluded($0.0) }))
    }
}
