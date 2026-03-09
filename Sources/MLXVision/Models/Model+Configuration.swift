//
//  Model+Configuration.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 09.03.2026.
//

import ReerCodable
import MLXNN

@Codable
enum ActivationType: String {

    case relu
    case gelu
    case silu

    var layer: UnaryLayer {
        switch self {
        case .relu:
            return ReLU()
        case .gelu:
            return GELU(approximation: .none)
        case .silu:
            return SiLU()
        }
    }
}
