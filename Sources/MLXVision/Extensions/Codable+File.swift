//
//  Codable+File.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.12.2025.
//

import Foundation
import ReerCodable

extension Decodable {

    static func decoded(
        from url: URL,
        using decoder: JSONDecoder = .init(),
        as type: Self.Type = Self.self
    ) throws -> Self {
        return try Self.decoded(
            from: Data(contentsOf: url),
            using: decoder,
            as: type
        )
    }

    static func decoded(
        from url: URL,
        filename: String,
        using decoder: JSONDecoder = .init(),
        as type: Self.Type = Self.self
    ) throws -> Self {
        return try Self.decoded(
            from: Data(contentsOf: url.appendingPathComponent(filename)),
            using: decoder,
            as: type
        )
    }
}
