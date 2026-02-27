//
//  ImagePreprocessor+Config.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.12.2025.
//

import ReerCodable

@Codable
enum ResampleMethod: Int {
    case nearest = 0
    case lanczos = 1
    case bilinear = 2
    case bicubic = 3
}

@Codable
enum TargetSize: Equatable, Hashable {
    case value(Int)
    case width(Int, height: Int)
    case shortestEdge(Int, longestEdge: Int?)
}

@Codable
struct ImagePreprocessorConfig {

    @CodingKey("resample")
    var resample: ResampleMethod = .bicubic

    @CustomCoding(TargetSizeCoding.self)
    var targetSize: TargetSize

    @CodingKey("image_mean")
    var imageMean: [Float] = [0.5, 0.5, 0.5]

    @CodingKey("image_std")
    var imageStd: [Float] = [0.5, 0.5, 0.5]
}

struct TargetSizeCoding: CodingCustomizable {

    static func encode(by encoder: any Encoder, key: String, value: TargetSize) throws {
        switch value {
        case .value(let size):
            try encoder.set(size, forKey: key)
        case .width(let width, let height):
            try encoder.set([width, height], forKey: key)
        case .shortestEdge(let shortestEdge, let longestEdge):
            try encoder.set(shortestEdge, forKey: key)
            if let longestEdge {
                try encoder.set(longestEdge, forKey: "max_size")
            }
        }
    }

    static func decode(by decoder: any Decoder, keys: [String]) throws -> TargetSize {
        let strategies: [(Decoder) throws -> TargetSize?] = [
            {
                let size = try $0.value(forKeys: "size") as Int
                let maxSize = try? $0.value(forKeys: "max_size") as Int
                return .shortestEdge(size, longestEdge: maxSize)
            },
            {
                let array = try $0.value(forKeys: "size") as [Int]
                if array.count == 2, let width = array.first, let height = array.last {
                    return .width(width, height: height)
                } else {
                    return nil
                }
            },
            {
                let dictionary = try $0.value(forKeys: "size") as [String: Int]
                if let width = dictionary["width"], let height = dictionary["height"] {
                    return .width(width, height: height)
                } else {
                    return nil
                }
            },
            {
                let dictionary = try $0.value(forKeys: "size") as [String: Int]
                if let shortestEdge = dictionary["shortest_edge"] {
                    return .shortestEdge(shortestEdge, longestEdge: dictionary["longest_edge"])
                } else {
                    return nil
                }
            },
        ]

        for strategy in strategies {
            if let decoded = try? strategy(decoder) {
                return decoded
            }
        }

        throw DecodingError.valueNotFound(
            TargetSize.self,
            DecodingError.Context(
                codingPath: [AnyCodingKey("size")],
                debugDescription: "Missing size value"
            )
        )
    }
}
