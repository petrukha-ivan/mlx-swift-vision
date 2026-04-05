//
//  ColorProvider.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 05.04.2026.
//

import CoreGraphics
import CoreImage
import Foundation

public enum ColorProvider: Sendable {
    case auto
    case fixed(CIColor)
}

extension ColorProvider {

    struct CacheKey: Hashable {
        let label: String
        let alpha: CGFloat
    }

    static let cache = Cache<CacheKey, CIColor>()

    func color(for label: String, alpha: CGFloat) -> CIColor {
        switch self {
        case .fixed(let color):
            return color.alpha == alpha ? color : color.withAlpha(alpha)
        case .auto:
            return Self.cache.value(for: CacheKey(label: label, alpha: alpha)) {
                let hash = label.hash
                return CIColor(
                    red: CGFloat(hash & 255) / 255.0,
                    green: CGFloat((hash >> 8) & 255) / 255.0,
                    blue: CGFloat((hash >> 16) & 255) / 255.0,
                    alpha: alpha
                )
            }
        }
    }
}

extension CIColor {

    func withAlpha(_ alpha: CGFloat) -> CIColor {
        CIColor(red: red, green: green, blue: blue, alpha: alpha)
    }

    var resolvedCGColor: CGColor {
        CGColor(
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            components: [red, green, blue, alpha]
        )!
    }
}
