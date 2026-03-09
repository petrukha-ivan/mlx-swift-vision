//
//  CGImage+Render.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 07.03.2026.
//

import CoreGraphics

extension CGImage {
    static func render(size: CGSize, draw: (CGContext) -> Void) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )

        guard let context else {
            return nil
        }

        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1, y: -1)
        draw(context)

        return context.makeImage()
    }
}
