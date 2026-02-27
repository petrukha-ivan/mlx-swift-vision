//
//  ImagePlaceholder.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 22.02.2026.
//

import SwiftUI
import CoreImage

struct ImagePlaceholder: View {

    let image: CIImage?
    let emptyTitle: String
    let emptySystemImage: String
    let emptyDescription: String

    var body: some View {
        Group {
            if let image = image?.image {
                image
                    .resizable()
                    .scaledToFit()
                    .clipShape(.rect(cornerRadius: 12))
            } else {
                ContentUnavailableView(
                    emptyTitle,
                    systemImage: emptySystemImage,
                    description: Text(emptyDescription)
                )
            }
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(contentMode: .fit)
    }
}

private let context = CIContext()
private extension CIImage {
    var image: Image? {
        context.createCGImage(self, from: extent).map {
            Image(decorative: $0, scale: 1, orientation: .up)
        }
    }
}
