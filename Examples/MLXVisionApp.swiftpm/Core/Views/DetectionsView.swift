//
//  DetectionsView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 22.02.2026.
//

import SwiftUI
import MLXVision

struct DetectionsView: View {

    typealias Item = LabeledResult & ScoredResult

    let results: [Item]

    var body: some View {
        ForEach(0..<5) { i in
            HStack(spacing: 8) {
                if results.indices.contains(i) {
                    row(results[i])
                } else {
                    EmptyView()
                }
            }
        }
    }

    @ViewBuilder
    func row(_ item: Item) -> some View {
        Group {

            Text(item.label)
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer(minLength: 8)

            Text(item.score.formatted(.number.precision(.fractionLength(2))))
                .lineLimit(1)
                .truncationMode(.middle)
                .layoutPriority(1)
        }
    }
}
