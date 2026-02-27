//
//  ThresholdSlider.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 22.02.2026.
//

import SwiftUI

struct ThresholdSlider: View {

    let title: String
    let description: String
    @Binding var value: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.caption)
                Spacer()
                Text(value.formatted(.number.precision(.fractionLength(2))))
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            Slider(value: $value, in: 0...1.0, step: 0.05) {
                EmptyView()
            } tick: { value in
                guard value == 0.0 || value.truncatingRemainder(dividingBy: 0.25) == 0 else {
                    return nil
                }

                return SliderTick(value) {
                    Text(value.formatted(.number.precision(.fractionLength(2))))
                }
            }

            Text(description)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}
