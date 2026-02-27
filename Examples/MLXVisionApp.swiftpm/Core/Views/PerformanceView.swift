//
//  PerformanceView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 20.02.2026.
//

import SwiftUI

struct PerformanceView: View {

    let performance: ModelRunnerPerformance

    var body: some View {
        Group {

            LabeledContent("Latest Processing") {
                performance.latestProcessingTime.map {
                    Text($0.formatted(.units(allowed: [.seconds, .milliseconds])))
                }
            }

            LabeledContent("Average Processing") {
                performance.averageProcessingTime.map {
                    Text($0.formatted(.units(allowed: [.seconds, .milliseconds])))
                }
            }

            LabeledContent("Frames per Second") {
                performance.framesPerSecond.map {
                    Text($0.formatted(.number.precision(.fractionLength(1))))
                }
            }
        }
    }
}
