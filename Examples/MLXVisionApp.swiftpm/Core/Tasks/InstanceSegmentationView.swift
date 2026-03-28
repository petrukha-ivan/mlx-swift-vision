//
//  InstanceSegmentationView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 21.02.2026.
//

import SwiftUI
import MLXVision

struct InstanceSegmentationView: View {

    @State var scoreThreshold: Float = 0.75
    @State var modelRunner: ModelRunner<InstanceSegmentationTask, (results: [InstanceSegmentationResult], annotatedImage: CIImage)>
    @Environment(InputSourceState.self) var inputState

    init(model: AnyModelForInstanceSegmentation) {
        let annotator = MaskAnnotator<InstanceSegmentationResult>()
        self.modelRunner = ModelRunner(model: model) { input, results in
            let annotatedImage = annotator.annotate(image: input.image, detections: results)
            return (results, annotatedImage)
        }
    }

    var body: some View {
        AdaptiveStack {
            Section("Input") {
                InputSourceView()
            }
            Section("Options") {
                ThresholdSlider(
                    title: "Score Threshold",
                    description: "Higher values show only more confident segments",
                    value: $scoreThreshold
                )
            }
        } content: {
            Section("Results") {
                ImagePlaceholder(
                    image: modelRunner.result?.annotatedImage,
                    emptyTitle: "Segmentation Results",
                    emptySystemImage: "square.stack.3d.up",
                    emptyDescription: "Run inference to see mask overlays"
                )
            }
        } footer: {
            Section("Performance") {
                PerformanceView(performance: modelRunner.performance) {
                    modelRunner.resetPerformance()
                }
            }
        }
        .task(id: [scoreThreshold.description]) {
            await submit(debounce: true)
        }
        .task(id: inputState.image) {
            await submit()
        }
    }

    func submit(debounce: Bool = false) async {
        guard let image = inputState.image else {
            return
        }

        let request = InstanceSegmentationRequest(image: image, scoreThreshold: scoreThreshold)
        await modelRunner.submit(request, debounce: debounce)
    }
}
