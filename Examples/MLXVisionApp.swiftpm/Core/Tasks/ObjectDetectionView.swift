//
//  ObjectDetectionView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 21.02.2026.
//

import SwiftUI
import MLXVision

struct ObjectDetectionView: View {

    @State var scoreThreshold: Float = 0.75
    @State var modelRunner: ModelRunner<ObjectDetectionTask, (results: [ObjectDetectionResult], annotatedImage: CIImage)>
    @Environment(InputSourceState.self) var inputState

    init(model: AnyModelForObjectDetection) {
        let annotator = ComposedAnnotator<ObjectDetectionResult>(BoxAnnotator(), LabelAnnotator())
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
                    description: "Higher values keep only more confident detections",
                    value: $scoreThreshold
                )
            }
        } content: {
            Section("Results") {
                ImagePlaceholder(
                    image: modelRunner.result?.annotatedImage,
                    emptyTitle: "Detection Results",
                    emptySystemImage: "viewfinder.rectangular",
                    emptyDescription: "Run inference to see labeled boxes"
                )
            }
        } footer: {
            Section("Performance") {
                PerformanceView(performance: modelRunner.performance)
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

        let request = ObjectDetectionRequest(image: image, scoreThreshold: scoreThreshold)
        await modelRunner.submit(request, debounce: debounce)
    }
}
