//
//  ZeroShotSegmentationView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 21.02.2026.
//

import SwiftUI
import MLXVision

struct ZeroShotSegmentationView: View {

    @State var prompt: String = ""
    @State var scoreThreshold: Float = 0.75
    @State var maskThreshold: Float = 0.5
    @State var modelRunner: ModelRunner<ZeroShotSegmentationTask, (results: [InstanceSegmentationResult], annotatedImage: CIImage)>
    @Environment(InputSourceState.self) var inputState

    init(model: AnyModelForZeroShotSegmentation) {
        let maskAnnotator = MaskAnnotator()
        self.modelRunner = ModelRunner(model: model) { input, results in
            let annotatedImage = maskAnnotator.annotate(image: input.image, detections: results)
            return (results, annotatedImage)
        }
    }

    var body: some View {
        AdaptiveStack {
            Section("Input") {
                InputSourceView()
            }
            Section {
                TextField("Prompt", text: $prompt)
                    .autocorrectionDisabled()
            } header: {
                Text("Options")
            } footer: {
                Text("Prompt describing a target object")
            }
            Section {
                ThresholdSlider(
                    title: "Score Threshold",
                    description: "Higher values keep only more confident segments",
                    value: $scoreThreshold
                )
                ThresholdSlider(
                    title: "Mask Threshold",
                    description: "Higher values make mask coverage more strict",
                    value: $maskThreshold
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
                PerformanceView(performance: modelRunner.performance)
            }
        }
        .task(id: [prompt, scoreThreshold.description, maskThreshold.description]) {
            await submit(debounce: true)
        }
        .task(id: inputState.image) {
            await submit()
        }
    }

    func submit(debounce: Bool = false) async {
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let image = inputState.image, !prompt.isEmpty else {
            return
        }

        let request = ZeroShotSegmentationRequest(image: image, prompt: prompt, scoreThreshold: scoreThreshold, maskThreshold: maskThreshold)
        await modelRunner.submit(request, debounce: debounce)
    }
}
