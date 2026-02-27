//
//  ImageClassificationView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 20.02.2026.
//

import SwiftUI
import MLXVision

struct ImageClassificationView: View {

    @State var modelRunner: ModelRunner<ImageClassificationTask, [ClassificationResult]>
    @Environment(InputSourceState.self) var inputState

    init(model: AnyModelForImageClassification) {
        self.modelRunner = ModelRunner(model: model) { _, results in
            results.top(5)
        }
    }

    var body: some View {
        AdaptiveStack {
            Section("Input") {
                InputSourceView()
            }
        } content: {
            Section("Results") {
                DetectionsView(results: modelRunner.result ?? [])
            }
        } footer: {
            Section("Performance") {
                PerformanceView(performance: modelRunner.performance)
            }
        }
        .task(id: inputState.image) {
            await submit()
        }
    }

    func submit() async {
        guard let image = inputState.image else {
            return
        }

        let request = ImageClassificationRequest(image: image)
        await modelRunner.submit(request)
    }
}
