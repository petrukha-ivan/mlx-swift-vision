//
//  ZeroShotClassificationView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 21.02.2026.
//

import SwiftUI
import MLXVision

struct ZeroShotClassificationView: View {

    @State var labels: String = "cat, dog, car, phone, laptop"
    @State var modelRunner: ModelRunner<ZeroShotClassificationTask, [ClassificationResult]>
    @Environment(InputSourceState.self) var inputState

    init(model: AnyModelForZeroShotClassification) {
        self.modelRunner = ModelRunner(model: model) { _, results in
            results.top(5)
        }
    }

    var body: some View {
        AdaptiveStack {
            Section("Input") {
                InputSourceView()
            }
            Section {
                TextField("Labels", text: $labels)
                    .autocorrectionDisabled()
            } header: {
                Text("Options")
            } footer: {
                Text("Comma-separated values")
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
        .task(id: [labels]) {
            await submit(debounce: true)
        }
        .task(id: inputState.image) {
            await submit()
        }
    }

    func submit(debounce: Bool = false) async {
        let labels = labels.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard let image = inputState.image, !labels.isEmpty else {
            return
        }

        let request = ZeroShotClassificationRequest(image: image, labels: labels)
        await modelRunner.submit(request, debounce: debounce)
    }
}
