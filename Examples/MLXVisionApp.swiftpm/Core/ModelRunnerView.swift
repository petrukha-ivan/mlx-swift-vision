//
//  ModelRunnerView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 20.02.2026.
//

import SwiftUI
import MLXVision
import Hub

enum ModelState {
    case waiting
    case loading(Double)
    case loaded(ModelContainer)
    case failed(Error)
}

enum ModelContainer {
    case imageClassification(AnyModelForImageClassification)
    case objectDetection(AnyModelForObjectDetection)
    case imageSegmentation(AnyModelForImageSegmentation)
    case zeroShotClassification(AnyModelForZeroShotClassification)
    case zeroShotSegmentation(AnyModelForZeroShotSegmentation)
}

struct ModelRunnerView: View {

    @State var modelSelection: ModelSelection
    @State var modelState: ModelState = .waiting
    @State var inputState = InputSourceState()
    @AppStorage("token") var token: String = ""

    var body: some View {
        Group {
            switch modelState {
            case .waiting:
                List {
                    Section("Loading") {
                        ProgressView(value: 0.0)
                    }
                }
            case .loading(let progress):
                List {
                    Section("Loading") {
                        ProgressView(value: progress)
                    }
                }
            case .loaded(let modelContainer):
                switch modelContainer {
                case .imageClassification(let model):
                    ImageClassificationView(model: model)
                case .objectDetection(let model):
                    ObjectDetectionView(model: model)
                case .imageSegmentation(let model):
                    ImageSegmentationView(model: model)
                case .zeroShotClassification(let model):
                    ZeroShotClassificationView(model: model)
                case .zeroShotSegmentation(let model):
                    ZeroShotSegmentationView(model: model)
                }
            case .failed(let error):
                List {
                    Section {
                        Text("Error description: \(error.localizedDescription)")
                    } header: {
                        Text("Loading")
                    } footer: {
                        Text("Check access token if the model has gated access")
                    }
                }
            }
        }
        .environment(inputState)
        .task(id: modelSelection) {
            do {
                try await loadModelContainer()
            } catch {
                modelState = .failed(error)
            }
        }
    }

    func loadModelContainer() async throws {
        let factory = ModelFactory.shared
        let hubApi = HubApi(hfToken: token)
        let source = ModelSource.hub(id: modelSelection.id, revision: modelSelection.revision, hubApi: hubApi)
        let overrides = ModelOverrides(
            inputSize: modelSelection.inputSize.map(CGFloat.init).map({ CGSize(width: $0, height: $0) }),
            quantizeBits: modelSelection.quantizeBits
        )

        let progressHandler: @Sendable (Double) -> Void = { progress in
            Task { @MainActor in
                self.modelState = .loading(progress)
            }
        }

        let modelContainer: ModelContainer =
            switch modelSelection.type {
            case .imageClassification:
                try await .imageClassification(
                    factory.load(source, for: ImageClassificationTask.self, overrides: overrides, progressHandler: progressHandler)
                )
            case .objectDetection:
                try await .objectDetection(
                    factory.load(source, for: ObjectDetectionTask.self, overrides: overrides, progressHandler: progressHandler)
                )
            case .imageSegmentation:
                try await .imageSegmentation(
                    factory.load(source, for: ImageSegmentationTask.self, overrides: overrides, progressHandler: progressHandler)
                )
            case .zeroShotClassification:
                try await .zeroShotClassification(
                    factory.load(source, for: ZeroShotClassificationTask.self, overrides: overrides, progressHandler: progressHandler)
                )
            case .zeroShotSegmentation:
                try await .zeroShotSegmentation(
                    factory.load(source, for: ZeroShotSegmentationTask.self, overrides: overrides, progressHandler: progressHandler)
                )
            }

        await MainActor.run {
            self.modelState = .loaded(modelContainer)
        }
    }
}

@MainActor
@Observable
class ModelRunner<T: VisionTask, Value> {

    typealias Input = T.Input
    typealias Output = T.Output

    private(set) var result: Value?
    private(set) var performance = ModelRunnerPerformance(
        latestProcessingTime: nil,
        averageProcessingTime: nil,
        framesPerSecond: nil
    )

    private let model: AnyPipeline<Input, Output>
    private let transform: @Sendable (Input, Output) throws -> Value

    private var pendingInput: Input?
    private var executionTask: Task<Void, Never>?
    private var recentProcessingTimes: [Duration] = []

    init(
        model: AnyPipeline<Input, Output>,
        transform: @escaping @Sendable (Input, Output) throws -> Value
    ) {
        self.model = model
        self.transform = transform
    }

    func submit(_ input: Input, debounce: Bool = false) async {
        if debounce {
            do {
                try await Task.sleep(for: .milliseconds(300))
            } catch {
                return
            }
        }

        pendingInput = input
        processPendingInput()
    }

    private func processPendingInput() {
        guard executionTask == nil, let input = pendingInput else {
            return
        }

        pendingInput = nil
        executionTask = Task.detached(priority: .userInitiated) { [model, transform, input] in
            defer {
                Task { @MainActor [weak self] in
                    self?.executionTask = nil
                    self?.processPendingInput()
                }
            }

            do {
                let clock = ContinuousClock()
                let start = clock.now
                let output = try model(input)
                let transformed = try transform(input, output)
                let processingTime = clock.now - start
                await MainActor.run { [weak self] in
                    self?.result = transformed
                    self?.updatePerformance(processingTime: processingTime)
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.result = nil
                }
            }
        }
    }

    private func updatePerformance(processingTime: Duration) {
        recentProcessingTimes.append(processingTime)
        if recentProcessingTimes.count > 100 {
            recentProcessingTimes.removeFirst()
        }

        let averageProcessingTime = recentProcessingTimes.reduce(.seconds(0), +) / recentProcessingTimes.count
        let framesPerSecond = Duration.seconds(1) / averageProcessingTime

        performance = ModelRunnerPerformance(
            latestProcessingTime: processingTime,
            averageProcessingTime: averageProcessingTime,
            framesPerSecond: framesPerSecond
        )
    }
}

struct ModelRunnerPerformance: Equatable {
    let latestProcessingTime: Duration?
    let averageProcessingTime: Duration?
    let framesPerSecond: Double?
}
