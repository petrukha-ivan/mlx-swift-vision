//
//  ModelFactory.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 02.11.2025.
//

import Foundation
import Hub
import MLX
import MLXNN
import ReerCodable

/// Errors returned by `ModelFactory` during pipeline loading.
public enum ModelFactoryError: Error {
    /// No registry exists for the requested task marker type.
    case unknownTask(task: String)
    /// The resolved `model_type` is not supported by the selected task registry.
    case unsupportedModelType(task: String, modelType: String)
    /// A registry creator returned a pipeline with incompatible generic input/output types.
    case incompatiblePipeline(task: String, modelType: String)
}

extension ModelFactoryError: LocalizedError {

    /// Human-readable error description.
    public var errorDescription: String? {
        switch self {
        case .unknownTask(let task):
            return "ModelFactory does not have a registry for task \(task)."
        case .unsupportedModelType(let task, let modelType):
            return "Model type \(modelType) is not supported for task \(task)."
        case .incompatiblePipeline(let task, let modelType):
            return "Created pipeline for model type \(modelType) is incompatible with task \(task)."
        }
    }
}

/// Source descriptor for loading model assets.
public enum ModelSource: ExpressibleByStringLiteral {
    /// Load model assets from Hugging Face Hub.
    case hub(id: String, revision: String, hubApi: HubApi = .shared)
    /// Load model assets from an existing local directory.
    case directory(URL)

    /// Creates a Hub source descriptor.
    public init(id: String, revision: String = "main", hubApi: HubApi = .shared) {
        self = .hub(id: id, revision: revision, hubApi: hubApi)
    }

    /// Creates a local-directory source descriptor.
    public init(directory: URL) {
        self = .directory(directory)
    }

    /// Creates a Hub source descriptor from a string literal model ID.
    public init(stringLiteral value: String) {
        self = .hub(id: value, revision: "main", hubApi: .shared)
    }
}

public struct ModelOverrides: Equatable, Hashable {

    public let inputSize: CGSize?
    public let modelDtype: DType?
    public let quantizeBits: Int?
    public let quantizeGroupSize: Int

    public init(
        inputSize: CGSize? = nil,
        modelDtype: DType? = .bfloat16,
        quantizeBits: Int? = nil,
        quantizeGroupSize: Int = 64
    ) {
        self.inputSize = inputSize
        self.modelDtype = modelDtype
        self.quantizeBits = quantizeBits
        self.quantizeGroupSize = quantizeGroupSize
    }
}

/// Loads typed inference pipelines for task marker types.
public final class ModelFactory: Sendable {

    /// Shared default instance.
    public static let shared = ModelFactory()

    private let registries: [ObjectIdentifier: ModelRegistry] = [
        ObjectIdentifier(ImageClassificationTask.self): .classificationModelsRegistry,
        ObjectIdentifier(ObjectDetectionTask.self): .objectDetectionModelsRegistry,
        ObjectIdentifier(InstanceSegmentationTask.self): .instanceSegmentationModelsRegistry,
        ObjectIdentifier(EmbeddingsExtractionTask.self): .embeddingsExtractionModelsRegistry,
        ObjectIdentifier(ZeroShotClassificationTask.self): .zeroShotClassificationModelsRegistry,
        ObjectIdentifier(ZeroShotSegmentationTask.self): .zeroShotSegmentationModelsRegistry,
    ]

    /// Loads a model from an explicit source descriptor.
    public func load<T: VisionTask>(
        _ source: ModelSource,
        for task: T.Type,
        overrides: ModelOverrides = ModelOverrides(),
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws -> AnyPipeline<T.Input, T.Output> {
        let modelDirectory: URL =
            switch source {
            case .directory(let directory):
                directory
            case .hub(let id, let revision, let hubApi):
                try await hubApi.snapshot(
                    from: id,
                    revision: revision,
                    matching: ["*.txt", "*.json", "*.safetensors"]
                ) { progress, _ in
                    progressHandler?(progress.fractionCompleted)
                }
            }

        let modelConfigURL = modelDirectory.appending(component: "config.json")
        let modelConfigData = try Data(contentsOf: modelConfigURL)
        let modelConfig = try ModelConfig.decoded(from: modelConfigData)
        let modelType = modelConfig.modelType

        let taskName = String(describing: task)
        guard let registry = registries[ObjectIdentifier(task)] else {
            throw ModelFactoryError.unknownTask(task: taskName)
        }

        let entry = try registry.create(modelDirectory: modelDirectory, modelType: modelType, taskName: taskName, overrides: overrides)
        guard let pipeline = entry as? AnyPipeline<T.Input, T.Output> else {
            throw ModelFactoryError.incompatiblePipeline(task: taskName, modelType: modelType)
        }

        let weightsURL = modelDirectory.appending(component: "model.safetensors")
        var weights = try MLX.loadArrays(url: weightsURL)
        if let modelDtype = overrides.modelDtype {
            weights = weights.mapValues {
                $0.dtype.isFloatingPoint ? $0.asType(modelDtype) : $0
            }
        }

        let parameters = ModuleParameters.unflattened(weights)
        try pipeline.model.update(parameters: parameters, verify: .all)
        pipeline.model.train(false)

        if let quantizeBits = overrides.quantizeBits {
            let quantizeGroupSize = overrides.quantizeGroupSize
            quantize(
                model: pipeline.model,
                groupSize: quantizeGroupSize,
                bits: quantizeBits,
                filter: { key, module in
                    if let linear = module as? Linear {
                        return linear.weight.dim(-1) % quantizeGroupSize == 0 && key.components(separatedBy: ".").last.flatMap(Int.init) == nil
                    } else {
                        return false
                    }
                }
            )
        }

        MLX.eval(pipeline.model)
        Memory.clearCache()

        return pipeline
    }
}

@Codable
private struct ModelConfig {

    @CodingKey("model_type")
    var modelType: String
}
