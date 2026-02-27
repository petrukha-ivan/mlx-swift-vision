//
//  ModelFactory+Registry.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.12.2025.
//

import Foundation

struct ModelRegistry: Sendable {

    typealias Creator = @Sendable (URL, ModelOverrides) throws -> Any

    private let creators: [String: Creator]

    init(creators: [String: Creator]) {
        self.creators = creators
    }

    func create(
        modelDirectory: URL,
        modelType: String,
        taskName: String,
        overrides: ModelOverrides
    ) throws -> Any {
        if let creator = creators[modelType] {
            try creator(modelDirectory, overrides)
        } else {
            throw ModelFactoryError.unsupportedModelType(task: taskName, modelType: modelType)
        }
    }
}

extension ModelRegistry {

    static var classificationModelsRegistry: ModelRegistry {
        ModelRegistry(creators: [
            "resnet": { url, overrides in
                let modelConfig = try ResNetForImageClassificationConfig.decoded(from: url.modelConfig)
                let model = ResNetModelForImageClassification(modelConfig)
                let imageProcessorConfig = try ImagePreprocessorConfig.decoded(from: url.preprocessorConfig)
                let imagePreprocessor = ImagePreprocessor(imageProcessorConfig, overrides: overrides)
                let processor = ResNetProcessor(modelConfig: modelConfig, imagePreprocessor: imagePreprocessor)
                return AnyModelForImageClassification(model: model, processor: processor)
            },
            "efficientnet": { url, overrides in
                let modelConfig = try EfficientNetForImageClassificationConfig.decoded(from: url.modelConfig)
                let model = EfficientNetModelForImageClassification(modelConfig)
                let imageProcessorConfig = try ImagePreprocessorConfig.decoded(from: url.preprocessorConfig)
                let imagePreprocessor = ImagePreprocessor(imageProcessorConfig, overrides: overrides)
                let processor = EfficientNetProcessor(modelConfig: modelConfig, imagePreprocessor: imagePreprocessor)
                return AnyModelForImageClassification(model: model, processor: processor)
            },
        ])
    }

    static var objectDetectionModelsRegistry: ModelRegistry {
        ModelRegistry(creators: [
            "detr": { url, overrides in
                let modelConfig = try DetrForObjectDetectionConfig.decoded(from: url.modelConfig)
                let model = DetrModelForObjectDetection(modelConfig)
                let imageProcessorConfig = try ImagePreprocessorConfig.decoded(from: url.preprocessorConfig)
                let imagePreprocessor = ImagePreprocessor(imageProcessorConfig, overrides: overrides)
                let processor = DetrForObjectDetectionProcessor(modelConfig: modelConfig, imagePreprocessor: imagePreprocessor)
                return AnyModelForObjectDetection(model: model, processor: processor)
            }
        ])
    }

    static var imageSegmentationModelsRegistry: ModelRegistry {
        ModelRegistry(creators: [
            "detr": { url, overrides in
                let modelConfig = try DetrForObjectDetectionConfig.decoded(from: url.modelConfig)
                let model = DetrModelForImageSegmentation(modelConfig)
                let imageProcessorConfig = try ImagePreprocessorConfig.decoded(from: url.preprocessorConfig)
                let imagePreprocessor = ImagePreprocessor(imageProcessorConfig, overrides: overrides)
                let processor = DetrForImageSegmentationProcessor(modelConfig: modelConfig, imagePreprocessor: imagePreprocessor)
                return AnyModelForImageSegmentation(model: model, processor: processor)
            }
        ])
    }

    static var embeddingsExtractionModelsRegistry: ModelRegistry {
        ModelRegistry(creators: [
            "clip": { url, overrides in
                let modelConfig = try CLIPConfig.decoded(from: url.modelConfig)
                let model = CLIPEmbeddingsModel(modelConfig)
                let imageProcessorConfig = try ImagePreprocessorConfig.decoded(from: url.preprocessorConfig)
                let imagePreprocessor = ImagePreprocessor(imageProcessorConfig, overrides: overrides)
                let textTokenizer = try CLIPTokenizer.from(url: url)
                let processor = CLIPEmbeddingsProcessor(imagePreprocessor: imagePreprocessor, textTokenizer: textTokenizer)
                return AnyModelForEmbeddingsExtraction(model: model, processor: processor)
            }
        ])
    }

    static var zeroShotClassificationModelsRegistry: ModelRegistry {
        ModelRegistry(creators: [
            "clip": { url, overrides in
                let modelConfig = try CLIPConfig.decoded(from: url.modelConfig)
                let model = CLIPModel(modelConfig)
                let imageProcessorConfig = try ImagePreprocessorConfig.decoded(from: url.preprocessorConfig)
                let imagePreprocessor = ImagePreprocessor(imageProcessorConfig, overrides: overrides)
                let textTokenizer = try CLIPTokenizer.from(url: url)
                let processor = CLIPProcessor(modelConfig: modelConfig, imagePreprocessor: imagePreprocessor, textTokenizer: textTokenizer)
                return AnyModelForZeroShotClassification(model: model, processor: processor)
            }
        ])
    }

    static var zeroShotSegmentationModelsRegistry: ModelRegistry {
        ModelRegistry(creators: [
            "sam3_video": { url, overrides in
                let modelConfig = try Sam3Config.decoded(from: url.modelConfig).with(overrides: overrides)
                let model = Sam3Model(modelConfig)
                let processorConfig = try Sam3ProcessorConfig.decoded(from: url.processorConfig)
                let imagePreprocessor = ImagePreprocessor(processorConfig.imageProcessor, overrides: overrides)
                let textTokenizer = try CLIPTokenizer.from(url: url)
                let processor = Sam3Processor(imagePreprocessor: imagePreprocessor, textTokenizer: textTokenizer)
                return AnyModelForZeroShotSegmentation(model: model, processor: processor)
            }
        ])
    }
}

fileprivate extension URL {

    var modelConfig: URL {
        self.appending(component: "config.json")
    }

    var preprocessorConfig: URL {
        self.appending(component: "preprocessor_config.json")
    }

    var processorConfig: URL {
        self.appending(component: "processor_config.json")
    }

    var tokenizerConfig: URL {
        self.appending(component: "tokenizer_config.json")
    }
}
