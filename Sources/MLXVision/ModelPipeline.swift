//
//  ModelPipeline.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.12.2025.
//

import MLXNN

protocol Predictor: MLXNN.Module {

    associatedtype Input
    associatedtype Output

    func predict(_ input: Input) throws -> Output
}

protocol Processor: AnyObject {

    associatedtype Request
    associatedtype ModelInput
    associatedtype ModelOutput
    associatedtype Result

    func preprocess(_ request: Request) throws -> ModelInput
    func postprocess(_ modelOutput: ModelOutput, _ request: Request) throws -> Result
}

class ModelPipeline<Model: Predictor, ModelProcessor: Processor> where Model.Input == ModelProcessor.ModelInput, Model.Output == ModelProcessor.ModelOutput {

    let model: Model
    let processor: ModelProcessor

    init(model: Model, processor: ModelProcessor) {
        self.model = model
        self.processor = processor
    }

    func process(_ request: ModelProcessor.Request) throws -> ModelProcessor.Result {
        let modelInput = try processor.preprocess(request)
        let modelOutput = try model.predict(modelInput)
        let processedOutput = try processor.postprocess(modelOutput, request)
        return processedOutput
    }

    func asAny() -> AnyPipeline<ModelProcessor.Request, ModelProcessor.Result> {
        AnyPipeline(pipeline: self)
    }
}

/// Type-erased inference pipeline composed of a processor and model.
public class AnyPipeline<Input, Output> {

    let model: any Predictor
    let processor: any Processor
    let process: (Input) throws -> Output

    init<Model, ModelProcessor>(
        pipeline: ModelPipeline<Model, ModelProcessor>
    ) where Input == ModelProcessor.Request, Output == ModelProcessor.Result {
        self.model = pipeline.model
        self.processor = pipeline.processor
        self.process = pipeline.process
    }

    public func callAsFunction(_ input: Input) throws -> Output {
        try process(input)
    }
}
