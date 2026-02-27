//
//  Pipeline.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 13.12.2025.
//

import MLXNN
import MLX

public protocol Predictor: AnyObject {

    associatedtype Input
    associatedtype Output

    func predict(_ input: Input) throws -> Output
}

public protocol Processor: AnyObject {

    associatedtype Request
    associatedtype ModelInput
    associatedtype ModelOutput
    associatedtype ProcessedOutput

    func preprocess(_ request: Request) throws -> ModelInput
    func postprocess(_ input: ModelOutput, _ request: Request) throws -> ProcessedOutput
}

/// Type-erased inference pipeline composed of a processor and model.
public class AnyPipeline<Input, Output> {

    public let model: any Predictor & MLXNN.Module
    public let processor: any Processor
    public let process: (Input) throws -> Output

    init<M: Predictor & Module, P: Processor>(
        model: M,
        processor: P
    ) where P.Request == Input, M.Input == P.ModelInput, M.Output == P.ModelOutput, P.ProcessedOutput == Output {
        self.model = model
        self.processor = processor
        self.process = { [unowned model, unowned processor] input in
            let modelInput = try processor.preprocess(input)
            let modelOutput = try model.predict(modelInput)
            let processedOutput = try processor.postprocess(modelOutput, input)
            return processedOutput
        }
    }

    /// Runs the full pipeline using function-call syntax.
    public func callAsFunction(_ input: Input) throws -> Output {
        try process(input)
    }
}
