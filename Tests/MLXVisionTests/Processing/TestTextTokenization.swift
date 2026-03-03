//
//  TestImageResizing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 18.02.2026.
//

import Testing
import Foundation
import Hub
@testable import MLXVision

struct TokenizationResult: Codable {
    let text: String
    let tokens: [Int]
}

@Test func testCLIPTokenizationAlignedWithTransformers() async throws {
    let resultsURL = try #require(Bundle.module.url(forResource: "clip_tokenizer_results", withExtension: "json"))
    let results = try [TokenizationResult].decoded(from: resultsURL)
    let hubApi = HubApi(downloadBase: URL.temporaryDirectory)
    let tokenizerURL = try await hubApi.snapshot(from: "openai/clip-vit-base-patch16", matching: ["*.txt", "*.json"])
    let tokenizer = try CLIPTokenizer.from(url: tokenizerURL)
    for result in results {
        #expect(result.tokens == tokenizer.encode(text: result.text), "\"\(result.text)\" encoding failed")
    }
}

@Test func testSigLIPTokenizationAlignedWithTransformers() async throws {
    let resultsURL = try #require(Bundle.module.url(forResource: "siglip_tokenizer_results", withExtension: "json"))
    let results = try [TokenizationResult].decoded(from: resultsURL)
    let hubApi = HubApi(downloadBase: URL.temporaryDirectory)
    let tokenizerURL = try await hubApi.snapshot(from: "google/siglip-base-patch16-224", matching: ["*.txt", "*.json"])
    let tokenizer = try SigLIPTokenizer.from(url: tokenizerURL)
    for result in results {
        #expect(result.tokens == tokenizer.encode(text: result.text), "\"\(result.text)\" encoding failed")
    }
}
