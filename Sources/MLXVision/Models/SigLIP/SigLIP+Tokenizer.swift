//
//  SigLIP+Tokenizer.swift
//  mlx-swift-vision
//
//  Created by Ivan Petrukha on 02.03.2026.
//

import Foundation
import ReerCodable

@Codable
@CodingContainer("model")
struct SigLIPTokenizerInfo {

    @CodingKey("vocab")
    let vocabulary: [VocabularyEntry]
}

struct VocabularyEntry: Codable {

    let token: String
    let score: Float

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        token = try container.decode(String.self)
        score = try container.decode(Float.self)
    }
}

class SigLIPTokenizer {

    struct TokenInfo {
        let id: Int
        let score: Float
    }

    class TrieNode {
        var children: [Character: TrieNode] = [:]
        var tokenInfo: TokenInfo? = nil
    }

    let pad, eos, unk: Int
    var trie = TrieNode()

    let patterns = (
        sequentialSpaces: #/\s+/#,
        wordParts: #/'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/#,
    )

    init(vocabulary: [VocabularyEntry]) {
        self.pad = vocabulary.firstIndex(where: { $0.token == "<pad>" }) ?? 0
        self.eos = vocabulary.firstIndex(where: { $0.token == "</s>" }) ?? 1
        self.unk = vocabulary.firstIndex(where: { $0.token == "<unk>" }) ?? 2
        for (id, entry) in vocabulary.enumerated() {
            var node = trie
            let tokenInfo = TokenInfo(id: id, score: entry.score)
            defer {
                node.tokenInfo = tokenInfo
            }

            for character in entry.token {
                if let next = node.children[character] {
                    node = next
                } else {
                    let next = TrieNode()
                    node.children[character] = next
                    node = next
                }
            }
        }
    }

    static func from(url: URL) throws -> SigLIPTokenizer {
        let tokenizerInfo = try SigLIPTokenizerInfo.decoded(from: url, filename: "tokenizer.json")
        let vocabulary = tokenizerInfo.vocabulary
        return SigLIPTokenizer(vocabulary: vocabulary)
    }

    func encode(text: String) -> [Int] {
        let text = text.lowercased().replacing(patterns.sequentialSpaces, with: " ").filter { !$0.isPunctuation }
        let words = text.matches(of: patterns.wordParts).map { "▁" + String(text[$0.range]) }
        let tokens = words.flatMap(tokenize)
        return tokens + [eos]
    }

    func tokenize(word: String) -> [Int] {
        let characters = Array(word)
        let n = characters.count

        var dp = [0.0] + Array(repeating: -Float.infinity, count: n)
        var back = Array<(previous: Int, info: TokenInfo)?>(repeating: nil, count: n + 1)

        for i in 0..<n {
            guard dp[i] > -Float.infinity else {
                continue
            }

            var j = i
            var node = trie
            while j < n {
                let character = characters[j]
                guard let next = node.children[character] else {
                    break
                }

                j += 1
                node = next
                if let info = node.tokenInfo {
                    let score = dp[i] + info.score
                    if score > dp[j] {
                        dp[j] = score
                        back[j] = (i, info)
                    }
                }
            }
        }

        var tokens: [Int] = []
        var current = n
        while current > 0, let step = back[current] {
            tokens.append(step.info.id)
            current = step.previous
        }

        return tokens.reversed()
    }
}
