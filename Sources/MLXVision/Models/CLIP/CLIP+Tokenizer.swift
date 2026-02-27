//
//  CLIP+Tokenizer.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 18.02.2026.
//

import Foundation

class CLIPTokenizer {

    struct Bigram: Hashable {
        let first: String
        let second: String
    }

    let bos, eos: Int
    let vocabulary: [String: Int]
    let mergeRanks: [Bigram: Int]

    let patterns = (
        sequentialSpaces: #/\s+/#,
        wordParts: #/<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/#
    )

    init(merges: [String], vocabulary: [String: Int]) {
        self.bos = vocabulary["<|startoftext|>"]!
        self.eos = vocabulary["<|endoftext|>"]!
        self.vocabulary = vocabulary
        self.mergeRanks = Dictionary(
            uniqueKeysWithValues:
                merges
                .map { $0.split(separator: " ").map(String.init) }
                .map { Bigram(first: $0[0], second: $0[1]) }
                .enumerated().map { ($0.element, $0.offset) }
        )
    }

    static func from(url: URL) throws -> CLIPTokenizer {
        let vocabulary = try [String: Int].decoded(from: url, filename: "vocab.json")
        let mergesURL = url.appendingPathComponent("merges.txt")
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let merges = Array(mergesText.components(separatedBy: .newlines).dropFirst().dropLast())
        return CLIPTokenizer(merges: merges, vocabulary: vocabulary)
    }

    func encode(text: String) -> [Int] {
        let text = text.lowercased().replacing(patterns.sequentialSpaces, with: " ")
        let words = text.matches(of: patterns.wordParts).map { String(text[$0.range]) }
        let tokens = words.flatMap(tokenize)
        return [bos] + tokens.compactMap({ vocabulary[$0] }) + [eos]
    }

    private func tokenize(word: String) -> [String] {
        var tokens = word.map { String($0) }
        let last = tokens.removeLast()
        tokens.append("\(last)</w>")

        while true {
            var bestRank = Int.max
            var bestBigram: Bigram?
            for i in 0..<(tokens.count - 1) {
                let bigram = Bigram(first: tokens[i], second: tokens[i + 1])
                if let rank = mergeRanks[bigram], rank < bestRank {
                    bestRank = rank
                    bestBigram = bigram
                }
            }

            guard let bestBigram else {
                break
            }

            var i = 0
            while i < tokens.count - 1 {
                if tokens[i] == bestBigram.first, tokens[i + 1] == bestBigram.second {
                    tokens.replaceSubrange(i...(i + 1), with: [bestBigram.first + bestBigram.second])
                } else {
                    i += 1
                }
            }
        }

        return tokens
    }
}
