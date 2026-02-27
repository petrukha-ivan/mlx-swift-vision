//
//  Dictionary+Inverted.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 05.11.2025.
//

import Foundation

extension Dictionary where Value: Hashable {
    var inverted: Dictionary<Value, Key> {
        Dictionary<Value, Key>(uniqueKeysWithValues: map({ ($1, $0) }))
    }
}

extension Dictionary where Key == Int {
    var flattened: [Value] {
        sorted(by: { $0.key < $1.key }).map({ $0.1 })
    }
}
