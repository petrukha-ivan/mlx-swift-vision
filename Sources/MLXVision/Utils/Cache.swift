//
//  Cache.swift
//  mlx-swift-vision
//
//  Created by Ivan Petrukha on 05.04.2026.
//

import Foundation

final class Cache<Key: Hashable, Value>: @unchecked Sendable {

    private let lock = NSLock()
    private var storage: [Key: Value] = [:]

    func value(for key: Key, orInsert makeValue: () -> Value) -> Value {
        lock.lock()
        defer {
            lock.unlock()
        }

        if let value = storage[key] {
            return value
        }

        let value = makeValue()
        storage[key] = value
        return value
    }
}
