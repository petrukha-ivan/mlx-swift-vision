//
//  Measure.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 05.03.2026.
//

import Foundation

package func measure<T>(_ label: String, terminator: String = "\n", operation: () async throws -> T) async rethrows -> T {
    let clock = ContinuousClock()
    let start = clock.now
    let value = try await operation()
    let duration = clock.now - start
    print("\(label) finished (\(duration.formatted(.units(allowed: [.minutes, .seconds, .milliseconds]))))", terminator: terminator)
    return value
}

package func measure<T>(_ label: String, terminator: String = "\n", operation: () throws -> T) rethrows -> T {
    let clock = ContinuousClock()
    let start = clock.now
    let value = try operation()
    let duration = clock.now - start
    print("\(label) finished (\(duration.formatted(.units(allowed: [.minutes, .seconds, .milliseconds]))))", terminator: terminator)
    return value
}
