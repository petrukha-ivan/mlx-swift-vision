//
//  AppSettings.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 28.03.2026.
//

import SwiftUI
import MLX

@Observable
final class AppSettings {

    static let tokenKey = "com.MLXVisionApp.token"
    static let cacheLimitKey = "com.MLXVisionApp.cacheLimit"

    static let deviceInfo = GPU.deviceInfo()
    static var deviceMemorySize: Double {
        Double(deviceInfo.memorySize)
    }

    @ObservationIgnored
    var token: String {
        get {
            access(keyPath: \.token)
            return _token
        }
        set {
            withMutation(keyPath: \.token) {
                _token = newValue
            }
        }
    }

    @ObservationIgnored
    var cacheLimit: Double {
        get {
            access(keyPath: \.cacheLimit)
            return _cacheLimit
        }
        set {
            withMutation(keyPath: \.cacheLimit) {
                _cacheLimit = newValue
                applyCacheLimit()
            }
        }
    }

    @ObservationIgnored
    @AppStorage(tokenKey)
    private var _token: String = ""

    @ObservationIgnored
    @AppStorage(cacheLimitKey)
    private var _cacheLimit: Double = 0

    func applyCacheLimit() {
        Memory.cacheLimit = Int(cacheLimit)
    }
}
