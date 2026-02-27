//
//  InputSourceState.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 21.02.2026.
//

import SwiftUI
import CoreImage
import Photos

@MainActor
@Observable
class InputSourceState {

    enum Mode: String, CaseIterable {
        case image = "Image"
        case camera = "Camera"
    }

    var image: CIImage?
    var mode: Mode = .image {
        didSet {
            Task {
                switch mode {
                case .image:
                    cameraManager.stop()
                case .camera:
                    try await cameraManager.start()
                }
            }
        }
    }

    //    private(set) var cameraPreviewImage: CIImage?
    //    private(set) var photoPreviewImage: CIImage?
    private(set) var errorMessage: String?

    private let cameraManager = CameraManager()

    init() {
        Task {
            for await frame in cameraManager.previewStream {
                switch mode {
                case .image:
                    break
                case .camera:
                    image = frame.centerCropped()
                }
            }
        }
    }

    //    func load(_ url: URL) throws {
    //        let data = try Data(contentsOf: url)
    //        let image = CIImage(data: data, options: [.applyOrientationProperty: true])
    //    }

    func updateImageFromFileData(_ data: Data) {
        image = CIImage(data: data, options: [.applyOrientationProperty: true])
    }

    func updateImageFromPhotoData(_ data: Data) {
        image = CIImage(data: data, options: [.applyOrientationProperty: true])?.centerCropped()
    }
}
