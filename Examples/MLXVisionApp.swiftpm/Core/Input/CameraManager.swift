//
//  CameraManager.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 21.02.2026.
//

import AVFoundation
import CoreImage

enum CameraManagerError: LocalizedError {
    case unauthorized
    case cameraUnavailable
}

class CameraManager: NSObject {

    private var previewStreamContinuation: AsyncStream<CIImage>.Continuation?
    lazy var previewStream = AsyncStream<CIImage> { continuation in
        previewStreamContinuation = continuation
    }

    private let captureSession = AVCaptureSession()
    private let captureOutputQueue = DispatchQueue(label: "com.mlx-swift-vision.CaptureOutputQueue")

    private var captureDevice: AVCaptureDevice?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var deviceInput: AVCaptureDeviceInput?
    private var videoOutput: AVCaptureVideoDataOutput?

    func start() async throws {
        try await checkAuthorization()
        try configureCaptureSession()
        captureSession.startRunning()
    }

    func stop() {
        captureSession.stopRunning()
    }
}

extension CameraManager {

    func checkAuthorization() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined where await AVCaptureDevice.requestAccess(for: .video):
            break
        default:
            throw CameraManagerError.unauthorized
        }
    }

    func configureCaptureSession() throws {
        guard captureDevice == nil else {
            return
        }

        captureSession.beginConfiguration()
        defer {
            captureSession.commitConfiguration()
        }

        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            throw CameraManagerError.cameraUnavailable
        }

        let rotationCoordinator = AVCaptureDevice.RotationCoordinator(device: captureDevice, previewLayer: nil)
        let deviceInput = try AVCaptureDeviceInput(device: captureDevice)
        let videoOutput = AVCaptureVideoDataOutput()

        captureSession.addInput(deviceInput)
        captureSession.addOutput(videoOutput)

        videoOutput.connection(with: .video)?.videoRotationAngle = rotationCoordinator.videoRotationAngleForHorizonLevelCapture
        videoOutput.setSampleBufferDelegate(self, queue: captureOutputQueue)

        self.captureDevice = captureDevice
        self.rotationCoordinator = rotationCoordinator
        self.deviceInput = deviceInput
        self.videoOutput = videoOutput
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if let imageBuffer = sampleBuffer.imageBuffer {
            previewStreamContinuation?.yield(CIImage(cvPixelBuffer: imageBuffer))
        }
    }
}
