//
//  InputSourceView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 22.02.2026.
//

import SwiftUI
import PhotosUI

struct InputSourceView: View {

    @State var pickedPhoto: PhotosPickerItem?
    @State var showFileImporter: Bool = false
    @Environment(InputSourceState.self) var inputState

    var body: some View {
        VStack(alignment: .center) {
            @Bindable var inputState = inputState
            Picker("Input", selection: $inputState.mode) {
                ForEach(InputSourceState.Mode.allCases, id: \.self) {
                    Text($0.rawValue).tag($0)
                }
            }
            .pickerStyle(.segmented)

            switch inputState.mode {
            case .image:
                #if targetEnvironment(macCatalyst)
                    ImagePlaceholder(
                        image: inputState.image,
                        emptyTitle: "Input Image",
                        emptySystemImage: "photo",
                        emptyDescription: "Choose an image to run inference"
                    )
                    Button {
                        showFileImporter = true
                    } label: {
                        Label("Choose File", systemImage: "photo")
                    }
                #else
                    PhotosPicker("Choose Photo", selection: $pickedPhoto, matching: .images)
                        .photosPickerStyle(.inline)
                        .photosPickerAccessoryVisibility(.hidden, edges: .top)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .clipShape(.rect(cornerRadius: 12))
                #endif
            case .camera:
                ImagePlaceholder(
                    image: inputState.image,
                    emptyTitle: "Live Camera",
                    emptySystemImage: "camera",
                    emptyDescription: "Waiting for camera frames"
                )
            }
        }
        .fileImporter(isPresented: $showFileImporter, allowedContentTypes: [.image]) { result in
            do {
                let url = try result.get()
                let data = try Data(contentsOf: url)
                inputState.updateImageFromFileData(data)
            } catch {
                assertionFailure(error.localizedDescription)
            }
        }
        .task(id: pickedPhoto) {
            do {
                if let pickedPhoto, let data = try await pickedPhoto.loadTransferable(type: Data.self) {
                    inputState.updateImageFromPhotoData(data)
                }
            } catch {
                assertionFailure(error.localizedDescription)
            }
        }
    }
}
