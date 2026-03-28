//
//  AppSettingsView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 20.02.2026.
//

import SwiftUI
import Foundation

struct AppSettingsView: View {

    @Environment(AppSettings.self) private var appSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                @Bindable var appSettings = appSettings

                Section {
                    VStack {
                        LabeledContent("Cache Limit") {
                            Text(
                                Measurement(
                                    value: appSettings.cacheLimit,
                                    unit: UnitInformationStorage.bytes
                                )
                                .formatted(.byteCount(style: .memory))
                            )
                        }

                        Slider(
                            value: $appSettings.cacheLimit,
                            in: 0...AppSettings.deviceMemorySize
                        )
                    }
                } header: {
                    Text("GPU")
                } footer: {
                    Text("A larger cache limit slightly improves inference speed")
                }

                Section {
                    SecureField("Access Token", text: $appSettings.token)
                } header: {
                    Text("Hugging Face")
                } footer: {
                    Text("Set access token to download gated models")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}
