//
//  MainApp.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 20.02.2026.
//

import SwiftUI

@main
struct App: SwiftUI.App {

    @State var modelSelection: ModelSelection?
    @State var appSettings = AppSettings()
    @State var showSettings = false

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                ModelSelectionView(modelSelection: $modelSelection)
                    .navigationTitle("MLX Vision")
                    .navigationDestination(item: $modelSelection) { modelSelection in
                        ModelRunnerView(modelSelection: modelSelection)
                            .navigationTitle(modelSelection.type.rawValue)
                            .navigationSubtitle(modelSelection.id)
                            #if targetEnvironment(macCatalyst)
                                .navigationBarTitleDisplayMode(.inline)
                            #endif
                    }
                    .toolbar {
                        ToolbarItem(placement: .topBarTrailing) {
                            Button {
                                showSettings = true
                            } label: {
                                Image(systemName: "gear")
                            }
                        }
                    }
                    .sheet(isPresented: $showSettings) {
                        AppSettingsView()
                    }
            }
            .environment(appSettings)
            .task {
                appSettings.applyCacheLimit()
            }
        }
    }
}
